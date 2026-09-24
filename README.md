# soundAkira

**Turn videos into zero-shot TTS training data.**

Give soundAkira local videos, folders, direct media links or YouTube videos and playlists. It returns a dataset of clean, single-speaker, transcribed utterances. Every speaker has an ID that is the same across all sources, and every utterance comes with a **reference prompt clip** of the same voice for voice-cloning training.

```
video / URL / playlist
  └─ fetch ─ extract ─ enhance ─┬─ vad ──────┐
                                ├─ diarize ──┼─ segment ─┬─ score
                                └─ transcribe┘           └─ embed
                                                               │
                         build: global speakers → references → export
```

| Stage | What it does | Default backend |
|---|---|---|
| fetch | Downloads audio only for URLs (YouTube, 1000+ sites, direct links) | yt-dlp |
| extract | Picks the dialogue track: skips commentary and audio description, prefers your language, can isolate the 5.1 centre channel | ffmpeg |
| enhance | Removes music, effects and noise, keeping voices. Streams in chunks with crossfades, so any length works | BS-RoFormer ([audio-separator](https://github.com/nomadkaraoke/python-audio-separator)); Demucs and DeepFilterNet optional |
| vad | Finds speech regions | Silero VAD |
| diarize | Who spoke when, within one source | pyannote `speaker-diarization-community-1` |
| transcribe | Text with word timestamps, per-language routing | faster-whisper `large-v3`; NVIDIA Parakeet optional; WhisperX forced alignment optional |
| segment | Builds single-speaker utterances cut on word boundaries. Drops overlapping speech and splits at sentence ends | built in |
| score / embed | Quality metrics and speaker embeddings for every segment | signal stats (DNSMOS optional); pyannote WeSpeaker |
| build | Merges speakers across sources, keeps IDs stable, picks leak-free references, filters, exports | built in |

## Install

Requires Python 3.10+ and [ffmpeg](https://ffmpeg.org) on `PATH`. For the ML backends, Python 3.10–3.12 is the safest choice: that is where torch, pyannote and NeMo publish wheels first.

```bash
pip install -e ".[standard]"     # yt-dlp + RoFormer + Silero + pyannote + faster-whisper
soundakira doctor                # check ffmpeg, GPU, HF token and installed backends
```

For YouTube, yt-dlp needs a JavaScript runtime: install [deno](https://deno.com). YouTube usually blocks cloud and datacenter IPs with a "confirm you're not a bot" check. On those machines, set `fetch.cookies_file` to an [exported cookies file](https://github.com/yt-dlp/yt-dlp/wiki/Extractors#exporting-youtube-cookies), or download elsewhere and pass local files.

The pyannote models are gated. Accept the terms for
[speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) and
[wespeaker-voxceleb-resnet34-LM](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM),
then `export HF_TOKEN=hf_...`.

| Extra | Adds |
|---|---|
| `fetch` | yt-dlp (URLs, YouTube) |
| `separation` | audio-separator (RoFormer / MDX / UVR models). Install `audio-separator[gpu]` on CUDA |
| `demucs`, `denoise` | Demucs, DeepFilterNet |
| `vad`, `diarization`, `asr` | Silero, pyannote, faster-whisper |
| `parakeet`, `align` | NVIDIA NeMo Parakeet ASR, WhisperX forced alignment |
| `quality` | DNSMOS scorer |

## Quickstart

```bash
soundakira init config.yaml                      # fully commented config
soundakira process sources.txt -c config.yaml    # per-source stages; safe to Ctrl-C and re-run
soundakira build -c config.yaml                  # speakers + references + export
soundakira status -c config.yaml --errors        # progress and failures
```

### Inputs

You can mix any of these:

- a video or audio file, or a folder (scanned recursively)
- a URL: a YouTube video, playlist or channel, any [yt-dlp site](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md), or a direct media link
- a manifest (`.txt`, `.csv`, `.tsv`, `.jsonl`) listing any of the above, with an optional **language hint** per line:

```text
# sources.txt
https://www.youtube.com/playlist?list=PLxxxx   en
https://youtu.be/dQw4w9WgXcQ                   en
/data/movies/film.mkv                          hi
episodes/                                      en
```

Source IDs are deterministic: `yt-<video id>`, `url-<hash>`, or `<file-slug>-<content hash>`. Re-running the same inputs picks up where the last run stopped.

## Output

```
dataset/
  wavs/00012/<utt_id>.wav     target utterances (24 kHz mono, 15–30 s by default)
  refs/00012/<ref_id>.wav     reference prompts (4–12 s, same speaker, different clip)
  metadata.csv                one row per utterance
  metadata.jsonl              same, plus word timings relative to the clip
  speakers.csv                per-speaker totals, split, sources
  dataset.json                total speakers, hours, splits, languages, drop reasons, config
```

`metadata.csv` columns:

| Column | |
|---|---|
| `utt_id`, `audio_path`, `text`, `text_tagged`, `language`, `duration` | the utterance |
| `speaker_id`, `speaker_name` | global speaker (same person, same ID, across all sources) |
| `split` | `train`, or `test` for held-out speakers (unseen-voice evaluation) |
| `ref_id`, `ref_audio_path`, `ref_text`, `ref_duration`, `ref_source_id` | voice prompt |
| `source_id`, `source_uri`, `source_start`, `source_end`, `num_speakers_in_source` | provenance |
| `asr_confidence`, `speaker_similarity`, `speech_ratio`, `overlap_ratio`, `snr_est_db`, ... | every metric, so you can re-filter without re-running |

A reference is never the target clip and never overlaps it in time. By default it comes from a *different source* when possible, so the model learns the voice rather than the room.

## How it scales

- **Resumable and incremental.** Each stage stores a fingerprint of its settings, its model versions and its upstream results. Re-runs skip finished work. Changing a setting re-runs only that stage and the stages after it. `--force vad` re-runs a stage and everything downstream.
- **Stage-major.** Each model loads once per run, and only one stage's models sit on the GPU at a time.
- **Sharding.** Run one process per GPU or machine against a shared `work_dir`, then build once:
  ```bash
  for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$i soundakira process sources.txt -c config.yaml --shard $i/4 & done; wait
  soundakira build -c config.yaml
  ```
- **Constant memory.** Enhancement streams audio in chunks, and clips are read by seeking. A three-hour film uses the same memory as a three-minute clip.
- **Disk.** `storage.keep_download: false` and `storage.keep_source_audio: false` delete intermediates; pruned files are re-created only if needed.

## Speakers across sources

1. **Per source:** a speaker's centroid is a duration-weighted *medoid-initialised* mean of its segment embeddings. Segments that don't match their own speaker are flagged (`speaker_similarity`), which catches diarization mistakes.
2. **Across sources:** agglomerative clustering on cosine distance (`speakers.cluster_threshold`). It also re-merges one person that diarization split in two.
3. **Stable IDs:** `work/speakers/registry.json` maps clusters to IDs by shared members, then by centroid similarity. Adding new videos never renumbers existing speakers. Names you edit in the registry are kept.
4. **Anchors (optional):** put a few clips per known person in `anchors/<Name>/*.wav` and set `speakers.anchors_dir`. Matching clusters get that name and are merged.

Tune `cluster_threshold` for your embedder: too low splits one person into several IDs, too high merges different people. `speakers.csv` and `metadata.csv` carry the similarity values, so you can check the result.

## Languages

English works out of the box. Other languages go through the ASR **router**: each source goes to the best model for its language, using the manifest hint or automatic detection.

```yaml
asr:
  name: router
  params:
    default: en
    detect_with: en
    routes:
      en: {name: faster_whisper, params: {model: large-v3}}
      hi: {name: faster_whisper, params: {model: large-v3}}
      hi-en: {name: my_hinglish_asr}      # any plugin, see below
```

All text handling is script-aware: speaking-rate metrics count Devanagari matras, sentence splitting knows `।` and `॥`, and repetition checks work in any script. See [`configs/indic.yaml`](configs/indic.yaml) and the roadmap below.

## Extending

Every model-backed step is a **component** looked up by `(kind, name)`. The kinds are `enhancer`, `vad`, `diarizer`, `asr`, `aligner`, `embedder` and `scorer`. A new backend is one class:

```python
from pydantic import BaseModel
from soundakira.components.base import Transcriber
from soundakira.registry import register
from soundakira.types import Transcript, Word

@register("asr", "my_asr")
class MyASR(Transcriber):
    class Params(BaseModel):          # validated from the YAML `params:` block
        model: str = "org/model"

    def load(self):                   # import heavy deps and load weights here
        ...

    def transcribe(self, audio, sr, speech, language):
        return Transcript(language, None, words=[Word(" hello", 0.0, 0.4, 0.98)])
```

Then use it with `asr: {name: my_asr, params: {model: ...}}`. You can also ship it as a separate package using an entry point, with no fork needed:

```toml
[project.entry-points."soundakira.plugins"]
"asr.my_asr" = "my_pkg.asr:MyASR"
```

See [docs/extending.md](docs/extending.md) for every interface, tag-aware ASR, and adding a new stage.

## Roadmap

- [ ] **Tag-aware ASR** (`[laugh]`, `[breath]`, emotion). The data model already carries it: `Word(kind="event")` flows into `text_tagged`.
- [ ] Indic backends: IndicConformer with an MMS forced aligner for Hindi and other Indian languages, and a Hinglish (code-switched) model
- [ ] DNSMOS or other learned quality scores on by default
- [ ] Near-duplicate removal (recaps, intros, re-uploads)
- [ ] Review UI for merging and renaming speakers and rejecting clips

## Responsible use

Only process media you have the rights to use, and follow the terms of the sites you download from. Voice cloning data identifies real people: get consent where the law or ethics require it.

## License

Apache-2.0
