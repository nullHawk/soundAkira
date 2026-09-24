# Extending soundAkira

## Components

A component wraps one model. The registry creates it from a config
`{name, params}`. It loads its weights in `load()`, which runs once per
stage, not once per source, and frees them in `unload()`.

| Kind | Base class | Method | Used by stage |
|---|---|---|---|
| `enhancer` | `Enhancer` | `process(audio (C,T), sr) -> (C',T)` | enhance (chain, chunked) |
| `vad` | `VAD` | `detect(audio, sr) -> list[Span]` | vad |
| `diarizer` | `Diarizer` | `diarize(audio, sr) -> list[Turn]` | diarize |
| `asr` | `Transcriber` | `transcribe(audio, sr, speech, language) -> Transcript` | transcribe |
| `aligner` | `Aligner` | `align(transcript, audio, sr) -> Transcript` | transcribe |
| `embedder` | `SpeakerEmbedder` | `embed(clips, sr) -> (N, D)` | embed, build (anchors) |
| `scorer` | `Scorer` | `score(audio, sr) -> dict[str, float]` | score |

Rules:

- Import heavy dependencies inside `load()`. `import soundakira` must stay torch-free.
- Declare `sample_rate` if the model needs a specific rate. The stage resamples for you.
- Put configuration in a pydantic `Params` model with `extra="forbid"`, so typos fail at startup.
- If a code change alters a component's output, bump its `version`. That invalidates cached results.
- Name every model in `params`. `identity()` includes the params in stage fingerprints, so switching models re-runs the right stages.

### Registering

```python
from soundakira.registry import register

@register("scorer", "my_metric")
class MyMetric(Scorer): ...
```

Or from your own package, without changing this repo:

```toml
[project.entry-points."soundakira.plugins"]
"scorer.my_metric" = "my_pkg.metrics:MyMetric"
```

`soundakira components` lists everything that is available.

## Recipes

### Text-only ASR (e.g. IndicConformer)

Return `Transcript.segments` with text and `words=[]`, set
`provides_word_timestamps = False`, and configure an aligner:

```yaml
asr:
  name: indic_conformer
  aligner: {name: mms_aligner}
```

The transcribe stage refuses a transcript that has segments but no word timings, so a missing aligner fails loudly.

### Tag-aware ASR (non-verbal events)

Emit events as words with `kind="event"`:

```python
Word(" [laugh]", 3.2, 3.9, kind="event")
```

Segmentation keeps events out of `text` and puts them in `text_tagged`
(`"hello [laugh] world."`). Events never start or split an utterance; they attach to the speaker's current run.

### New quality metric

Any key a scorer returns becomes a metric column and a filter field:

```yaml
quality:
  scorers: [{name: signal}, {name: my_metric}]
export:
  filters:
    - {field: my_metric_score, min: 3.0, on_missing: drop}
```

## Adding a stage

Subclass `soundakira.pipeline.stages.Stage`:

```python
class LanguageIdStage(Stage):
    name = "lid"
    requires = ("enhance",)

    def build_components(self): return [self._make("lid", self.cfg.lid)]
    def params(self): return {...}                   # anything that changes the output
    def outputs(self, ws): return [ws.dir / "lid.json"]
    def run(self, ws): ...; return {"language": lang}  # stats stored in manifest.json
```

Then insert it into `STAGES` in dependency order. Fingerprints chain through `requires`, so stages downstream of a changed stage re-run automatically.

## Testing without models

`tests/fakes.py` registers fake diarizer, ASR and embedder components that decode synthetic tone "speech". `tests/test_e2e.py` runs the whole pipeline with them, including ffmpeg, caching, clustering, references and export, in about a second. Use the same pattern to test a new stage or a change to the dataset logic.
