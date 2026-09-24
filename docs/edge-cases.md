# Edge cases: what's handled and what's open

## Input and media

| Case | Handling |
|---|---|
| Multiple audio tracks (dubs, commentary, audio description) | `extract` skips tracks flagged as commentary or visual-impaired, or titled "commentary"/"description". It prefers the source's language hint, then `prefer_languages`, then the default track. `extract.stream_index` forces a track. |
| 5.1 / 7.1 mixes | `extract.channels: center` isolates the front-centre dialogue channel before separation. It falls back to a downmix when there is no centre channel. |
| Non-zero stream start times, VFR video | Only the audio stream is decoded, so all timestamps are relative to that audio. |
| No audio / near-empty audio | That source fails at `extract` with a clear error. Other sources continue. |
| Playlists, channels | Expanded with yt-dlp flat extraction. Each video becomes its own source. |
| YouTube bot check on cloud/datacenter IPs | The fetch fails with an actionable message: set `fetch.cookies_file`, or download elsewhere and pass local files. Other sources continue. |
| Same file listed twice, or under two paths | Local IDs come from a content fingerprint, so duplicates collapse to one source. |
| Interrupted runs, crashes mid-write | Every artifact is written atomically (temp file + rename). A stage counts as done only after its outputs exist. |
| Very long sources (3 h+) | Enhancement streams in chunks with crossfades; clip reads seek. Memory stays constant. |

## Audio content

| Case | Handling |
|---|---|
| Music, SFX, background noise | RoFormer vocal isolation. Optionally chain DeepFilterNet for residual noise. |
| Separation artifacts hurting ASR | `enhance.analysis_source: original` runs VAD, diarization and ASR on the untouched mix, while clips are still cut from the clean audio. |
| Overlapping speech | Words inside overlap regions break utterances, so no clip contains two voices. `overlap_ratio` is also a filter. |
| Other speakers' non-verbal sounds at clip edges | Padding never extends into another speaker's diarization turn. |
| Singing | Partially: the vocal stem keeps singing. `chars_per_sec` and `asr_confidence` filters remove most of it. |
| Clipping | `clip_ratio` metric and filter. |

## Transcription

| Case | Handling |
|---|---|
| Whisper drifting into lowercase, unpunctuated text | `condition_on_previous_text: true` (default). On podcast audio it cut unpunctuated segments from 10–72% to 0–5% with no loss of words. A punctuated `initial_prompt` fixes the punctuation but silently drops 7–19% of words, so it is not used. |
| Whisper hallucination on silence or music | faster-whisper's internal VAD, `condition_on_previous_text: false`, `hallucination_silence_threshold`, and compression-ratio and log-prob fallbacks. At export: `repetition_ratio`, `speech_ratio` (text over non-speech) and `asr_confidence` filters. |
| Word timestamps that swallow the preceding silence | Words are trimmed to VAD speech (`segmentation.refine_with_vad`). An aligner can also be configured. |
| Backends without word confidence (Parakeet) | Confidence metrics are absent; filters with `on_missing: keep` skip them. |
| Backends without word timestamps | Transcribe fails with a message asking for `asr.aligner`. It never guesses. |
| Unspaced scripts (zh/ja/th) | `Word.text` carries its own leading whitespace, so joining is always correct. |
| Devanagari punctuation (। ॥) and matras | Sentence detection and speaking-rate metrics are Unicode-category based. |

## Segmentation

| Case | Handling |
|---|---|
| Long monologues | Split at sentence ends, then clauses, then long pauses. The splitter prefers pieces of at least `preferred_min_duration`. |
| Most turns shorter than 15 s | Short segments are not wasted: they form the reference-prompt pool. `dataset.json` reports the drop reasons so you can tune the length window. |
| Mid-word cuts | Cuts only happen on word boundaries, with padding bounded by the neighbouring words. |
| Diarization boundaries a few hundred ms off ("…apps? Multiple \| reasons.") | The speaker change is snapped to the best pause or sentence end *inside the diarization's transition zone* (`snap_speaker_changes`). An unconstrained search moved changes the wrong way. |
| Clips starting or ending mid-sentence | Fragments of up to `trim_to_sentence` seconds are trimmed. The rest are flagged with `sentence_start` / `sentence_end`, which can be used as filters. |
| Words just outside a diarization turn | Attached to the nearest turn within `speaker_max_distance`. Otherwise they are a hard break. |

## Speakers

| Case | Handling |
|---|---|
| Local labels are meaningless across sources | Cross-source clustering of per-source centroids gives global IDs. |
| Diarization splits one person in two | Clustering re-merges them (there is no cannot-link constraint within a source). |
| Diarization mislabels some segments | A medoid-initialised centroid with trimming; `speaker_similarity` exposes the outliers, and the export filter drops them. |
| Speakers with very little audio | Excluded when they have less than `min_local_duration` of consistent speech. |
| IDs changing when videos are added | The registry matches new clusters to old IDs, first by shared members, then by centroid similarity. |
| Known people | Anchor clips name the clusters and merge them. |

## References

| Case | Handling |
|---|---|
| Leakage (the prompt is the target, or overlaps it) | Always excluded. |
| Prompt and target from the same recording | Another source is preferred (`prefer_other_source`). |
| Speaker has only long turns | Prompts are derived as word-aligned prefixes of long segments, ending at a sentence boundary where possible. |
| One clip used as the prompt for everything | Deterministic spreading across the top-k ranked candidates. |
| No valid prompt | The target is dropped (`references.require`), or kept with an empty reference. |

## Known limitations and open work

- **`cluster` diarizer (the no-gated-models fallback)** has no overlap detection, and its boundaries are only accurate to about half a hop. Snapping and trimming compensate, but pyannote is preferred when available.
- **Speaker count per source is unknown.** pyannote estimates it. For sources with known casts, pass `diarization.params.min_speakers/max_speakers`.
- **Cluster thresholds depend on the embedder.** Calibrate `cluster_threshold` on a few labelled sources before a large run.
- **Code-switched speech (Hinglish).** Whisper `large-v3` tends to output a single script or translate. It needs a dedicated model behind the router.
- **Singing, crowd speech, and TV or phone audio inside a scene** are only partly filtered.
- **Emotional extremes** (shouting, whispering) can be split into separate diarization speakers. Clustering usually re-merges them, but not always.
- **No near-duplicate detection yet** (recaps, cold opens, re-uploads).
