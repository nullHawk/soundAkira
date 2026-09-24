import pytest
import yaml
from pydantic import ValidationError

from soundakira import registry
from soundakira.audio.ffmpeg import (
    AudioStream,
    build_extract_command,
    parse_audio_streams,
    select_audio_stream,
)
from soundakira.components.base import ComponentContext, Scorer
from soundakira.config import PipelineConfig, default_config_text, load_config
from soundakira.sources.resolve import looks_like_playlist, resolve_inputs, youtube_video_id


def test_packaged_default_yaml_matches_model_defaults():
    from_yaml = PipelineConfig.model_validate(yaml.safe_load(default_config_text()))
    assert from_yaml == PipelineConfig()


def test_overrides_env_and_typos(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_TOKEN", "abc")
    p = tmp_path / "c.yaml"
    p.write_text("hf_token: ${MY_TOKEN}\nexport:\n  sample_rate: 22050\n")
    cfg = load_config(p, ["export.min_duration=10", "asr.params={model: large-v3-turbo}"])
    assert cfg.hf_token == "abc" and cfg.export.sample_rate == 22050
    assert cfg.export.min_duration == 10 and cfg.asr.params["model"] == "large-v3-turbo"
    with pytest.raises(ValidationError):
        load_config(None, ["export.sample_rat=1"])
    with pytest.raises(ValidationError):
        load_config(None, ["export.max_duration=60"])  # longer than segmentation allows


def test_registry_plugins_and_param_validation():
    @registry.register("scorer", "const")
    class Const(Scorer):
        def score(self, audio, sr):
            return {"x": 1.0}

    c = registry.create("scorer", "const", {}, ComponentContext())
    assert c.name == "const" and "const" in registry.available("scorer")
    with pytest.raises(registry.ComponentError, match="no scorer named"):
        registry.resolve("scorer", "nope")
    with pytest.raises(registry.ComponentError, match="invalid params"):
        registry.create("vad", "silero", {"thresold": 0.3}, ComponentContext())


@pytest.mark.parametrize(
    "url,vid",
    [
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=10", "dQw4w9WgXcQ"),
        ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://m.youtube.com/shorts/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://example.com/video.mp4", None),
    ],
)
def test_youtube_ids(url, vid):
    assert youtube_video_id(url) == vid


def test_playlist_detection():
    assert looks_like_playlist("https://www.youtube.com/playlist?list=PL123")
    assert looks_like_playlist("https://www.youtube.com/@somechannel")
    assert not looks_like_playlist("https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PL123")


def test_resolve_manifest_with_language_hints(tmp_path):
    media = tmp_path / "a.mp4"
    media.write_bytes(b"x" * 100)
    manifest = tmp_path / "list.txt"
    manifest.write_text(
        "# comment\n"
        "https://youtu.be/dQw4w9WgXcQ hindi\n"
        "a.mp4\n"
        "https://youtu.be/dQw4w9WgXcQ\n"  # duplicate -> deduped
    )
    sources = resolve_inputs([str(manifest)], default_language="en")
    by_kind = {s.kind: s for s in sources}
    assert len(sources) == 2
    assert by_kind["url"].source_id == "yt-dQw4w9WgXcQ" and by_kind["url"].language == "hi"
    assert by_kind["local"].language == "en" and by_kind["local"].source_id.startswith("a-")
    # IDs are deterministic
    assert resolve_inputs([str(manifest)], "en") == sources


def _stream(i, **kw):
    base = dict(
        index=i,
        codec="aac",
        channels=2,
        channel_layout="stereo",
        sample_rate=48000,
        language=None,
        title=None,
        is_default=False,
        is_commentary=False,
        is_audio_description=False,
    )
    return AudioStream(**{**base, **kw})


def test_stream_selection_skips_commentary_and_prefers_language():
    streams = [
        _stream(0, language="eng", is_default=True, title="Director's Commentary"),
        _stream(1, language="hin", is_default=True),
        _stream(2, language="eng", channels=6, channel_layout="5.1"),
        _stream(3, language="eng", is_audio_description=True),
    ]
    assert select_audio_stream(streams, ["eng"], ["commentary"]).index == 2
    assert select_audio_stream(streams, ["hin", "eng"], ["commentary"]).index == 1
    assert select_audio_stream(streams, [], [], explicit_index=3).index == 3


def test_center_channel_extraction_command(tmp_path):
    cmd = build_extract_command(
        tmp_path / "in.mkv",
        tmp_path / "o.flac",
        _stream(1, channels=6, channel_layout="5.1(side)"),
        44100,
        "center",
    )
    assert "pan=mono|c0=c2" in cmd and cmd[cmd.index("-map") + 1] == "0:a:1"
    stereo = build_extract_command(
        tmp_path / "in.mkv", tmp_path / "o.flac", _stream(0), 44100, "center"
    )
    assert not any("pan=" in c for c in stereo)  # no centre channel in stereo: plain downmix


def test_parse_audio_streams():
    probe = {
        "streams": [
            {"codec_type": "video"},
            {
                "codec_type": "audio",
                "codec_name": "ac3",
                "channels": 6,
                "channel_layout": "5.1",
                "tags": {"language": "ENG", "title": "Main"},
                "disposition": {"default": 1},
            },
        ]
    }
    [s] = parse_audio_streams(probe)
    assert s.index == 0 and s.language == "eng" and s.is_default and s.channels == 6
