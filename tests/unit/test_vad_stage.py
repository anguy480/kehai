"""The voice-activity stage.

Silero is trained on speech and correctly rejects the pure tones the synthetic
generators use, so the stage's own logic is tested against a deterministic stub
detector. One separate test runs the real model on a speech-like signal to prove
the wiring - sample rates, seconds conversion, configuration plumbing - is
right; it asserts only onset placement, because the model does not sustain on
synthetic input.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tests.conftest import WINTER_FOLDER
from tests.synth import generators as gen
from vc_multimodal.config import AppConfig, load_config
from vc_multimodal.contracts import SPEECH_SCHEMA, validate
from vc_multimodal.features.spans import Span, covered_duration
from vc_multimodal.io_utils import read_parquet
from vc_multimodal.paths import DataRoots, RawSession
from vc_multimodal.stages import diarize as diarize_stage
from vc_multimodal.stages import extract_audio as audio_stage
from vc_multimodal.stages import vad as stage

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT = REPO_ROOT / "config" / "default.yaml"


class StubDetector:
    """Returns pre-set spans, so the stage's arithmetic is what gets tested."""

    def __init__(self, spans: list[Span] | None = None) -> None:
        self.spans = spans or []
        self.calls: list[tuple[int, int]] = []

    def detect(self, samples: np.ndarray, sample_rate: int) -> tuple[Span, ...]:
        self.calls.append((int(samples.size), int(sample_rate)))
        return tuple(self.spans)


class ProportionalDetector:
    """Marks the first half of whatever it is given as speech.

    Lets `per_segment` mode be checked: each call sees a different slice, so the
    results differ per segment rather than being a fixed answer.
    """

    def __init__(self) -> None:
        self.calls = 0

    def detect(self, samples: np.ndarray, sample_rate: int) -> tuple[Span, ...]:
        self.calls += 1
        seconds = samples.size / sample_rate
        return (Span(0.0, seconds / 2.0),) if seconds > 0 else ()


# ---------------------------------------------------------------------------
# reading the audio this pipeline writes
# ---------------------------------------------------------------------------
def test_a_mono_wav_is_read_as_floats(tmp_path: Path):
    samples = np.array([0.0, 0.5, -0.5, 1.0], dtype=np.float64)
    path = gen.write_wav(tmp_path / "a.wav", samples, 16000)

    restored, sample_rate = stage.read_mono_wav(path)

    assert sample_rate == 16000
    assert restored.shape == samples.shape
    assert restored == pytest.approx(samples, abs=1e-4)


def test_a_stereo_wav_is_refused(tmp_path: Path):
    """Later stages assume mono; a stereo file here means something went wrong."""
    stereo = np.zeros((100, 2))
    path = gen.write_wav(tmp_path / "s.wav", stereo, 16000)
    with pytest.raises(ValueError, match="expected mono 16-bit"):
        stage.read_mono_wav(path)


# ---------------------------------------------------------------------------
# grouping and attribution
# ---------------------------------------------------------------------------
def test_segments_are_grouped_and_merged_per_speaker():
    frame = pd.DataFrame(
        {
            "speaker": ["SPEAKER_00", "SPEAKER_00", "SPEAKER_01"],
            "start_s": [0.0, 1.0, 5.0],
            "end_s": [1.0, 2.0, 6.0],
        }
    )
    grouped = stage.segments_by_speaker(frame)
    assert grouped["SPEAKER_00"] == (Span(0.0, 2.0),)
    assert grouped["SPEAKER_01"] == (Span(5.0, 6.0),)


def test_detected_speech_is_attributed_to_the_speaker_whose_segment_it_falls_in():
    """The whole point: diarization says who, the detector says when."""
    by_speaker = {"SPEAKER_00": [Span(0.0, 10.0)], "SPEAKER_01": [Span(10.0, 20.0)]}
    detected = [Span(2.0, 4.0), Span(12.0, 14.0)]

    result = {
        item.speaker: item.spans for item in stage.refine_by_intersection(detected, by_speaker)
    }

    assert result["SPEAKER_00"] == (Span(2.0, 4.0),)
    assert result["SPEAKER_01"] == (Span(12.0, 14.0),)


def test_silence_inside_a_segment_is_discarded():
    """A diarized segment spanning silence is exactly what this stage fixes."""
    by_speaker = {"SPEAKER_00": [Span(0.0, 60.0)]}
    detected = [Span(5.0, 10.0)]

    result = stage.refine_by_intersection(detected, by_speaker)

    assert covered_duration(result[0].spans) == pytest.approx(5.0)


def test_speech_outside_every_segment_is_dropped():
    by_speaker = {"SPEAKER_00": [Span(0.0, 5.0)]}
    result = stage.refine_by_intersection([Span(50.0, 60.0)], by_speaker)
    assert result[0].spans == ()


def test_speech_straddling_a_speaker_change_is_split_between_them():
    by_speaker = {"SPEAKER_00": [Span(0.0, 5.0)], "SPEAKER_01": [Span(5.0, 10.0)]}
    result = {
        i.speaker: i.spans for i in stage.refine_by_intersection([Span(4.0, 6.0)], by_speaker)
    }
    assert result["SPEAKER_00"] == (Span(4.0, 5.0),)
    assert result["SPEAKER_01"] == (Span(5.0, 6.0),)


def test_per_segment_mode_detects_inside_each_segment_separately():
    samples = np.zeros(16000 * 20)
    by_speaker = {"SPEAKER_00": [Span(0.0, 4.0), Span(10.0, 14.0)]}
    detector = ProportionalDetector()

    result = stage.refine_per_segment(samples, 16000, by_speaker, detector)

    assert detector.calls == 2
    # Each 4 s segment yields its own first half, shifted into recording time.
    assert result[0].spans == (Span(0.0, 2.0), Span(10.0, 12.0))


def test_per_segment_mode_skips_segments_beyond_the_audio():
    samples = np.zeros(16000 * 5)
    detector = ProportionalDetector()
    result = stage.refine_per_segment(samples, 16000, {"SPEAKER_00": [Span(50.0, 60.0)]}, detector)
    assert detector.calls == 0
    assert result[0].spans == ()


# ---------------------------------------------------------------------------
# the written table
# ---------------------------------------------------------------------------
def test_the_speech_table_carries_speakers_and_no_text():
    """Roles come later; transcripts never come here at all."""
    speech = [stage.SpeakerSpeech("SPEAKER_00", (Span(0.0, 1.0),))]
    frame = stage.speech_frame(28, speech)

    validate(frame, SPEECH_SCHEMA)
    assert list(frame.columns) == ["session_id", "speaker", "start_s", "end_s"]
    assert "text" not in frame.columns
    assert "role" not in frame.columns


def test_the_speech_table_is_ordered_by_time():
    speech = [
        stage.SpeakerSpeech("SPEAKER_01", (Span(5.0, 6.0),)),
        stage.SpeakerSpeech("SPEAKER_00", (Span(0.0, 1.0),)),
    ]
    frame = stage.speech_frame(28, speech)
    assert list(frame["start_s"]) == [0.0, 5.0]


def test_an_empty_speech_table_still_satisfies_the_contract():
    validate(stage.speech_frame(28, []), SPEECH_SCHEMA)


# ---------------------------------------------------------------------------
# QC
# ---------------------------------------------------------------------------
def _session(session_id: int = 28) -> RawSession:
    return RawSession(
        session_id=session_id,
        wave="winter",
        date_folder=WINTER_FOLDER,
        path=Path(f"/nowhere/{session_id}.mp4"),
    )


def test_the_retained_fraction_says_how_much_segment_time_was_silence():
    record = stage.qc_record(
        _session(),
        mode="intersect",
        segments={"SPEAKER_00": [Span(0.0, 100.0)]},
        speech=[stage.SpeakerSpeech("SPEAKER_00", (Span(0.0, 40.0),))],
    )
    assert record["segment_seconds"] == pytest.approx(100.0)
    assert record["speech_seconds"] == pytest.approx(40.0)
    assert record["retained_fraction"] == pytest.approx(0.4)


def test_a_session_with_no_detected_speech_is_flagged():
    record = stage.qc_record(
        _session(),
        mode="intersect",
        segments={"SPEAKER_00": [Span(0.0, 100.0)]},
        speech=[stage.SpeakerSpeech("SPEAKER_00", ())],
    )
    assert stage.FLAG_NO_SPEECH in record["flags"]


def test_segments_that_are_mostly_silence_are_flagged():
    record = stage.qc_record(
        _session(),
        mode="intersect",
        segments={"SPEAKER_00": [Span(0.0, 100.0)]},
        speech=[stage.SpeakerSpeech("SPEAKER_00", (Span(0.0, 10.0),))],
    )
    assert stage.FLAG_MOSTLY_SILENT in record["flags"]


def test_a_healthy_session_has_no_flags():
    record = stage.qc_record(
        _session(),
        mode="intersect",
        segments={"SPEAKER_00": [Span(0.0, 100.0)]},
        speech=[stage.SpeakerSpeech("SPEAKER_00", (Span(0.0, 85.0),))],
    )
    assert record["flags"] == ""


# ---------------------------------------------------------------------------
# running the stage
# ---------------------------------------------------------------------------
@pytest.fixture
def prepared(raw_tree: Path, roots: DataRoots, tmp_path: Path, ffmpeg_bin: str) -> Any:
    """Sessions with audio extracted and diarization imported."""
    scratch = tmp_path / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    import_dir = roots.work / "diarization"
    import_dir.mkdir(parents=True, exist_ok=True)

    def factory(*plan: tuple[int, str]) -> AppConfig:
        for session_id, folder in plan:
            session = gen.alternating_session(
                session_id, n_turns=8, turn_s=2.0, gap_s=0.5, duration=22.0
            )
            gen.write_session_mp4(
                raw_tree / folder / f"{session_id}.mp4",
                session,
                tmp_dir=scratch,
                stereo_layout="mono",
                ffmpeg=ffmpeg_bin,
            )
            gen.write_srt(import_dir / f"{session_id}.srt", session)
        config = load_config(DEFAULT, overrides={"diarization.import_dir": "diarization"})
        audio_stage.run(config, roots, workers=1)
        diarize_stage.run(config, roots, workers=1)
        return config

    return factory


@pytest.mark.slow
def test_the_stage_writes_speech_spans_and_a_qc_table(roots: DataRoots, prepared: Any):
    config = prepared((28, WINTER_FOLDER), (3, "January 17 2026"))
    detector = StubDetector([Span(0.0, 30.0)])

    result = stage.run(config, roots, workers=1, detector=detector)

    assert result.report.ok
    assert sorted(result.frame["session_id"]) == [3, 28]
    for session_id in (3, 28):
        assert stage.speech_path(roots, session_id).exists()
        validate(read_parquet(stage.speech_path(roots, session_id)), SPEECH_SCHEMA)


@pytest.mark.slow
def test_speech_spans_land_in_the_work_tree(roots: DataRoots, prepared: Any):
    config = prepared((28, WINTER_FOLDER))
    stage.run(config, roots, workers=1, detector=StubDetector([Span(0.0, 30.0)]))
    assert stage.speech_path(roots, 28).is_relative_to(roots.work)


@pytest.mark.slow
def test_detected_speech_is_confined_to_the_diarized_segments(roots: DataRoots, prepared: Any):
    """A detector claiming speech everywhere must not invent speaker time."""
    config = prepared((28, WINTER_FOLDER))
    stage.run(config, roots, workers=1, detector=StubDetector([Span(0.0, 1000.0)]))

    speech = read_parquet(stage.speech_path(roots, 28))
    segments = read_parquet(diarize_stage.segments_path(roots, 28))
    assert speech["end_s"].max() <= segments["end_s"].max() + 1e-6


@pytest.mark.slow
def test_nothing_extends_past_the_audio_that_decoded(roots: DataRoots, prepared: Any):
    """Which matters for the recording whose file is truncated."""
    config = prepared((28, WINTER_FOLDER))
    stage.run(config, roots, workers=1, detector=StubDetector([Span(0.0, 1000.0)]))

    samples, sample_rate = stage.read_mono_wav(audio_stage.audio_path(roots, 28))
    speech = read_parquet(stage.speech_path(roots, 28))
    assert speech["end_s"].max() <= len(samples) / sample_rate + 1e-6


@pytest.mark.slow
def test_per_segment_mode_is_selectable(roots: DataRoots, prepared: Any):
    prepared((28, WINTER_FOLDER))
    config = load_config(
        DEFAULT,
        overrides={"diarization.import_dir": "diarization", "vad.mode": "per_segment"},
    )
    detector = ProportionalDetector()

    result = stage.run(config, roots, workers=1, detector=detector)

    assert detector.calls > 1  # once per segment, not once per session
    assert result.frame.iloc[0]["mode"] == "per_segment"


@pytest.mark.slow
def test_missing_diarization_fails_that_session_with_a_pointer(roots: DataRoots, prepared: Any):
    config = prepared((28, WINTER_FOLDER), (3, "January 17 2026"))
    diarize_stage.segments_path(roots, 28).unlink()

    result = stage.run(config, roots, workers=1, detector=StubDetector([Span(0.0, 30.0)]))

    assert [o.session_id for o in result.report.failed] == [28]
    assert "vc diarize" in result.report.failed[0].message


@pytest.mark.slow
def test_missing_audio_fails_that_session_with_a_pointer(roots: DataRoots, prepared: Any):
    config = prepared((28, WINTER_FOLDER))
    audio_stage.audio_path(roots, 28).unlink()

    result = stage.run(config, roots, workers=1, detector=StubDetector([Span(0.0, 30.0)]))

    assert "vc extract-audio" in result.report.failed[0].message


@pytest.mark.slow
def test_completed_sessions_are_skipped_but_stay_in_the_table(roots: DataRoots, prepared: Any):
    config = prepared((28, WINTER_FOLDER))
    detector = StubDetector([Span(0.0, 30.0)])
    stage.run(config, roots, workers=1, detector=detector)

    second = stage.run(config, roots, workers=1, detector=detector)

    assert len(second.report.skipped) == 1
    assert len(second.frame) == 1


@pytest.mark.slow
def test_force_recomputes(roots: DataRoots, prepared: Any):
    config = prepared((28, WINTER_FOLDER))
    detector = StubDetector([Span(0.0, 30.0)])
    stage.run(config, roots, workers=1, detector=detector)
    assert (
        len(stage.run(config, roots, workers=1, force=True, detector=detector).report.succeeded)
        == 1
    )


# ---------------------------------------------------------------------------
# the real detector, on a speech-like signal
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_the_real_detector_finds_a_speech_onset_where_it_belongs(tmp_path: Path):
    """Proves the wiring: sample rate, seconds conversion, config plumbing.

    Asserts only the onset. Silero is trained on speech and does not sustain on
    a synthesised signal, so anything more would be testing the model rather
    than this code.
    """
    sample_rate = 16000
    silence = np.zeros(sample_rate)
    speech = gen.voiced_signal(3.0, sample_rate=sample_rate)
    audio = np.concatenate([silence, speech, silence])

    detector = stage.SileroDetector(load_config(DEFAULT).vad)
    spans = detector.detect(audio, sample_rate)

    assert spans, "the detector should find something in a voiced signal"
    assert spans[0].start == pytest.approx(1.0, abs=0.2)
    assert detector.version().startswith("silero-vad/")


@pytest.mark.slow
def test_the_real_detector_rejects_pure_tones(tmp_path: Path):
    """Why the stage tests use a stub: the tone generators cannot exercise it."""
    session = gen.alternating_session(28, n_turns=4, turn_s=2.0, gap_s=0.5, duration=12.0)
    tones = gen.session_waveform(session)

    spans = stage.SileroDetector(load_config(DEFAULT).vad).detect(tones, session.sample_rate)

    assert covered_duration(spans) < 1.0


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
def _qc_row(session_id: int, **overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "session_id": session_id,
        "wave": "winter",
        "mode": "intersect",
        "n_segments": 40,
        "n_speech_spans": 120,
        "segment_seconds": 600.0,
        "speech_seconds": 420.0,
        "retained_fraction": 0.7,
        "n_speakers": 2,
        "flags": "",
    }
    row.update(overrides)
    return row


def test_the_summary_explains_why_a_low_retained_fraction_is_expected():
    text = "\n".join(stage.summarise(stage.build_frame([_qc_row(1)])))
    assert "share of diarized segment time that is actually speech" in text
    assert "diarized segments span silence" in text


def test_the_summary_lists_flagged_sessions():
    frame = stage.build_frame([_qc_row(1), _qc_row(2, flags=stage.FLAG_NO_SPEECH)])
    text = "\n".join(stage.summarise(frame))
    assert f"{stage.FLAG_NO_SPEECH}: 1 session(s) [2]" in text


def test_the_summary_of_nothing():
    assert stage.summarise(pd.DataFrame()) == ["no sessions were processed"]


def test_an_empty_qc_table_is_typed():
    frame = stage.build_frame([])
    assert str(frame["session_id"].dtype) == "int64"
    assert frame.empty
