"""The prosody stage and its backends.

The two restrictions that define the stage are tested first: participant speech
only, and overlapping speech excluded. Both exist because a measurement taken
on the wrong voice, or across two voices at once, is worse than a missing one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tests.conftest import WINTER_FOLDER, place_fake_media
from tests.synth import generators as gen
from vc_multimodal.config import AppConfig, load_config
from vc_multimodal.contracts import feature_schema, validate
from vc_multimodal.features.prosody_math import (
    FEATURE_NAMES,
    SpanMeasures,
    voiced_frequencies,
)
from vc_multimodal.features.spans import Span
from vc_multimodal.features.turn_math import ROLE_PARTICIPANT, ROLE_PSYCHIATRIST
from vc_multimodal.io_utils import read_parquet, write_parquet
from vc_multimodal.paths import DataRoots, RawSession
from vc_multimodal.prosody import (
    OpenSmileBackend,
    ParselmouthBackend,
    ProsodyError,
    get_backend,
)
from vc_multimodal.roles import RoleMapping, write_role_mapping
from vc_multimodal.stages import prosody as stage
from vc_multimodal.stages import vad as vad_stage

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT = REPO_ROOT / "config" / "default.yaml"
ROLES = {"SPEAKER_00": ROLE_PSYCHIATRIST, "SPEAKER_01": ROLE_PARTICIPANT}


def _session(session_id: int = 28) -> RawSession:
    return RawSession(
        session_id=session_id,
        wave="winter",
        date_folder=WINTER_FOLDER,
        path=Path(f"/nowhere/{session_id}.mp4"),
    )


def spans(*pairs: tuple[float, float]) -> tuple[Span, ...]:
    return tuple(Span(start, end) for start, end in pairs)


# ---------------------------------------------------------------------------
# choosing what to measure
# ---------------------------------------------------------------------------
def test_only_participant_speech_is_measured():
    by_role = {
        ROLE_PARTICIPANT: spans((0.0, 5.0)),
        ROLE_PSYCHIATRIST: spans((10.0, 20.0)),
    }
    chosen, _ = stage.analysis_spans(by_role, exclude_overlap=True, min_analysis_s=0.3)
    assert chosen == spans((0.0, 5.0))


def test_overlapping_speech_is_excluded():
    """A pitch value measured across two voices describes neither."""
    by_role = {
        ROLE_PARTICIPANT: spans((0.0, 10.0)),
        ROLE_PSYCHIATRIST: spans((4.0, 6.0)),
    }
    chosen, excluded = stage.analysis_spans(by_role, exclude_overlap=True, min_analysis_s=0.3)
    assert chosen == spans((0.0, 4.0), (6.0, 10.0))
    assert excluded == pytest.approx(2.0)


def test_the_excluded_overlap_is_reported_even_when_kept():
    """So the QC column says how much overlap there was either way."""
    by_role = {
        ROLE_PARTICIPANT: spans((0.0, 10.0)),
        ROLE_PSYCHIATRIST: spans((4.0, 6.0)),
    }
    chosen, excluded = stage.analysis_spans(by_role, exclude_overlap=False, min_analysis_s=0.3)
    assert chosen == spans((0.0, 10.0))
    assert excluded == pytest.approx(2.0)


def test_fragments_too_short_to_measure_are_dropped():
    """A 100 ms fragment yields a pitch value, and it is noise."""
    by_role = {ROLE_PARTICIPANT: spans((0.0, 0.1), (1.0, 5.0))}
    chosen, _ = stage.analysis_spans(by_role, exclude_overlap=True, min_analysis_s=0.3)
    assert chosen == spans((1.0, 5.0))


def test_overlap_removal_can_leave_only_fragments():
    by_role = {
        ROLE_PARTICIPANT: spans((0.0, 5.0)),
        ROLE_PSYCHIATRIST: spans((0.2, 4.8)),
    }
    chosen, excluded = stage.analysis_spans(by_role, exclude_overlap=True, min_analysis_s=0.3)
    assert chosen == ()
    assert excluded == pytest.approx(4.6)


def test_nothing_is_measured_past_the_audio_that_exists():
    """Which matters for the recording whose file is truncated."""
    by_role = {ROLE_PARTICIPANT: spans((40.0, 400.0))}
    chosen, _ = stage.analysis_spans(
        by_role, exclude_overlap=True, min_analysis_s=0.3, extent=Span(0.0, 46.3)
    )
    assert chosen == spans((40.0, 46.3))


def test_no_participant_speech_means_nothing_to_measure():
    chosen, excluded = stage.analysis_spans(
        {ROLE_PSYCHIATRIST: spans((0.0, 10.0))}, exclude_overlap=True, min_analysis_s=0.3
    )
    assert chosen == ()
    assert excluded == 0.0


def test_an_unassigned_speaker_counts_as_another_voice_for_overlap():
    """Safer than ignoring it: an unknown voice still contaminates a measure."""
    by_role = {ROLE_PARTICIPANT: spans((0.0, 10.0)), "unknown": spans((4.0, 6.0))}
    chosen, excluded = stage.analysis_spans(by_role, exclude_overlap=True, min_analysis_s=0.3)
    assert chosen == spans((0.0, 4.0), (6.0, 10.0))
    assert excluded == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# measuring, and tolerating failures
# ---------------------------------------------------------------------------
class StubBackend(ParselmouthBackend):
    """Measures nothing, optionally failing on some spans."""

    def __init__(self, *, fail_every: int | None = None) -> None:
        self.calls = 0
        self.fail_every = fail_every

    def measure(self, samples: np.ndarray, sample_rate: int, *, config: Any) -> SpanMeasures:
        self.calls += 1
        if self.fail_every and self.calls % self.fail_every == 0:
            msg = "scripted failure"
            raise ProsodyError(msg)
        duration = samples.size / sample_rate
        n = max(2, int(duration * 100))
        return SpanMeasures(
            duration_s=duration,
            f0_hz=np.full(n, 150.0),
            intensity_db=np.tile([60.0, 70.0], n // 2),
            harmonicity_db=np.full(n, 12.0),
            intensity_frame_step_s=0.01,
            jitter_local=0.01,
            shimmer_local=0.04,
        )


def test_every_span_is_measured():
    samples = np.zeros(16000 * 20)
    backend = StubBackend()
    measures, failed = stage.measure_spans(
        samples, 16000, spans((0.0, 5.0), (6.0, 10.0)), backend=backend, config=load_config(DEFAULT)
    )
    assert backend.calls == 2
    assert len(measures) == 2
    assert failed == 0


def test_one_unmeasurable_span_does_not_cost_the_others():
    samples = np.zeros(16000 * 30)
    backend = StubBackend(fail_every=2)
    measures, failed = stage.measure_spans(
        samples,
        16000,
        spans((0.0, 5.0), (6.0, 10.0), (11.0, 15.0)),
        backend=backend,
        config=load_config(DEFAULT),
    )
    assert len(measures) == 2
    assert failed == 1


def test_a_span_beyond_the_audio_is_counted_as_failed():
    backend = StubBackend()
    measures, failed = stage.measure_spans(
        np.zeros(100), 16000, spans((50.0, 60.0)), backend=backend, config=load_config(DEFAULT)
    )
    assert measures == []
    assert failed == 1


# ---------------------------------------------------------------------------
# the feature row
# ---------------------------------------------------------------------------
def _row(**kwargs: Any) -> dict[str, object]:
    defaults: dict[str, Any] = {
        "measures": [
            SpanMeasures(
                duration_s=30.0,
                f0_hz=np.full(100, 150.0),
                intensity_db=np.tile([60.0, 70.0], 50),
                harmonicity_db=np.full(100, 12.0),
                intensity_frame_step_s=0.01,
                jitter_local=0.01,
                shimmer_local=0.04,
            )
        ],
        "mapping": RoleMapping(28, ROLES, "assigned"),
        "n_failed": 0,
        "overlap_excluded_s": 1.5,
        "had_participant_speech": True,
        "backend_name": "parselmouth",
    }
    defaults.update(kwargs)
    return stage.feature_row(_session(), **defaults)


def test_the_row_carries_every_feature_and_the_qc_columns():
    row = _row()
    assert set(FEATURE_NAMES) <= set(row)
    assert set(stage.QC_COLUMNS) <= set(row)
    assert row["qc__overlap_excluded_seconds"] == pytest.approx(1.5)
    assert row["qc__backend"] == "parselmouth"


def test_the_semitone_reference_is_recorded_as_qc_not_as_a_feature():
    """Absolute pitch as a feature would reintroduce the sex difference that
    semitone normalisation exists to remove."""
    row = _row()
    assert row["qc__f0_median_hz"] == pytest.approx(150.0)
    assert not any("median_hz" in name for name in FEATURE_NAMES)


def test_a_session_with_no_participant_speech_is_flagged():
    row = _row(measures=[], had_participant_speech=False)
    assert stage.FLAG_NO_PARTICIPANT in row["qc__flags"]
    assert row["prosody__f0_semitone_sd"] is None


def test_too_little_analysed_speech_is_flagged():
    row = _row(
        measures=[
            SpanMeasures(
                duration_s=2.0,
                f0_hz=np.full(10, 150.0),
                intensity_db=np.full(10, 70.0),
                harmonicity_db=np.full(10, 12.0),
                intensity_frame_step_s=0.01,
            )
        ]
    )
    assert stage.FLAG_TOO_LITTLE_SPEECH in row["qc__flags"]


def test_an_unvoiced_session_is_flagged():
    row = _row(
        measures=[
            SpanMeasures(
                duration_s=40.0,
                f0_hz=np.zeros(100),
                intensity_db=np.full(100, 70.0),
                harmonicity_db=np.full(100, 12.0),
                intensity_frame_step_s=0.01,
            )
        ]
    )
    assert stage.FLAG_NO_VOICED_FRAMES in row["qc__flags"]


def test_failed_spans_are_flagged():
    assert stage.FLAG_SPANS_FAILED in _row(n_failed=2)["qc__flags"]


def test_a_manual_role_mapping_is_flagged():
    row = _row(mapping=RoleMapping(28, ROLES, "manual"))
    assert stage.FLAG_MANUAL_ROLES in row["qc__flags"]


def test_a_healthy_session_has_no_flags():
    assert _row()["qc__flags"] == ""


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
def test_the_feature_table_follows_the_naming_convention():
    frame = stage.build_frame([_row()])
    validate(frame, feature_schema([*FEATURE_NAMES, *stage.QC_COLUMNS]))


def test_an_empty_feature_table_is_typed_and_valid():
    frame = stage.build_frame([])
    validate(frame, feature_schema([*FEATURE_NAMES, *stage.QC_COLUMNS]))
    assert str(frame["session_id"].dtype) == "int64"


def test_the_span_table_records_durations_without_the_contours():
    """The contours are large and nothing downstream needs them."""
    measure = SpanMeasures(
        duration_s=5.0,
        f0_hz=np.full(100, 150.0),
        intensity_db=np.full(100, 70.0),
        harmonicity_db=np.full(100, 12.0),
        intensity_frame_step_s=0.01,
        jitter_local=0.01,
        shimmer_local=0.04,
    )
    frame = stage.span_frame(28, spans((0.0, 5.0)), [measure])
    assert list(frame.columns) == [
        "session_id",
        "span_index",
        "start_s",
        "end_s",
        "duration_s",
        "n_pitch_frames",
        "n_voiced_frames",
        "jitter_local",
        "shimmer_local",
    ]
    assert frame.iloc[0]["n_voiced_frames"] == 100


# ---------------------------------------------------------------------------
# backends
# ---------------------------------------------------------------------------
def test_parselmouth_is_always_available():
    backend = ParselmouthBackend()
    assert backend.available()
    assert backend.unavailable_reason() == ""
    assert backend.version().startswith("parselmouth/")


def test_parselmouth_measures_a_voiced_signal():
    samples = gen.voiced_signal(3.0, f0=150.0)
    result = ParselmouthBackend().measure(samples, 16000, config=load_config(DEFAULT).prosody)

    assert result.duration_s == pytest.approx(3.0, abs=0.01)
    assert result.n_voiced > 0
    assert result.intensity_db.size > 0
    assert result.jitter_local is not None
    assert result.shimmer_local is not None


def test_parselmouth_tracks_the_pitch_it_was_given():
    samples = gen.voiced_signal(3.0, f0=200.0)
    result = ParselmouthBackend().measure(samples, 16000, config=load_config(DEFAULT).prosody)
    median = float(np.median(voiced_frequencies(result.f0_hz)))
    assert median == pytest.approx(200.0, rel=0.15)


def test_parselmouth_refuses_an_empty_span():
    with pytest.raises(ProsodyError, match="empty span"):
        ParselmouthBackend().measure(np.zeros(0), 16000, config=load_config(DEFAULT).prosody)


def test_a_span_too_short_for_voice_quality_yields_none_not_zero():
    """Praat needs several pitch periods; zero jitter would be a claim."""
    samples = gen.voiced_signal(0.05, f0=150.0)
    result = ParselmouthBackend().measure(samples, 16000, config=load_config(DEFAULT).prosody)
    assert result.jitter_local is None
    assert result.shimmer_local is None


def test_the_opensmile_backend_is_an_extension_point_not_an_implementation(
    default_config: AppConfig,
):
    backend = OpenSmileBackend(default_config.prosody.opensmile)
    assert not backend.available()
    reason = backend.unavailable_reason()
    assert "extension point" in reason
    assert "88 features" in reason
    assert "feature budget" in reason
    with pytest.raises(ProsodyError, match="extension point"):
        backend.measure(np.zeros(100), 16000, config=default_config.prosody)


def test_the_default_backend_is_parselmouth(default_config: AppConfig):
    assert isinstance(get_backend(default_config), ParselmouthBackend)


def test_enabling_opensmile_fails_with_an_explanation():
    config = load_config(DEFAULT, overrides={"prosody.opensmile.enabled": True})
    with pytest.raises(ProsodyError, match="extension point"):
        get_backend(config)


# ---------------------------------------------------------------------------
# running the stage
# ---------------------------------------------------------------------------
@pytest.fixture
def prepared(roots: DataRoots, raw_tree: Path, tmp_path: Path, ffmpeg_bin: str) -> Any:
    """A session with voiced audio, speech spans and a role assignment."""

    def factory(session_id: int = 28, *, turn_s: float = 6.0) -> AppConfig:
        session = gen.alternating_session(
            session_id, n_turns=8, turn_s=turn_s, gap_s=1.0, lead_in_s=1.0, duration=58.0
        )
        place_fake_media(roots.data, WINTER_FOLDER, [session_id])
        # Voiced audio, written where extract-audio would have put it.
        gen.write_wav(
            roots.work_path("audio", f"{session_id}.wav"),
            gen.voiced_session_waveform(session),
            session.sample_rate,
        )
        rows = [(session_id, u.speaker, u.start, u.end) for u in session.utterances]
        frame = pd.DataFrame(rows, columns=["session_id", "speaker", "start_s", "end_s"])
        frame["session_id"] = frame["session_id"].astype("int64")
        frame["speaker"] = frame["speaker"].astype("string")
        for column in ("start_s", "end_s"):
            frame[column] = frame[column].astype("float64")
        write_parquet(vad_stage.speech_path(roots, session_id), frame)
        write_role_mapping(roots.work, session_id, ROLES)
        return load_config(DEFAULT)

    return factory


@pytest.mark.slow
def test_the_stage_measures_and_writes_both_tables(roots: DataRoots, prepared: Any):
    config = prepared()

    result = stage.run(config, roots, workers=1)

    assert result.report.ok
    assert result.path.exists()
    assert stage.prosody_path(roots, 28).exists()
    validate(result.frame, feature_schema([*FEATURE_NAMES, *stage.QC_COLUMNS]))


@pytest.mark.slow
def test_real_measures_are_produced_for_voiced_audio(roots: DataRoots, prepared: Any):
    config = prepared()

    row = stage.run(config, roots, workers=1).frame.iloc[0]

    assert row["prosody__f0_semitone_sd"] is not None
    assert row["prosody__intensity_mean_db"] > 0
    assert row["prosody__voiced_fraction"] > 0
    assert row["qc__f0_median_hz"] > 0
    assert row["qc__n_spans_analysed"] > 0


@pytest.mark.slow
def test_only_the_participants_speech_is_analysed(roots: DataRoots, prepared: Any):
    """Four of the eight turns are the participant's, at six seconds each."""
    config = prepared(turn_s=6.0)

    row = stage.run(config, roots, workers=1).frame.iloc[0]

    assert row["qc__analysed_seconds"] == pytest.approx(24.0, abs=1.0)


@pytest.mark.slow
def test_the_per_span_table_stays_in_the_work_tree(roots: DataRoots, prepared: Any):
    config = prepared()
    stage.run(config, roots, workers=1)
    assert stage.prosody_path(roots, 28).is_relative_to(roots.work)
    spans_table = read_parquet(stage.prosody_path(roots, 28))
    assert len(spans_table) == 4


@pytest.mark.slow
def test_missing_speech_spans_point_at_the_vad_stage(
    roots: DataRoots, raw_tree: Path, default_config: AppConfig
):
    place_fake_media(roots.data, WINTER_FOLDER, [28])
    write_role_mapping(roots.work, 28, ROLES)

    result = stage.run(default_config, roots, workers=1)

    assert "vc vad" in result.report.failed[0].message


@pytest.mark.slow
def test_a_missing_role_mapping_stops_that_session(roots: DataRoots, prepared: Any):
    config = prepared()
    stage.prosody_path(roots, 28).unlink(missing_ok=True)
    (roots.work / "roles" / "28.json").unlink()

    result = stage.run(config, roots, workers=1)

    assert [o.session_id for o in result.report.failed] == [28]
    assert "vc assign-speakers" in result.report.failed[0].message


@pytest.mark.slow
def test_completed_sessions_are_skipped(roots: DataRoots, prepared: Any):
    config = prepared()
    stage.run(config, roots, workers=1)
    assert len(stage.run(config, roots, workers=1).report.skipped) == 1


@pytest.mark.slow
def test_force_recomputes(roots: DataRoots, prepared: Any):
    config = prepared()
    stage.run(config, roots, workers=1)
    assert len(stage.run(config, roots, workers=1, force=True).report.succeeded) == 1


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
def test_the_summary_reports_the_headline_measures():
    text = "\n".join(stage.summarise(stage.build_frame([_row()])))
    assert "F0 variability (semitones)" in text
    assert "jitter (local)" in text
    assert "overlapping speech excluded" in text


def test_the_summary_explains_the_semitone_reference():
    text = "\n".join(stage.summarise(stage.build_frame([_row()])))
    assert "each speaker's own median" in text
    assert "does not enter them" in text


def test_the_summary_of_nothing():
    assert stage.summarise(pd.DataFrame()) == ["no sessions were measured"]
