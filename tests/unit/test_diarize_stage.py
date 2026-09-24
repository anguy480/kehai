"""The diarize stage and its backends.

The import backend is the preferred path, so it is tested against
whisper-diarization-shaped files including the ones that will not match. The
pyannote backend is tested through an injected fake pipeline, so nothing here
needs the optional extra, a Hugging Face token or a model download.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from tests.conftest import SUMMER_FOLDER, WINTER_FOLDER, place_fake_media
from tests.synth import generators as gen
from vc_multimodal.config import AppConfig, ConfigError, load_config
from vc_multimodal.contracts import DIARIZATION_QC_SCHEMA, validate
from vc_multimodal.diarization import (
    DiarizationError,
    ImportBackend,
    PyannoteBackend,
    Segment,
    get_backend,
    scan_import_dir,
)
from vc_multimodal.diarization.import_backend import find_file, parse_file
from vc_multimodal.diarization.pyannote_backend import segments_from_annotation
from vc_multimodal.paths import DataRoots, RawSession
from vc_multimodal.stages import diarize as stage

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT = REPO_ROOT / "config" / "default.yaml"
PATTERNS = ("{session_id}.srt", "{session_id}.txt", "{session_id}/{session_id}.srt")


@pytest.fixture
def import_dir(roots: DataRoots) -> Path:
    """The folder the professor's diarization output would be dropped into."""
    target = roots.work / "diarization"
    target.mkdir(parents=True, exist_ok=True)
    return target


def write_srt_for(import_dir: Path, session_id: int, **kwargs: Any) -> Path:
    """Write a whisper-diarization style SRT for one session."""
    session = gen.alternating_session(
        session_id, n_turns=kwargs.pop("n_turns", 8), turn_s=1.5, gap_s=0.5, duration=18.0
    )
    return gen.write_srt(import_dir / f"{session_id}.srt", session, **kwargs)


def config_with_import(**overrides: Any) -> AppConfig:
    """The shipped config, pointed at the import directory."""
    base: dict[str, Any] = {"diarization.import_dir": "diarization"}
    base.update(overrides)
    return load_config(DEFAULT, overrides=base)


# ---------------------------------------------------------------------------
# file matching, and reporting what could not be placed
# ---------------------------------------------------------------------------
def test_a_session_is_matched_to_its_file(import_dir: Path):
    write_srt_for(import_dir, 28)
    assert find_file(import_dir, 28, PATTERNS) == import_dir / "28.srt"


def test_patterns_are_tried_in_order(import_dir: Path):
    """A preferred format can be listed first."""
    (import_dir / "28.txt").write_text("Speaker 0: x\n", encoding="utf-8")
    write_srt_for(import_dir, 28)
    assert find_file(import_dir, 28, PATTERNS).suffix == ".srt"


def test_a_per_session_subdirectory_layout_is_supported(import_dir: Path):
    nested = import_dir / "28"
    nested.mkdir()
    write_srt_for(nested, 28)
    assert find_file(import_dir, 28, PATTERNS) == nested / "28.srt"


def test_an_unmatched_session_is_reported(import_dir: Path):
    write_srt_for(import_dir, 28)
    scan = scan_import_dir(import_dir, [28, 3], PATTERNS)
    assert scan.missing == (3,)
    assert not scan.ok


def test_files_that_match_no_session_are_reported(import_dir: Path):
    """A name that does not match is far likelier to be a naming difference."""
    write_srt_for(import_dir, 28)
    (import_dir / "interview_28_final.srt").write_text("x", encoding="utf-8")
    (import_dir / "session-3.srt").write_text("x", encoding="utf-8")

    scan = scan_import_dir(import_dir, [28], PATTERNS)

    assert set(scan.unmatched) == {"interview_28_final.srt", "session-3.srt"}
    assert not scan.ok
    report = "\n".join(scan.report_lines())
    assert "matched no session" in report
    assert "import_patterns" in report
    assert "interview_28_final.srt" in report


def test_timeless_files_are_reported_separately(import_dir: Path):
    write_srt_for(import_dir, 28)
    (import_dir / "28_speaker.txt").write_text("Speaker 0: x\n", encoding="utf-8")

    scan = scan_import_dir(import_dir, [28], PATTERNS)

    assert scan.timeless == ("28_speaker.txt",)
    assert "no timestamps" in "\n".join(scan.report_lines())


def test_dotfiles_are_ignored(import_dir: Path):
    write_srt_for(import_dir, 28)
    (import_dir / ".DS_Store").write_bytes(b"x")
    assert scan_import_dir(import_dir, [28], PATTERNS).ok


def test_a_clean_directory_reports_ok(import_dir: Path):
    for session_id in (28, 3):
        write_srt_for(import_dir, session_id)
    scan = scan_import_dir(import_dir, [28, 3], PATTERNS)
    assert scan.ok
    assert len(scan.matched) == 2


def test_a_missing_import_directory_says_where_it_should_live(roots: DataRoots):
    with pytest.raises(DiarizationError, match="not a directory"):
        scan_import_dir(roots.work / "absent", [28], PATTERNS)
    with pytest.raises(DiarizationError, match=re.escape("$VC_WORK_ROOT")):
        scan_import_dir(roots.work / "absent", [28], PATTERNS)


# ---------------------------------------------------------------------------
# parsing a located file
# ---------------------------------------------------------------------------
def test_a_timeless_file_explains_what_to_supply_instead(import_dir: Path):
    path = import_dir / "28.txt"
    path.write_text("Speaker 0: x\n", encoding="utf-8")
    with pytest.raises(DiarizationError, match="no timestamps"):
        parse_file(path)
    with pytest.raises(DiarizationError, match=re.escape(".srt")):
        parse_file(path)


def test_an_unrecognised_extension_is_refused(import_dir: Path):
    path = import_dir / "28.docx"
    path.write_text("x", encoding="utf-8")
    with pytest.raises(DiarizationError, match="unrecognised diarization format"):
        parse_file(path)


def test_an_rttm_file_is_parsed(import_dir: Path):
    path = import_dir / "28.rttm"
    path.write_text(
        "SPEAKER 28 1 0.5 1.5 <NA> <NA> SPEAKER_00 <NA> <NA>\n"
        "SPEAKER 28 1 2.5 1.5 <NA> <NA> SPEAKER_01 <NA> <NA>\n",
        encoding="utf-8",
    )
    segments = parse_file(path)
    assert len(segments) == 2
    assert all(segment.text is None for segment in segments)


def test_invalid_utf8_is_replaced_rather_than_fatal(import_dir: Path):
    """Transcripts from Japanese tooling are occasionally not UTF-8."""
    path = import_dir / "28.srt"
    path.write_bytes(b"1\n00:00:00,000 --> 00:00:01,000\nSpeaker 0: \xff\xfe bad bytes\n")
    assert len(parse_file(path)) == 1


# ---------------------------------------------------------------------------
# the backend
# ---------------------------------------------------------------------------
def _session(session_id: int = 28, wave: str = "winter") -> RawSession:
    return RawSession(
        session_id=session_id,
        wave=wave,
        date_folder=WINTER_FOLDER,
        path=Path(f"/nowhere/{session_id}.mp4"),
    )


def test_the_import_backend_reads_a_session(import_dir: Path):
    write_srt_for(import_dir, 28)
    backend = ImportBackend(import_dir, PATTERNS)

    segments = backend.segments(_session(28))

    assert backend.available()
    assert len(segments) == 8
    assert {segment.speaker for segment in segments} == {"SPEAKER_00", "SPEAKER_01"}


def test_the_import_backend_names_what_it_looked_for(import_dir: Path):
    backend = ImportBackend(import_dir, PATTERNS)
    with pytest.raises(DiarizationError, match="no diarization file for session 28"):
        backend.segments(_session(28))
    with pytest.raises(DiarizationError, match=re.escape("28.srt")):
        backend.segments(_session(28))


def test_an_unavailable_import_backend_explains_itself(roots: DataRoots):
    backend = ImportBackend(roots.work / "absent", PATTERNS)
    assert not backend.available()
    assert "import_dir" in backend.unavailable_reason()
    with pytest.raises(DiarizationError, match="not a directory"):
        backend.segments(_session(28))


def test_the_backend_records_where_it_read_from(import_dir: Path):
    assert ImportBackend(import_dir, PATTERNS).version() == "import/diarization"


def test_keep_text_is_honoured_by_the_backend(import_dir: Path):
    write_srt_for(import_dir, 28)
    with_text = ImportBackend(import_dir, PATTERNS, keep_text=True).segments(_session(28))
    without = ImportBackend(import_dir, PATTERNS, keep_text=False).segments(_session(28))
    assert all(segment.text for segment in with_text)
    assert all(segment.text is None for segment in without)


# ---------------------------------------------------------------------------
# backend selection
# ---------------------------------------------------------------------------
def test_the_configured_backend_is_built(roots: DataRoots, import_dir: Path):
    backend = get_backend(config_with_import(), roots)
    assert isinstance(backend, ImportBackend)
    assert backend.import_dir == import_dir


def test_import_without_a_directory_is_a_configuration_error(roots: DataRoots):
    """`vc inventory` must still load a config that has not settled this yet."""
    config = load_config(DEFAULT, overrides={"diarization.import_dir": None})
    assert config.diarization.import_dir is None
    with pytest.raises(ConfigError, match="import_dir"):
        get_backend(config, roots)


def test_the_pyannote_backend_can_be_selected(roots: DataRoots):
    config = load_config(DEFAULT, overrides={"diarization.backend": "pyannote"})
    assert isinstance(get_backend(config, roots), PyannoteBackend)


def test_import_paths_are_resolved_under_the_work_root(roots: DataRoots, import_dir: Path):
    """So no absolute path to clinical data is ever committed."""
    backend = get_backend(config_with_import(), roots)
    assert isinstance(backend, ImportBackend)
    assert backend.import_dir.is_relative_to(roots.work)


# ---------------------------------------------------------------------------
# the pyannote fallback, without pyannote installed
# ---------------------------------------------------------------------------
@dataclass
class _Turn:
    start: float
    end: float


class _FakeAnnotation:
    """Quacks like a pyannote Annotation."""

    def __init__(self, tracks: list[tuple[_Turn, str, str]]) -> None:
        self._tracks = tracks

    def itertracks(self, *, yield_label: bool = False) -> list[tuple[_Turn, str, str]]:
        assert yield_label
        return self._tracks


class _FakePipeline:
    def __init__(self, annotation: _FakeAnnotation, *, fail: bool = False) -> None:
        self.annotation = annotation
        self.fail = fail
        self.calls: list[dict[str, Any]] = []

    def __call__(self, path: str, **kwargs: Any) -> _FakeAnnotation:
        self.calls.append({"path": path, **kwargs})
        if self.fail:
            msg = "pipeline exploded"
            raise RuntimeError(msg)
        return self.annotation


def test_an_annotation_becomes_normalised_segments():
    annotation = _FakeAnnotation(
        [
            (_Turn(0.5, 2.0), "A", "SPEAKER_01"),
            (_Turn(2.5, 4.0), "B", "SPEAKER_00"),
        ]
    )
    segments = segments_from_annotation(annotation)
    assert [segment.speaker for segment in segments] == ["SPEAKER_01", "SPEAKER_00"]
    assert segments[0].start == pytest.approx(0.5)
    # pyannote carries no transcript, which is why it is the safer source.
    assert all(segment.text is None for segment in segments)


def test_pyannote_labels_are_normalised_too():
    annotation = _FakeAnnotation([(_Turn(0.0, 1.0), "A", "speaker_3")])
    assert segments_from_annotation(annotation)[0].speaker == "SPEAKER_03"


def test_vanishingly_short_turns_are_dropped():
    annotation = _FakeAnnotation(
        [(_Turn(0.0, 0.001), "A", "SPEAKER_00"), (_Turn(1.0, 2.0), "B", "SPEAKER_00")]
    )
    assert len(segments_from_annotation(annotation)) == 1


def test_an_annotation_with_nothing_usable_is_an_error():
    with pytest.raises(DiarizationError, match="no speech segments"):
        segments_from_annotation(_FakeAnnotation([]))


def test_an_injected_pipeline_makes_the_backend_available(default_config: AppConfig):
    pipeline = _FakePipeline(_FakeAnnotation([(_Turn(0.0, 1.0), "A", "SPEAKER_00")]))
    backend = PyannoteBackend(default_config.diarization.pyannote, pipeline=pipeline)

    assert backend.available()
    assert backend.unavailable_reason() == ""
    segments = backend.segments(_session(28))

    assert len(segments) == 1
    assert pipeline.calls[0]["num_speakers"] == 2


def test_pyannote_records_the_model_and_revision(default_config: AppConfig):
    backend = PyannoteBackend(default_config.diarization.pyannote)
    version = backend.version()
    assert "pyannote/speaker-diarization-3.1" in version
    # An unpinned model is recorded as unpinned rather than passing for pinned.
    assert "unpinned" in version


def test_a_pinned_revision_is_recorded(default_config: AppConfig):
    config = load_config(DEFAULT, overrides={"diarization.pyannote.revision": "abc123"})
    assert "@abc123" in PyannoteBackend(config.diarization.pyannote).version()


def test_pyannote_without_a_token_explains_itself(
    default_config: AppConfig, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    backend = PyannoteBackend(default_config.diarization.pyannote)
    reason = backend.unavailable_reason()
    assert "HF_TOKEN" in reason or "not installed" in reason
    assert not backend.available()


def test_a_pipeline_failure_names_the_session(default_config: AppConfig):
    pipeline = _FakePipeline(_FakeAnnotation([]), fail=True)
    backend = PyannoteBackend(default_config.diarization.pyannote, pipeline=pipeline)
    with pytest.raises(DiarizationError, match="pyannote failed on session 28"):
        backend.segments(_session(28))


# ---------------------------------------------------------------------------
# the stage
# ---------------------------------------------------------------------------
@pytest.fixture
def cohort(raw_tree: Path, import_dir: Path, tmp_path: Path, ffmpeg_bin: str) -> list[int]:
    """Three recordings with matching diarization output."""
    scratch = tmp_path / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    plan = [(28, WINTER_FOLDER), (3, "January 17 2026"), (210, SUMMER_FOLDER)]
    for session_id, folder in plan:
        session = gen.alternating_session(
            session_id, n_turns=8, turn_s=1.5, gap_s=0.5, duration=18.0
        )
        gen.write_session_mp4(
            raw_tree / folder / f"{session_id}.mp4",
            session,
            tmp_dir=scratch,
            stereo_layout="mono",
            ffmpeg=ffmpeg_bin,
        )
        gen.write_srt(import_dir / f"{session_id}.srt", session)
    return [3, 28, 210]


@pytest.mark.slow
def test_the_stage_writes_segments_and_a_qc_table(roots: DataRoots, cohort: list[int]):
    result = stage.run(config_with_import(), roots, workers=1)

    assert result.report.ok
    assert sorted(result.frame["session_id"]) == cohort
    validate(result.frame, DIARIZATION_QC_SCHEMA)
    for session_id in cohort:
        assert stage.segments_path(roots, session_id).exists()


@pytest.mark.slow
def test_segments_are_written_where_transcripts_belong(roots: DataRoots, cohort: list[int]):
    """Under the work root, never the output root, because they carry text."""
    stage.run(config_with_import(), roots, workers=1)
    path = stage.segments_path(roots, 28)
    assert path.is_relative_to(roots.work)
    assert not list(roots.out.rglob("*.parquet"))


@pytest.mark.slow
def test_transcript_text_is_stored_only_when_configured(roots: DataRoots, cohort: list[int]):
    stage.run(config_with_import(), roots, workers=1)
    with_text = pd.read_parquet(stage.segments_path(roots, 28))
    assert "text" in with_text.columns

    stage.run(config_with_import(**{"diarization.keep_text": False}), roots, workers=1, force=True)
    without = pd.read_parquet(stage.segments_path(roots, 28))
    assert "text" not in without.columns


@pytest.mark.slow
def test_the_qc_table_never_contains_transcript_text(roots: DataRoots, cohort: list[int]):
    result = stage.run(config_with_import(), roots, workers=1)
    written = result.path.read_text(encoding="utf-8")
    # The synthetic transcripts say "turn 0", "turn 1", and so on.
    assert "turn" not in written
    assert "text" not in result.frame.columns


@pytest.mark.slow
def test_speaker_counts_are_recorded_and_flagged(roots: DataRoots, cohort: list[int]):
    result = stage.run(config_with_import(), roots, workers=1)
    assert set(result.frame["n_speakers"]) == {2}
    assert all(flags == "" for flags in result.frame["flags"])


@pytest.mark.slow
def test_a_single_speaker_session_is_flagged_not_corrected(
    roots: DataRoots, import_dir: Path, cohort: list[int]
):
    """One voice usually means the diarizer merged two, which matters downstream."""
    (import_dir / "28.srt").write_text(
        "1\n00:00:01,000 --> 00:00:05,000\nSpeaker 0: only one voice\n", encoding="utf-8"
    )

    result = stage.run(config_with_import(), roots, workers=1)
    row = result.frame.loc[result.frame["session_id"] == 28].iloc[0]

    assert row["n_speakers"] == 1
    assert stage.FLAG_SPEAKER_COUNT in row["flags"]


@pytest.mark.slow
def test_coverage_is_measured_against_the_recording(roots: DataRoots, cohort: list[int]):
    result = stage.run(config_with_import(), roots, workers=1)
    coverage = result.frame["coverage_fraction"].dropna()
    assert not coverage.empty
    assert (coverage > 0.0).all()
    assert (coverage <= 1.5).all()


@pytest.mark.slow
def test_low_coverage_is_flagged(roots: DataRoots, import_dir: Path, cohort: list[int]):
    (import_dir / "28.srt").write_text(
        "1\n00:00:01,000 --> 00:00:02,000\nSpeaker 0: a\n\n"
        "2\n00:00:03,000 --> 00:00:04,000\nSpeaker 1: b\n",
        encoding="utf-8",
    )
    result = stage.run(config_with_import(), roots, workers=1)
    row = result.frame.loc[result.frame["session_id"] == 28].iloc[0]
    assert stage.FLAG_LOW_COVERAGE in row["flags"]


@pytest.mark.slow
def test_a_session_with_no_file_fails_alone(roots: DataRoots, import_dir: Path, cohort: list[int]):
    (import_dir / "28.srt").unlink()

    result = stage.run(config_with_import(), roots, workers=1)

    assert [o.session_id for o in result.report.failed] == [28]
    assert len(result.frame) == 2
    assert sorted(result.frame["session_id"]) == [3, 210]


@pytest.mark.slow
def test_unmatched_files_are_carried_into_the_report(
    roots: DataRoots, import_dir: Path, cohort: list[int]
):
    (import_dir / "interview_final.srt").write_text("x", encoding="utf-8")
    result = stage.run(config_with_import(), roots, workers=1)
    assert any("matched no session" in note for note in result.report.notes)


@pytest.mark.slow
def test_completed_sessions_are_skipped_but_still_reported(roots: DataRoots, cohort: list[int]):
    """A skipped session belongs in the QC table, re-read rather than re-parsed."""
    stage.run(config_with_import(), roots, workers=1)

    second = stage.run(config_with_import(), roots, workers=1)

    assert len(second.report.skipped) == 3
    assert len(second.frame) == 3
    assert second.frame["has_text"].all()


@pytest.mark.slow
def test_force_re_diarizes(roots: DataRoots, cohort: list[int]):
    stage.run(config_with_import(), roots, workers=1)
    assert len(stage.run(config_with_import(), roots, workers=1, force=True).report.succeeded) == 3


@pytest.mark.slow
def test_an_unavailable_backend_stops_the_stage_rather_than_failing_62_times(
    roots: DataRoots, cohort: list[int]
):
    config = load_config(DEFAULT, overrides={"diarization.import_dir": "absent-folder"})
    with pytest.raises(DiarizationError, match="not a directory"):
        stage.run(config, roots, workers=1)


@pytest.mark.slow
def test_an_undecodable_recording_still_diarizes_from_its_file(
    roots: DataRoots, import_dir: Path, cohort: list[int]
):
    """Diarization reads a file, not the media, so a bad mp4 is not fatal here."""
    place_fake_media(roots.data, WINTER_FOLDER, [29])
    write_srt_for(import_dir, 29)

    result = stage.run(config_with_import(), roots, workers=1)

    assert 29 in set(result.frame["session_id"])
    # The coverage figure needs the media, so it is simply absent for that one.
    row = result.frame.loc[result.frame["session_id"] == 29].iloc[0]
    assert pd.isna(row["coverage_fraction"])
    assert row["n_segments"] > 0


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
def _qc_row(session_id: int, **overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "session_id": session_id,
        "wave": "winter",
        "backend": "import",
        "n_segments": 120,
        "n_speakers": 2,
        "speakers": "SPEAKER_00;SPEAKER_01",
        "segment_seconds": 600.0,
        "covered_seconds": 580.0,
        "overlap_seconds": 20.0,
        "coverage_fraction": 0.83,
        "has_text": True,
        "flags": "",
    }
    row.update(overrides)
    return row


def test_summary_reports_speaker_counts(default_config: AppConfig):
    frame = stage.build_frame([_qc_row(1), _qc_row(2, n_speakers=3)])
    text = "\n".join(stage.summarise(frame, default_config))
    assert "speakers per session" in text
    assert "do not have exactly 2 speakers: [2]" in text
    assert "flagged, not corrected" in text


def test_summary_mentions_where_transcripts_live(default_config: AppConfig):
    text = "\n".join(stage.summarise(stage.build_frame([_qc_row(1)]), default_config))
    assert "never in a handoff bundle" in text


def test_summary_explains_high_coverage(default_config: AppConfig):
    """Otherwise 0.9 coverage looks like a bug rather than the expected shape."""
    text = "\n".join(stage.summarise(stage.build_frame([_qc_row(1)]), default_config))
    assert "can span" in text
    assert "vc vad" in text


def test_summary_of_nothing(default_config: AppConfig):
    assert stage.summarise(pd.DataFrame(), default_config) == ["no sessions were diarized"]


def test_an_empty_qc_table_satisfies_the_contract():
    validate(stage.build_frame([]), DIARIZATION_QC_SCHEMA)


def test_segments_frame_types_are_declared():
    frame = stage.segments_frame(28, [Segment("SPEAKER_00", 0.0, 1.0, "hi")])
    assert str(frame["session_id"].dtype) == "int64"
    assert str(frame["start_s"].dtype) == "float64"
    assert "text" in frame.columns


def test_segments_frame_omits_the_text_column_when_there_is_none():
    frame = stage.segments_frame(28, [Segment("SPEAKER_00", 0.0, 1.0, None)])
    assert "text" not in frame.columns
