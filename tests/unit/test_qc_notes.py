"""Tests for human-confirmed QC notes.

The property that matters most: a note is scoped to a modality, so marking a
session's facial features unusable must withhold exactly those and leave its
audio features untouched. Dropping the session would throw away good data;
leaving the values in place would let whoever did not read the note model them.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from vc_multimodal import qc_notes
from vc_multimodal.qc_notes import (
    MODALITY_FAMILIES,
    STATUSES,
    QcNote,
    QcNoteError,
    QcNotes,
    columns_for_families,
    families_of,
    record,
)

WHEN = datetime(2026, 9, 28, tzinfo=UTC)

BLUR_REASON = (
    "participant's camera is too out of focus for face tracking; confirmed by "
    "watching the recording"
)


def write_notes(path: Path, rows: list[dict[str, object]]) -> Path:
    pd.DataFrame(rows, columns=list(qc_notes.COLUMNS)).to_csv(path, index=False)
    return path


def blur_row(session_id: int = 43, modality: str = "face") -> dict[str, object]:
    return {
        "session_id": session_id,
        "modality": modality,
        "status": "unavailable",
        "reason": BLUR_REASON,
        "recorded_by": "tester",
        "recorded_on": "2026-09-28",
    }


class TestLoading:
    def test_a_note_is_read_back(self, tmp_path: Path) -> None:
        loaded = qc_notes.load(write_notes(tmp_path / "n.csv", [blur_row()]))
        assert len(loaded) == 1
        assert loaded.notes[0].session_id == 43
        assert loaded.notes[0].modality == "face"

    def test_no_file_means_no_notes(self, tmp_path: Path) -> None:
        assert not qc_notes.load(tmp_path / "absent.csv")

    def test_an_unknown_modality_is_refused(self, tmp_path: Path) -> None:
        path = write_notes(tmp_path / "n.csv", [blur_row(modality="eyebrows")])
        with pytest.raises(QcNoteError, match="modality"):
            qc_notes.load(path)

    def test_an_unknown_status_is_refused(self, tmp_path: Path) -> None:
        row = blur_row() | {"status": "probably fine"}
        with pytest.raises(QcNoteError, match="status"):
            qc_notes.load(write_notes(tmp_path / "n.csv", [row]))

    def test_a_reason_too_short_to_act_on_is_refused(self, tmp_path: Path) -> None:
        row = blur_row() | {"reason": "bad"}
        with pytest.raises(QcNoteError, match="too short to be a finding"):
            qc_notes.load(write_notes(tmp_path / "n.csv", [row]))

    def test_the_line_number_is_named(self, tmp_path: Path) -> None:
        rows = [blur_row(43), blur_row(3) | {"status": "nonsense"}]
        with pytest.raises(QcNoteError, match="line 3"):
            qc_notes.load(write_notes(tmp_path / "n.csv", rows))

    def test_a_malformed_note_is_refused_rather_than_skipped(self, tmp_path: Path) -> None:
        # A note silently dropped is worse than no note: the finding is then
        # invisible while appearing to be recorded.
        rows = [blur_row(43), blur_row(3) | {"modality": "typo"}]
        with pytest.raises(QcNoteError):
            qc_notes.load(write_notes(tmp_path / "n.csv", rows))

    def test_two_notes_for_one_modality_are_refused(self, tmp_path: Path) -> None:
        rows = [blur_row(43), blur_row(43)]
        with pytest.raises(QcNoteError, match="more than one note"):
            qc_notes.load(write_notes(tmp_path / "n.csv", rows))

    def test_two_modalities_of_one_session_are_fine(self, tmp_path: Path) -> None:
        rows = [blur_row(43, "face"), blur_row(43, "prosody")]
        assert len(qc_notes.load(write_notes(tmp_path / "n.csv", rows))) == 2

    def test_missing_columns_say_what_is_needed(self, tmp_path: Path) -> None:
        path = tmp_path / "n.csv"
        pd.DataFrame({"session_id": [43]}).to_csv(path, index=False)
        with pytest.raises(QcNoteError, match="missing column"):
            qc_notes.load(path)


class TestScoping:
    def test_a_face_note_covers_both_windows(self) -> None:
        # A camera problem does not respect the speaking/listening split.
        assert families_of(["face"]) == ("face_speaking", "face_listening")

    def test_an_audio_note_covers_turns_and_prosody(self) -> None:
        assert families_of(["audio"]) == ("turns", "prosody")

    def test_every_modality_maps_to_at_least_one_family(self) -> None:
        for modality in MODALITY_FAMILIES:
            assert MODALITY_FAMILIES[modality]

    def test_only_the_named_families_columns_are_selected(self) -> None:
        columns = [
            "turns__latency_median",
            "prosody__f0_semitone_sd",
            "face_speaking__au12_mean",
            "face_listening__au06_p90",
            "qc__flags",
        ]
        selected = columns_for_families(columns, ["face_speaking", "face_listening"])
        assert selected == ("face_speaking__au12_mean", "face_listening__au06_p90")

    def test_qc_columns_are_never_selected(self) -> None:
        assert columns_for_families(["qc__flags"], ["face_speaking"]) == ()

    def test_only_unavailable_withholds_features(self) -> None:
        note = QcNote(43, "face", "degraded", BLUR_REASON, "t", "2026-09-28")
        assert not note.blanks_features
        assert QcNote(43, "face", "unavailable", BLUR_REASON, "t", "x").blanks_features

    @pytest.mark.parametrize("status", STATUSES)
    def test_every_status_is_loadable(self, tmp_path: Path, status: str) -> None:
        row = blur_row() | {"status": status}
        assert len(qc_notes.load(write_notes(tmp_path / "n.csv", [row]))) == 1


class TestRecording:
    def test_a_note_is_written_and_dated(self, tmp_path: Path) -> None:
        path = tmp_path / "n.csv"
        note = record(
            path,
            session_id=43,
            modality="face",
            status="unavailable",
            reason=BLUR_REASON,
            recorded_by="tester",
            now=WHEN,
        )
        assert note.recorded_on == "2026-09-28"
        assert qc_notes.load(path).notes[0].recorded_by == "tester"

    def test_recording_twice_is_refused_by_default(self, tmp_path: Path) -> None:
        path = tmp_path / "n.csv"
        record(
            path,
            session_id=43,
            modality="face",
            status="unavailable",
            reason=BLUR_REASON,
            recorded_by="t",
            now=WHEN,
        )
        with pytest.raises(QcNoteError, match="already has a face note"):
            record(
                path,
                session_id=43,
                modality="face",
                status="note",
                reason=BLUR_REASON,
                recorded_by="t",
                now=WHEN,
            )

    def test_replace_changes_the_existing_note(self, tmp_path: Path) -> None:
        path = tmp_path / "n.csv"
        record(
            path,
            session_id=43,
            modality="face",
            status="unavailable",
            reason=BLUR_REASON,
            recorded_by="t",
            now=WHEN,
        )
        record(
            path,
            session_id=43,
            modality="face",
            status="degraded",
            reason=BLUR_REASON,
            recorded_by="t",
            now=WHEN,
            replace=True,
        )
        loaded = qc_notes.load(path)
        assert len(loaded) == 1
        assert loaded.notes[0].status == "degraded"

    def test_notes_for_other_sessions_survive(self, tmp_path: Path) -> None:
        path = tmp_path / "n.csv"
        record(
            path,
            session_id=43,
            modality="face",
            status="unavailable",
            reason=BLUR_REASON,
            recorded_by="t",
            now=WHEN,
        )
        record(
            path,
            session_id=3,
            modality="prosody",
            status="degraded",
            reason="background noise throughout, confirmed by listening",
            recorded_by="t",
            now=WHEN,
        )
        assert qc_notes.load(path).sessions == (3, 43)

    def test_an_invalid_note_is_never_written(self, tmp_path: Path) -> None:
        path = tmp_path / "n.csv"
        with pytest.raises(QcNoteError):
            record(
                path,
                session_id=43,
                modality="face",
                status="unavailable",
                reason="bad",
                recorded_by="t",
                now=WHEN,
            )
        assert not path.exists()


class TestReporting:
    def notes(self) -> QcNotes:
        return QcNotes(notes=(QcNote(43, "face", "unavailable", BLUR_REASON, "t", "2026-09-28"),))

    def test_the_label_is_compact_enough_for_a_qc_column(self) -> None:
        assert self.notes().labels(43) == "face=unavailable"

    def test_the_reason_travels_in_full(self) -> None:
        assert BLUR_REASON in self.notes().reasons(43)

    def test_the_report_names_who_confirmed_it_and_when(self) -> None:
        text = "\n".join(self.notes().report_lines())
        assert "confirmed by t on 2026-09-28" in text
        assert "UNAVAILABLE" in text

    def test_the_markdown_row_is_a_table_row(self) -> None:
        row = self.notes().markdown_rows()[0]
        assert row.startswith("| 43 | face | unavailable |")

    def test_a_session_with_no_note_reports_nothing(self) -> None:
        assert self.notes().labels(9) == ""
        assert self.notes().for_session(9) == ()

    def test_unavailable_families_are_listed_once(self) -> None:
        both = QcNotes(
            notes=(
                QcNote(43, "face", "unavailable", BLUR_REASON, "t", "x"),
                QcNote(43, "face_speaking", "unavailable", BLUR_REASON, "t", "x"),
            )
        )
        assert both.unavailable_families(43) == ("face_speaking", "face_listening")

    def test_a_degraded_note_withholds_nothing(self) -> None:
        degraded = QcNotes(notes=(QcNote(43, "face", "degraded", BLUR_REASON, "t", "x"),))
        assert degraded.unavailable_families(43) == ()
        assert degraded.labels(43) == "face=degraded"
