"""Tests for per-session table merging.

Two properties, both of which were broken in shipped code:

1. A run over a subset keeps the rows for every session it did not touch.
2. A session the runner skips never ends up with an empty row, because a row
   that exists but holds no measurements is indistinguishable from a failed
   measurement and silently corrupts every rate computed over the table.

The first half of this file tests the shared helper directly. The second half
asserts that every stage which writes a per-session table actually uses it -
the audit that found these bugs was done by hand, and this is what keeps it
done.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pandas as pd
import pytest

from vc_multimodal.session_tables import (
    ID_COLUMN,
    SessionTableError,
    carry_forward,
    combine,
    has_row,
    rows_with_values,
)
from vc_multimodal.stages import (
    aggregate,
    assign_speakers,
    diarize,
    extract_audio,
    face,
    prosody,
    turns,
    vad,
    verify_layout,
)

#: Every stage that writes one row per session.
SESSION_TABLE_STAGES = (
    aggregate,
    assign_speakers,
    diarize,
    extract_audio,
    face,
    prosody,
    turns,
    vad,
    verify_layout,
)


def existing_table(path: Path, session_ids: list[int], value: float = 1.0) -> Path:
    frame = pd.DataFrame(
        {
            ID_COLUMN: session_ids,
            "wave": ["winter"] * len(session_ids),
            "measured": [value + index for index in range(len(session_ids))],
        }
    )
    frame.to_csv(path, index=False)
    return path


class TestCarryForward:
    def test_rows_for_sessions_not_recomputed_are_kept(self, tmp_path: Path) -> None:
        path = existing_table(tmp_path / "qc.csv", [1, 2, 3])
        carried = carry_forward(path, computed={2})
        assert carried.session_ids == (1, 3)

    def test_recomputed_sessions_are_dropped_so_the_fresh_row_wins(self, tmp_path: Path) -> None:
        path = existing_table(tmp_path / "qc.csv", [1, 2, 3])
        carried = carry_forward(path, computed={1, 2, 3})
        assert not carried
        assert carried.session_ids == ()

    def test_no_existing_table_carries_nothing(self, tmp_path: Path) -> None:
        assert not carry_forward(tmp_path / "absent.csv", computed={1})

    def test_a_file_with_no_session_id_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "qc.csv"
        pd.DataFrame({"something": [1, 2]}).to_csv(path, index=False)
        with pytest.raises(SessionTableError, match="no session_id column"):
            carry_forward(path, computed={1})

    def test_the_refusal_says_how_to_proceed(self, tmp_path: Path) -> None:
        path = tmp_path / "qc.csv"
        pd.DataFrame({"something": [1]}).to_csv(path, index=False)
        with pytest.raises(SessionTableError) as excinfo:
            carry_forward(path, computed={1})
        assert "--force" in str(excinfo.value)
        assert "Move it aside" in str(excinfo.value)

    def test_force_starts_fresh_instead_of_refusing(self, tmp_path: Path) -> None:
        path = tmp_path / "qc.csv"
        pd.DataFrame({"something": [1]}).to_csv(path, index=False)
        assert not carry_forward(path, computed={1}, force=True)

    def test_rows_with_no_usable_identifier_are_dropped(self, tmp_path: Path) -> None:
        path = tmp_path / "qc.csv"
        pd.DataFrame({ID_COLUMN: [1, "nonsense"], "measured": [1.0, 2.0]}).to_csv(path, index=False)
        carried = carry_forward(path, computed={9})
        assert carried.session_ids == (1,)

    def test_a_column_the_old_table_lacks_is_reported(self, tmp_path: Path) -> None:
        # Otherwise carried rows are quietly empty in the new column, which is
        # the failure mode this whole module exists to prevent.
        path = existing_table(tmp_path / "qc.csv", [1, 2])
        carried = carry_forward(
            path, computed={2}, columns=[ID_COLUMN, "measured", "added_later"], stage="face"
        )
        assert carried.absent_columns == ("added_later",)
        assert any("added_later" in note for note in carried.notes("face"))

    def test_matching_columns_produce_no_note(self, tmp_path: Path) -> None:
        path = existing_table(tmp_path / "qc.csv", [1, 2])
        carried = carry_forward(path, computed={2}, columns=[ID_COLUMN, "wave", "measured"])
        assert carried.absent_columns == ()
        assert not any("predates" in note for note in carried.notes("face"))


class TestCombine:
    def test_fresh_rows_replace_carried_ones(self, tmp_path: Path) -> None:
        path = existing_table(tmp_path / "qc.csv", [1, 2, 3])
        carried = carry_forward(path, computed={2})
        merged = combine({2: {ID_COLUMN: 2, "measured": 99.0}}, carried)
        by_session = {int(row[ID_COLUMN]): row for row in merged}
        assert by_session[2]["measured"] == 99.0

    def test_the_result_is_ordered_by_session(self, tmp_path: Path) -> None:
        path = existing_table(tmp_path / "qc.csv", [10, 3])
        carried = carry_forward(path, computed={7})
        merged = combine({7: {ID_COLUMN: 7}}, carried)
        assert [int(row[ID_COLUMN]) for row in merged] == [3, 7, 10]

    def test_carried_and_fresh_rows_are_both_present(self, tmp_path: Path) -> None:
        path = existing_table(tmp_path / "qc.csv", [1, 2, 3])
        carried = carry_forward(path, computed={2})
        merged = combine({2: {ID_COLUMN: 2}}, carried)
        assert len(merged) == 3


class TestHasRow:
    def test_a_present_session_is_found(self, tmp_path: Path) -> None:
        path = existing_table(tmp_path / "qc.csv", [1, 2])
        assert has_row(path, 1)

    def test_an_absent_session_is_not(self, tmp_path: Path) -> None:
        path = existing_table(tmp_path / "qc.csv", [1, 2])
        assert not has_row(path, 9)

    def test_a_missing_table_has_no_rows(self, tmp_path: Path) -> None:
        assert not has_row(tmp_path / "absent.csv", 1)

    def test_a_foreign_file_has_no_rows(self, tmp_path: Path) -> None:
        path = tmp_path / "qc.csv"
        path.write_text("not,a,table\n1,2,3\n")
        assert not has_row(path, 1)


class TestRowsWithValues:
    def test_only_present_values_are_counted(self) -> None:
        frame = pd.DataFrame({"letterbox_detected": [True, None, False]})
        assert len(rows_with_values(frame, "letterbox_detected")) == 2

    def test_an_absent_column_yields_nothing(self) -> None:
        assert rows_with_values(pd.DataFrame({"a": [1]}), "missing").empty


class TestEveryStageUsesIt:
    """The audit, kept done.

    These bugs were found by hand after a partial rerun destroyed real coverage.
    A new stage that writes a per-session table and forgets to merge would
    reintroduce them silently, so the requirement is asserted rather than
    remembered.
    """

    @pytest.mark.parametrize("module", SESSION_TABLE_STAGES, ids=lambda m: m.STAGE)
    def test_the_stage_merges_with_what_is_already_written(self, module: object) -> None:
        source = inspect.getsource(module)  # type: ignore[arg-type]
        if module is extract_audio:
            # Rebuilds the whole table from every sidecar on disk, which reaches
            # the same place by a different route and loses nothing.
            assert 'glob("*.stats.json")' in source
            return
        assert "carry_forward(" in source, f"{module.STAGE} overwrites its table"  # type: ignore[attr-defined]
        assert "combine(" in source

    @pytest.mark.parametrize("module", [diarize, face, prosody, turns, vad], ids=lambda m: m.STAGE)
    def test_a_stage_that_skips_requires_a_row_before_skipping(self, module: object) -> None:
        # Otherwise "skipped" can mean "has no row and never will", which is how
        # a blank row or a missing one gets into the table.
        source = inspect.getsource(module)  # type: ignore[arg-type]
        assert "has_row(" in source, f"{module.STAGE} can skip a session with no row"  # type: ignore[attr-defined]
