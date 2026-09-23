"""Role mapping: the one thing this pipeline must never guess.

If the participant and psychiatrist are swapped, nothing crashes. Prosody is
measured on the wrong voice and the speaking/listening split for facial
features inverts, producing confidently wrong features. So the rule is that a
mapping either exists as recorded evidence, or the caller is told how to make
one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from vc_multimodal.paths import DataRoots
from vc_multimodal.roles import (
    ROLE_PARTICIPANT,
    ROLE_PSYCHIATRIST,
    SOURCE_ASSIGNED,
    SOURCE_MANUAL,
    RoleMapping,
    RolesUnavailableError,
    load_manual_roles,
    load_role_mapping,
    manual_roles_path,
    roles_path,
    spans_by_role,
    write_role_mapping,
)

MAPPING = {"SPEAKER_00": ROLE_PSYCHIATRIST, "SPEAKER_01": ROLE_PARTICIPANT}


# ---------------------------------------------------------------------------
# the mapping object
# ---------------------------------------------------------------------------
def test_roles_can_be_looked_up_both_ways():
    mapping = RoleMapping(28, MAPPING, SOURCE_ASSIGNED)
    assert mapping.role_of("SPEAKER_01") == ROLE_PARTICIPANT
    assert mapping.speakers_for(ROLE_PARTICIPANT) == ("SPEAKER_01",)
    assert mapping.has_participant
    assert mapping.has_psychiatrist


def test_an_unmapped_speaker_is_unknown_rather_than_assumed():
    mapping = RoleMapping(28, MAPPING, SOURCE_ASSIGNED)
    assert mapping.role_of("SPEAKER_02") == "unknown"


def test_two_labels_may_share_one_role():
    """A diarizer that split one voice produces two labels for one person."""
    mapping = RoleMapping(
        28,
        {
            "SPEAKER_00": ROLE_PSYCHIATRIST,
            "SPEAKER_01": ROLE_PARTICIPANT,
            "SPEAKER_02": ROLE_PARTICIPANT,
        },
        SOURCE_ASSIGNED,
    )
    assert mapping.speakers_for(ROLE_PARTICIPANT) == ("SPEAKER_01", "SPEAKER_02")


def test_a_mapping_with_no_participant_says_so():
    mapping = RoleMapping(28, {"SPEAKER_00": ROLE_PSYCHIATRIST}, SOURCE_ASSIGNED)
    assert not mapping.has_participant
    assert mapping.has_psychiatrist


# ---------------------------------------------------------------------------
# writing and reading an assignment
# ---------------------------------------------------------------------------
def test_an_assignment_round_trips(roots: DataRoots):
    write_role_mapping(roots.work, 28, MAPPING, evidence={"margin": 0.42})

    loaded = load_role_mapping(roots.work, 28)

    assert loaded.by_speaker == MAPPING
    assert loaded.source == SOURCE_ASSIGNED
    assert loaded.session_id == 28


def test_the_evidence_is_kept_alongside_the_decision(roots: DataRoots):
    """So a questionable assignment can be re-examined later."""
    path = write_role_mapping(roots.work, 28, MAPPING, evidence={"margin": 0.42})
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["evidence"]["margin"] == 0.42


def test_an_unknown_role_is_refused(roots: DataRoots):
    with pytest.raises(ValueError, match="unknown role"):
        write_role_mapping(roots.work, 28, {"SPEAKER_00": "interviewer"})


def test_a_corrupt_assignment_is_reported(roots: DataRoots):
    path = roles_path(roots.work, 28)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not json", encoding="utf-8")
    with pytest.raises(RolesUnavailableError, match="could not read"):
        load_role_mapping(roots.work, 28)


# ---------------------------------------------------------------------------
# the hand-written table
# ---------------------------------------------------------------------------
def write_manual(roots: DataRoots, rows: list[tuple[int, str, str]]) -> Path:
    frame = pd.DataFrame(rows, columns=["session_id", "speaker", "role"])
    path = manual_roles_path(roots.work)
    frame.to_csv(path, index=False)
    return path


def test_a_hand_written_table_is_used_when_no_assignment_exists(roots: DataRoots):
    """So a few sessions can be piloted before the reference clip exists."""
    write_manual(
        roots, [(28, "SPEAKER_00", ROLE_PSYCHIATRIST), (28, "SPEAKER_01", ROLE_PARTICIPANT)]
    )

    mapping = load_role_mapping(roots.work, 28)

    assert mapping.by_speaker == MAPPING
    assert mapping.source == SOURCE_MANUAL


def test_an_assignment_takes_precedence_over_the_hand_written_table(roots: DataRoots):
    """Recorded evidence beats a human's note."""
    write_manual(roots, [(28, "SPEAKER_00", ROLE_PARTICIPANT)])
    write_role_mapping(roots.work, 28, MAPPING)

    assert load_role_mapping(roots.work, 28).source == SOURCE_ASSIGNED


def test_the_table_may_cover_only_some_sessions(roots: DataRoots):
    write_manual(roots, [(28, "SPEAKER_00", ROLE_PSYCHIATRIST)])
    assert load_role_mapping(roots.work, 28).source == SOURCE_MANUAL
    with pytest.raises(RolesUnavailableError):
        load_role_mapping(roots.work, 3)


def test_a_table_missing_columns_is_reported(roots: DataRoots):
    path = manual_roles_path(roots.work)
    path.write_text("session_id,speaker\n28,SPEAKER_00\n", encoding="utf-8")
    with pytest.raises(RolesUnavailableError, match="missing column"):
        load_manual_roles(path)


def test_a_table_with_an_invalid_role_is_reported(roots: DataRoots):
    path = write_manual(roots, [(28, "SPEAKER_00", "interviewer")])
    with pytest.raises(RolesUnavailableError, match="not a valid role table"):
        load_manual_roles(path)


def test_duplicate_rows_for_one_speaker_are_reported(roots: DataRoots):
    path = write_manual(
        roots, [(28, "SPEAKER_00", ROLE_PSYCHIATRIST), (28, "SPEAKER_00", ROLE_PARTICIPANT)]
    )
    with pytest.raises(RolesUnavailableError, match="not a valid role table"):
        load_manual_roles(path)


def test_a_table_covering_several_sessions(roots: DataRoots):
    path = write_manual(
        roots,
        [
            (28, "SPEAKER_00", ROLE_PSYCHIATRIST),
            (28, "SPEAKER_01", ROLE_PARTICIPANT),
            (3, "SPEAKER_00", ROLE_PARTICIPANT),
        ],
    )
    table = load_manual_roles(path)
    assert set(table) == {28, 3}
    assert table[3]["SPEAKER_00"] == ROLE_PARTICIPANT


# ---------------------------------------------------------------------------
# refusing to guess
# ---------------------------------------------------------------------------
def test_with_no_mapping_at_all_the_error_explains_both_routes(roots: DataRoots):
    with pytest.raises(RolesUnavailableError) as caught:
        load_role_mapping(roots.work, 28)

    message = str(caught.value)
    assert "vc assign-speakers" in message
    assert "reference/" in message
    assert "roles.csv" in message
    assert "diarization_qc.csv" in message


# ---------------------------------------------------------------------------
# applying a mapping to speech spans
# ---------------------------------------------------------------------------
def test_speech_spans_are_regrouped_by_role():
    speech = pd.DataFrame(
        {
            "speaker": ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"],
            "start_s": [0.0, 2.0, 4.0],
            "end_s": [1.0, 3.0, 5.0],
        }
    )
    grouped = spans_by_role(speech, RoleMapping(28, MAPPING, SOURCE_ASSIGNED))

    assert grouped[ROLE_PSYCHIATRIST] == [(0.0, 1.0), (4.0, 5.0)]
    assert grouped[ROLE_PARTICIPANT] == [(2.0, 3.0)]


def test_an_unmapped_speaker_lands_under_unknown_rather_than_a_role():
    speech = pd.DataFrame({"speaker": ["SPEAKER_07"], "start_s": [0.0], "end_s": [1.0]})
    grouped = spans_by_role(speech, RoleMapping(28, MAPPING, SOURCE_ASSIGNED))
    assert "unknown" in grouped
    assert ROLE_PARTICIPANT not in grouped


def test_an_empty_speech_table_groups_to_nothing():
    speech = pd.DataFrame({"speaker": [], "start_s": [], "end_s": []})
    assert spans_by_role(speech, RoleMapping(28, MAPPING, SOURCE_ASSIGNED)) == {}
