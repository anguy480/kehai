"""Data contracts, including the requirement that failures do not quote data."""

from __future__ import annotations

import pandas as pd
import pytest

from vc_multimodal.contracts import (
    FEATURE_FAMILIES,
    INVENTORY_SCHEMA,
    SEGMENT_SCHEMA,
    ContractError,
    columns_in_families,
    family_of,
    feature_schema,
    validate,
)


def _inventory_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "session_id": 28,
        "wave": "winter",
        "date_folder": "December 21 2025",
        "relpath": "December 21 2025/28.mp4",
        "readable": True,
        "duration_s": 640.0,
        "size_bytes": 123456,
        "video_codec": "h264",
        "width": 1920,
        "height": 1080,
        "fps": 25.0,
        "fps_variable": False,
        "n_audio_streams": 1,
        "audio_codec": "aac",
        "audio_channels": 1,
        "audio_sample_rate": 48000,
        "flags": "",
    }
    row.update(overrides)
    return row


# Nullable columns need an explicit dtype: a column that is entirely missing
# would otherwise land as `object` and fail the contract. The inventory stage
# casts the same way, which is what makes an all-unreadable run still valid.
_INT_COLUMNS = (
    "size_bytes",
    "width",
    "height",
    "n_audio_streams",
    "audio_channels",
    "audio_sample_rate",
)
_FLOAT_COLUMNS = ("duration_s", "fps")
_STR_COLUMNS = ("video_codec", "audio_codec")


def _inventory_frame(*rows: dict[str, object]) -> pd.DataFrame:
    frame = pd.DataFrame(list(rows) or [_inventory_row()])
    for column in _INT_COLUMNS:
        frame[column] = frame[column].astype("Int64")
    for column in _FLOAT_COLUMNS:
        frame[column] = frame[column].astype("float64")
    for column in _STR_COLUMNS:
        frame[column] = frame[column].astype("object")
    return frame


def test_a_well_formed_inventory_validates():
    validate(_inventory_frame(), INVENTORY_SCHEMA)


def test_inventory_allows_an_unreadable_file_with_missing_metadata():
    """A file ffprobe cannot read is still reported, with nulls, not dropped."""
    row = _inventory_row(
        readable=False,
        duration_s=None,
        video_codec=None,
        width=None,
        height=None,
        fps=None,
        n_audio_streams=None,
        audio_codec=None,
        audio_channels=None,
        audio_sample_rate=None,
        flags="unreadable",
    )
    validate(_inventory_frame(row), INVENTORY_SCHEMA)


def test_inventory_rejects_duplicate_session_ids():
    frame = _inventory_frame(_inventory_row(), _inventory_row())
    with pytest.raises(ContractError, match="failed validation"):
        validate(frame, INVENTORY_SCHEMA)


def test_inventory_rejects_an_unexpected_column():
    frame = _inventory_frame()
    frame["k6_score"] = 12
    with pytest.raises(ContractError):
        validate(frame, INVENTORY_SCHEMA)


def test_inventory_rejects_a_negative_duration():
    with pytest.raises(ContractError):
        validate(_inventory_frame(_inventory_row(duration_s=-1.0)), INVENTORY_SCHEMA)


def test_segments_must_end_after_they_start():
    frame = pd.DataFrame(
        {
            "session_id": [1, 1],
            "speaker": ["SPEAKER_00", "SPEAKER_01"],
            "start_s": [0.0, 5.0],
            "end_s": [1.0, 3.0],
        }
    )
    with pytest.raises(ContractError, match="must end after it starts"):
        validate(frame, SEGMENT_SCHEMA)


def test_segments_may_carry_text_in_the_work_tree():
    """Transcript text is permitted in intermediates, never in the handoff."""
    frame = pd.DataFrame(
        {
            "session_id": [1],
            "speaker": ["SPEAKER_00"],
            "start_s": [0.0],
            "end_s": [1.0],
            "text": ["こんにちは"],
        }
    )
    validate(frame, SEGMENT_SCHEMA)


# ---------------------------------------------------------------------------
# the redaction property
# ---------------------------------------------------------------------------
def test_validation_failures_do_not_quote_the_offending_values():
    """Failure messages reach logs, so they must not carry data values."""
    frame = pd.DataFrame(
        {
            "session_id": [1],
            "speaker": ["SPEAKER_00"],
            "start_s": [0.0],
            "end_s": [-99999.5],
        }
    )
    with pytest.raises(ContractError) as caught:
        validate(frame, SEGMENT_SCHEMA, context="session 1")
    message = str(caught.value)
    assert "-99999.5" not in message
    assert "session 1" in message
    assert "end_s" in message or "end_after_start" in message


def test_redaction_can_be_switched_off_for_synthetic_data():
    frame = pd.DataFrame({"session_id": [1], "speaker": ["A"], "start_s": [0.0], "end_s": [-42.0]})
    with pytest.raises(ContractError) as caught:
        validate(frame, SEGMENT_SCHEMA, redact=False)
    assert "-42" in str(caught.value)


def test_a_single_column_failure_is_also_redacted():
    frame = pd.DataFrame({"session_id": ["not-a-number"]})
    with pytest.raises(ContractError) as caught:
        validate(frame, feature_schema([], require_wave=False))
    assert "not-a-number" not in str(caught.value)


# ---------------------------------------------------------------------------
# feature naming convention
# ---------------------------------------------------------------------------
def test_feature_schema_accepts_the_naming_convention():
    schema = feature_schema(["turns__latency_median", "prosody__f0_semitone_sd", "qc__flags"])
    frame = pd.DataFrame(
        {
            "session_id": [1],
            "wave": ["winter"],
            "turns__latency_median": [0.8],
            "prosody__f0_semitone_sd": [2.1],
            "qc__flags": ["none"],
        }
    )
    validate(frame, schema)


@pytest.mark.parametrize(
    "bad",
    [
        "latency_median",  # no family
        "turns_latency",  # single underscore
        "Turns__latency",  # capitalised family
        "unknown__thing",  # family not in the list
        "turns__Latency",  # capitalised name
        "turns__",  # empty name
    ],
)
def test_feature_schema_rejects_names_off_convention(bad: str):
    with pytest.raises(ContractError, match="naming convention"):
        feature_schema([bad])


def test_every_family_is_accepted():
    schema = feature_schema([f"{family}__x" for family in FEATURE_FAMILIES])
    assert len(schema.columns) == len(FEATURE_FAMILIES) + 2


def test_features_must_be_numeric_but_qc_columns_need_not_be():
    schema = feature_schema(["turns__n", "qc__note"])
    frame = pd.DataFrame(
        {"session_id": [1], "wave": ["winter"], "turns__n": ["twelve"], "qc__note": ["fine"]}
    )
    with pytest.raises(ContractError):
        validate(frame, schema)


def test_family_of_identifies_features_and_ignores_the_rest():
    assert family_of("face_speaking__jaw_open_sd") == "face_speaking"
    assert family_of("session_id") is None
    assert family_of("qc__flags") is None


def test_columns_in_families_selects_by_family_preserving_order():
    columns = [
        "session_id",
        "turns__a",
        "prosody__b",
        "face_speaking__c",
        "turns__d",
        "qc__e",
    ]
    assert columns_in_families(columns, ["turns", "prosody"]) == (
        "turns__a",
        "prosody__b",
        "turns__d",
    )


def test_columns_in_families_with_no_families_selects_nothing():
    assert columns_in_families(["turns__a"], []) == ()
