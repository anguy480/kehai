"""The plan of docs/prereg/fusion.md, as constants.

Every choice the pre-registration fixes is here: the six modalities and their
columns, the 15 feature sets, the 14 comparisons, the sessions and the
settings. The four tables' SHA-256 are not repeated. They are the ones already
fixed in the LLM ratings and per-modality plans, and the tables are read
through those plans' own loaders, which refuse any other file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from vc_multimodal.exploratory.llm_ratings.analyze import TEXT_COLUMNS
from vc_multimodal.exploratory.llm_ratings.plan import FEATURES as LLM_COLUMNS

PREREG: Final = "docs/prereg/fusion.md"
TARGETS: Final = ("K6", "SRS2")

TEXT: Final = "text"
LLM: Final = "llm"
PROSODY: Final = "prosody"
TURNS: Final = "turns"
FACE_SPEAKING: Final = "face_speaking"
FACE_LISTENING: Final = "face_listening"

#: Where each modality is read from. A column from any other table is refused.
TABLE_TEXT: Final = "text table"
TABLE_LLM: Final = "LLM table"
TABLE_BUNDLE: Final = "bundle"
TABLE_FACE: Final = "face table"

PROSODY_COLUMNS: Final = (
    "prosody__f0_semitone_sd",
    "prosody__f0_semitone_iqr",
    "prosody__f0_semitone_range",
    "prosody__f0_semitone_mean_abs_delta",
    "prosody__voiced_fraction",
    "prosody__intensity_mean_db",
    "prosody__intensity_sd_db",
    "prosody__intensity_range_db",
    "prosody__jitter_local",
    "prosody__shimmer_local",
    "prosody__hnr_db",
    "prosody__speech_rate_proxy",
)

#: Zero in all 62 sessions, because whisper-diarization partitions time
#: (ADR 12, amendment of 2026-09-24). Left out here, unlike the per-modality
#: evaluation, which kept them at zero.
CONSTANT_TURNS: Final = ("turns__overlap_ratio", "turns__interruption_rate")

TURNS_COLUMNS: Final = (
    "turns__n_per_minute",
    "turns__participant_speaking_ratio",
    "turns__latency_mean",
    "turns__latency_median",
    "turns__latency_sd",
    "turns__participant_turn_duration_mean",
    "turns__participant_turn_duration_sd",
    "turns__psychiatrist_turn_duration_mean",
    "turns__pause_within_mean",
    "turns__pause_within_rate",
)

_AUS: Final = ("au01", "au06", "au12")
_STATISTICS: Final = ("mean", "sd", "p90")


def _face(window: str) -> tuple[str, ...]:
    return tuple(f"{window}__{au}_{statistic}" for au in _AUS for statistic in _STATISTICS)


#: Each modality's columns, in the order the full model takes them.
MODALITIES: Final[dict[str, tuple[str, ...]]] = {
    TEXT: TEXT_COLUMNS,
    LLM: tuple(LLM_COLUMNS),
    PROSODY: PROSODY_COLUMNS,
    TURNS: TURNS_COLUMNS,
    FACE_SPEAKING: _face(FACE_SPEAKING),
    FACE_LISTENING: _face(FACE_LISTENING),
}

SOURCES: Final[dict[str, str]] = {
    TEXT: TABLE_TEXT,
    LLM: TABLE_LLM,
    PROSODY: TABLE_BUNDLE,
    TURNS: TABLE_BUNDLE,
    FACE_SPEAKING: TABLE_FACE,
    FACE_LISTENING: TABLE_FACE,
}

FULL: Final = "full"
TEXT_ONLY: Final = "text"
#: The full set less llm, which Amendment 1 compares with text as text + nonverbal.
TEXT_PLUS_NONVERBAL: Final = "full_minus_llm"


def _without(*dropped: str) -> tuple[str, ...]:
    return tuple(m for m in MODALITIES if m not in dropped)


#: The 15 feature sets, as modalities, in the pre-registration's order.
FEATURE_SETS: Final[dict[str, tuple[str, ...]]] = {
    FULL: tuple(MODALITIES),
    TEXT_ONLY: (TEXT,),
    **{f"full_minus_{m}": _without(m) for m in MODALITIES},
    "full_minus_face": _without(FACE_SPEAKING, FACE_LISTENING),
    "full_minus_language": _without(TEXT, LLM),
    **{f"text_plus_{m}": (TEXT, m) for m in MODALITIES if m != TEXT},
}

#: Sets that also get repeated 5-fold, and the one set that gets a null.
STABILITY_SETS: Final = (FULL, TEXT_ONLY)
PERMUTED_SETS: Final = (FULL,)


def set_columns(feature_set: str) -> tuple[str, ...]:
    """A feature set's columns, modality by modality."""
    return tuple(c for m in FEATURE_SETS[feature_set] for c in MODALITIES[m])


KIND_DROP_ONE: Final = "drop_one"
KIND_GROUPED_DROP: Final = "grouped_drop"
KIND_ADD_ONE: Final = "add_one"
KIND_AMENDMENT_1: Final = "amendment_1"


@dataclass(frozen=True, slots=True)
class Comparison:
    """A larger model against a smaller one. Every difference is larger minus smaller."""

    name: str
    kind: str
    larger: str
    smaller: str


#: The 14 descriptive comparisons: 13 from the plan, and Amendment 1's.
COMPARISONS: Final = (
    *(Comparison(f"drop_{m}", KIND_DROP_ONE, FULL, f"full_minus_{m}") for m in MODALITIES),
    Comparison("drop_face", KIND_GROUPED_DROP, FULL, "full_minus_face"),
    Comparison("drop_language", KIND_GROUPED_DROP, FULL, "full_minus_language"),
    *(
        Comparison(f"add_{m}", KIND_ADD_ONE, f"text_plus_{m}", TEXT_ONLY)
        for m in MODALITIES
        if m != TEXT
    ),
    Comparison("text_plus_nonverbal", KIND_AMENDMENT_1, TEXT_PLUS_NONVERBAL, TEXT_ONLY),
)

#: Left out of every model: its face is unavailable (camera out of focus).
EXCLUDED_SESSIONS: Final = (43,)
#: Kept, with the flag carried into the results (face degraded).
FLAGGED_SESSIONS: Final = (52,)

#: The bundle the prosody and turn-taking features come from, under the output
#: root's `handoff/`.
BUNDLE_NAME: Final = "20260929_4d8a5a1eab8d"
