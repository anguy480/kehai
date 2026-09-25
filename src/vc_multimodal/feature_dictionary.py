"""What every column in the handoff table means.

The person who runs the analysis did not write the extraction code and will not
read it. A column they cannot interpret is a column they will either misread or
drop, so the bundle carries a description of each one, and this module refuses
to describe a column it does not know.

That refusal is the point. A generated feature name is easy to add and easy to
forget to document; here, adding a feature without saying what it means stops
the handoff rather than shipping an unexplained column.

Descriptions are written from the definitions in `features/turn_math.py`,
`features/prosody_math.py` and `features/aggregate_math.py`, including the
awkward parts - which denominator a rate uses, what a None means, and where a
value is constant because of how the data were produced.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import pandas as pd

from vc_multimodal.config import AppConfig
from vc_multimodal.features.aggregate_math import stats_plan

#: Turn-taking features. Rates differ in their denominator, which is stated,
#: because "per minute" alone would be ambiguous between session length and
#: speaking time.
TURN_DESCRIPTIONS: Final[dict[str, tuple[str, str]]] = {
    "turns__n_per_minute": (
        "turns/min",
        "Turns per minute of recording. A turn is a maximal run of one role's "
        "speech with short gaps bridged, so this measures conversational pace "
        "rather than the number of diarized segments.",
    ),
    "turns__participant_speaking_ratio": (
        "ratio",
        "The participant's share of all speech time, counting overlap once per "
        "speaker. Empty when nobody spoke, rather than zero, which would claim "
        "the participant was silent while the other person talked.",
    ),
    "turns__overlap_ratio": (
        "ratio",
        "Share of speech time in which both roles spoke at once. CONSTANT AT "
        "ZERO in this bundle: the diarizer partitions time, so simultaneous "
        "speech cannot be represented. See the README.",
    ),
    "turns__latency_mean": (
        "seconds",
        "Mean gap between one role finishing and the other starting, over "
        "transitions that are genuine responses: interruptions (negative gaps) "
        "and gaps longer than the configured maximum are excluded, not averaged "
        "in.",
    ),
    "turns__latency_median": (
        "seconds",
        "Median response gap, over the same transitions as the mean. Preferred "
        "over the mean for a short session, where one long pause moves the mean "
        "and not the median.",
    ),
    "turns__latency_sd": (
        "seconds",
        "Standard deviation of the response gaps: how variable the participant's "
        "timing is, rather than how fast.",
    ),
    "turns__interruption_rate": (
        "events/min",
        "Transitions per minute of recording that began before the previous "
        "speaker finished. Counted separately from response latency, never "
        "averaged into it. CONSTANT AT ZERO in this bundle, for the same reason "
        "as the overlap ratio.",
    ),
    "turns__participant_turn_duration_mean": (
        "seconds",
        "Mean length of the participant's turns.",
    ),
    "turns__participant_turn_duration_sd": (
        "seconds",
        "Standard deviation of the participant's turn lengths.",
    ),
    "turns__psychiatrist_turn_duration_mean": (
        "seconds",
        "Mean length of the psychiatrist's turns. Included as a check on the "
        "interviewer's behaviour, not as a participant measure: a session where "
        "this is unusual is one to look at before interpreting the rest.",
    ),
    "turns__pause_within_mean": (
        "seconds",
        "Mean length of silences inside the participant's own turns, counting "
        "only silences longer than the configured minimum.",
    ),
    "turns__pause_within_rate": (
        "events/min",
        "Within-turn pauses per minute OF THE PARTICIPANT'S SPEECH, not per "
        "minute of recording: a participant who speaks for two minutes of a ten "
        "minute session is described by their own speech, not by the silence "
        "around it.",
    ),
}

#: Prosodic features, measured inside the participant's diarized speech only.
PROSODY_DESCRIPTIONS: Final[dict[str, tuple[str, str]]] = {
    "prosody__f0_semitone_sd": (
        "semitones",
        "Variability of pitch, in semitones relative to the speaker's own median "
        "F0. Reduced pitch variability is the most replicated acoustic correlate "
        "of autistic-type speech, which is why it is a primary feature.",
    ),
    "prosody__f0_semitone_iqr": (
        "semitones",
        "Interquartile range of pitch in semitones: the same construct as the "
        "standard deviation, robust to a few badly tracked frames.",
    ),
    "prosody__f0_semitone_range": (
        "semitones",
        "Pitch range in semitones, measured between percentiles rather than "
        "extremes, so one octave-error frame does not define it.",
    ),
    "prosody__f0_semitone_mean_abs_delta": (
        "semitones",
        "Mean absolute pitch change between consecutive voiced frames: how much "
        "the pitch moves moment to moment, as opposed to how wide it ranges "
        "overall.",
    ),
    "prosody__voiced_fraction": (
        "ratio",
        "Fraction of analysed frames with a detected pitch. Low values indicate "
        "a quiet or noisy recording as readily as a breathy speaker, so this is "
        "as much a QC measure as a prosodic one.",
    ),
    "prosody__intensity_mean_db": (
        "dB",
        "Mean intensity. NOT COMPARABLE ACROSS SESSIONS in absolute terms: Zoom "
        "applies its own gain, so this reflects the recording chain as well as "
        "the speaker. Included for completeness; prefer the spread measures.",
    ),
    "prosody__intensity_sd_db": (
        "dB",
        "Variability of intensity within the participant's speech. Unlike the "
        "mean, a spread is largely unaffected by a constant gain.",
    ),
    "prosody__intensity_range_db": (
        "dB",
        "Intensity range, measured between percentiles rather than extremes.",
    ),
    "prosody__jitter_local": (
        "ratio",
        "Local jitter: cycle-to-cycle variation in pitch period, pooled over the "
        "participant's speech weighted by the length of each span.",
    ),
    "prosody__shimmer_local": (
        "ratio",
        "Local shimmer: cycle-to-cycle variation in amplitude, pooled the same way as jitter.",
    ),
    "prosody__hnr_db": (
        "dB",
        "Harmonics-to-noise ratio: voice quality, lower being breathier or "
        "noisier. Sensitive to background noise as well as to the voice.",
    ),
    "prosody__speech_rate_proxy": (
        "syllables/s",
        "Syllable nuclei per second of the participant's speech, from intensity "
        "peaks. A PROXY, not a syllable count: it is not validated against "
        "Japanese phonology and should be read as relative, not absolute.",
    ),
}

#: How each window is described.
WINDOW_DESCRIPTIONS: Final[dict[str, str]] = {
    "face_speaking": "while the participant was speaking",
    "face_listening": "while the psychiatrist was speaking and the participant was not",
}

#: How each statistic is described.
STAT_DESCRIPTIONS: Final[dict[str, str]] = {
    "mean": "Mean of",
    "sd": "Standard deviation of",
    "p90": "90th percentile of",
}

#: Head pose measures, which are not action units and are not gaze.
POSE_DESCRIPTIONS: Final[dict[str, str]] = {
    "head_pitch": "head pitch (nodding, rotation about the left-right axis)",
    "head_yaw": "head yaw (turning, rotation about the vertical axis)",
    "head_roll": "head roll (tilting, rotation about the viewing axis)",
}

#: The QC columns, which are not features and must not be modelled.
QC_DESCRIPTIONS: Final[dict[str, tuple[str, str]]] = {
    "session_id": ("id", "Session identifier. One session is one participant."),
    "wave": ("category", "Recruitment wave, winter or summer."),
    "qc__stages_missing": (
        "list",
        "Pipeline stages that produced nothing for this session, semicolon "
        "separated. A row with anything here has missing features by cause, not "
        "by chance.",
    ),
    "qc__role_source": (
        "category",
        "How the diarized speakers were assigned to roles. Sessions differing "
        "here were not established the same way.",
    ),
    "qc__face_backend": (
        "category",
        "Which facial backend measured this session. MediaPipe blendshape scores "
        "and OpenFace action unit intensities are NOT comparable; a table mixing "
        "them describes nothing. See the README.",
    ),
    "qc__speaking_seconds": ("seconds", "Total participant speech the features rest on."),
    "qc__listening_seconds": (
        "seconds",
        "Total time the psychiatrist spoke and the participant did not.",
    ),
    "qc__face_frames_speaking": (
        "count",
        "Frames measured in the speaking window. A small count makes that "
        "window's features noisy however clean they look.",
    ),
    "qc__face_frames_listening": ("count", "Frames measured in the listening window."),
    "qc__face_measured_speaking": (
        "ratio",
        "Fraction of the speaking window in which a face was found and measured. "
        "The honest denominator for the speaking features.",
    ),
    "qc__face_measured_listening": (
        "ratio",
        "Fraction of the listening window in which a face was found and measured.",
    ),
    "qc__flags": (
        "list",
        "Quality flags raised for this session, semicolon separated. Flags are "
        "reported rather than acted on: which sessions to exclude is the "
        "analyst's decision, and it should be made once and stated.",
    ),
}


class DictionaryError(ValueError):
    """Raised when a column in the table has no description."""


@dataclass(frozen=True, slots=True)
class Entry:
    """One row of the feature dictionary."""

    name: str
    family: str
    unit: str
    tier: str
    description: str


def _face_entries(config: AppConfig) -> dict[str, tuple[str, str]]:
    """Descriptions for the generated facial columns.

    Built from the same plan that generates the names, rather than by parsing
    them, so a name and its description cannot drift apart.
    """
    described: dict[str, tuple[str, str]] = {}
    plan = stats_plan(
        config.face.unit_keys,
        config.aggregate.peak_action_units,
        config.aggregate.pose_measures,
    )
    for window, when in WINDOW_DESCRIPTIONS.items():
        for measure, stats in plan.items():
            if measure in POSE_DESCRIPTIONS:
                what = POSE_DESCRIPTIONS[measure]
                unit = "degrees"
                extra = (
                    " Head pose is NOT gaze and must not be read as a proxy for it; see the README."
                )
            else:
                unit_config = config.face.unit(measure)
                if unit_config is None:
                    msg = (
                        f"measure {measure!r} is neither a configured action unit nor a "
                        f"known head pose measure, so it cannot be described"
                    )
                    raise DictionaryError(msg)
                what = f"{measure.upper()} ({unit_config.description})"
                unit = "backend score"
                extra = ""
            for stat in stats:
                name = f"{window}__{measure}_{stat}"
                lead = STAT_DESCRIPTIONS[stat]
                peak = (
                    " A high percentile rather than a mean, because these "
                    "distributions are mostly zero: an expression that happens "
                    "sometimes is visible in the peak and averaged away in the mean."
                    if stat == "p90"
                    else ""
                )
                described[name] = (unit, f"{lead} {what}, {when}.{peak}{extra}")
    return described


def descriptions(config: AppConfig) -> dict[str, tuple[str, str]]:
    """Every column this project can describe, as name to (unit, description)."""
    return {
        **QC_DESCRIPTIONS,
        **TURN_DESCRIPTIONS,
        **PROSODY_DESCRIPTIONS,
        **_face_entries(config),
    }


def family_of(name: str) -> str:
    """The feature family a column belongs to."""
    if name.startswith("qc__"):
        return "qc"
    if "__" not in name:
        return "identifier"
    return name.split("__", 1)[0]


def entries(config: AppConfig, columns: Sequence[str]) -> tuple[Entry, ...]:
    """Describe every column, in table order.

    Raises:
        DictionaryError: if any column has no description. Shipping an
            undescribed column is worse than failing here: the analyst either
            guesses at it or drops it, and neither is visible in the results.
    """
    known = descriptions(config)
    primary = set(config.model.tiers.primary_columns)

    undescribed = [name for name in columns if name not in known and not name.startswith("text__")]
    if undescribed:
        msg = (
            f"{len(undescribed)} column(s) in the feature table have no description: "
            f"{undescribed}. Add them to feature_dictionary.py - a column the analyst "
            f"cannot interpret will be misread or dropped, and neither shows up in the "
            f"results."
        )
        raise DictionaryError(msg)

    built: list[Entry] = []
    for name in columns:
        family = family_of(name)
        if name.startswith("text__"):
            unit, description = (
                "manuscript units",
                "Text feature from the lab's manuscript, carried through unchanged. "
                "See that manuscript for its definition.",
            )
        else:
            unit, description = known[name]
        if family in {"qc", "identifier"}:
            tier = "not a feature"
        elif name in primary:
            tier = "confirmatory"
        else:
            tier = "exploratory"
        built.append(Entry(name=name, family=family, unit=unit, tier=tier, description=description))
    return tuple(built)


def build(config: AppConfig, columns: Sequence[str]) -> pd.DataFrame:
    """The feature dictionary as a table."""
    rows = [
        {
            "name": entry.name,
            "family": entry.family,
            "unit": entry.unit,
            "tier": entry.tier,
            "description": entry.description,
        }
        for entry in entries(config, columns)
    ]
    return pd.DataFrame(rows, columns=["name", "family", "unit", "tier", "description"])
