"""Turn-taking arithmetic.

Everything the turn features are built from, as pure functions over speech
spans and a role mapping. No filesystem, no audio, no configuration objects
beyond plain numbers, so the mathematics that matters most to this project can
be tested against hand-checked examples.

Definitions used throughout, since none of them is the only reasonable choice:

* A **turn** is a maximal run of one role's speech with no speech by the other
  role beginning in between. Short gaps inside one role's speech are bridged
  first, so drawing breath does not end a turn.
* **Response latency** is the participant's speech onset minus the preceding
  psychiatrist offset. It is negative when the participant begins before the
  psychiatrist has finished, which is an interruption rather than a fast
  response, so the two are counted separately and never averaged together.
* A **within-turn pause** is a silence inside one role's own turn, which is a
  different phenomenon from the gap between speakers and is kept apart from it.
* **Listening** means the other role is speaking and the participant is not.
  Silence belongs to neither.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Final

from vc_multimodal.features.spans import (
    Span,
    covered_duration,
    gaps,
    merge,
    overlap_duration,
    subtract,
)

ROLE_PARTICIPANT: Final = "participant"
ROLE_PSYCHIATRIST: Final = "psychiatrist"

SECONDS_PER_MINUTE: Final = 60.0

# A standard deviation needs at least two observations.
_MIN_FOR_SD: Final = 2


@dataclass(frozen=True, slots=True)
class Turn:
    """One role's uninterrupted stretch of the conversation."""

    index: int
    role: str
    span: Span
    n_spans: int = 1

    @property
    def start(self) -> float:
        """Start time in seconds."""
        return self.span.start

    @property
    def end(self) -> float:
        """End time in seconds."""
        return self.span.end

    @property
    def duration(self) -> float:
        """Length in seconds."""
        return self.span.duration


def build_turns(
    by_role: Mapping[str, Sequence[Span]], *, merge_gap: float = 0.0
) -> tuple[Turn, ...]:
    """Group speech spans into turns.

    Args:
        by_role: Role to that role's speech spans.
        merge_gap: Bridge silences up to this long inside one role's speech, so
            a breath does not end a turn.

    Returns:
        Turns in time order, numbered from zero.
    """
    labelled: list[tuple[Span, str]] = [
        (span, role) for role, spans in by_role.items() for span in merge(spans, gap=merge_gap)
    ]
    labelled.sort(key=lambda item: (item[0].start, item[0].end, item[1]))

    turns: list[Turn] = []
    for span, role in labelled:
        if turns and turns[-1].role == role:
            previous = turns[-1]
            turns[-1] = Turn(
                index=previous.index,
                role=role,
                span=Span(previous.span.start, max(previous.span.end, span.end)),
                n_spans=previous.n_spans + 1,
            )
        else:
            turns.append(Turn(index=len(turns), role=role, span=span))
    return tuple(turns)


@dataclass(frozen=True, slots=True)
class Latency:
    """One transition from the psychiatrist to the participant."""

    turn_index: int
    seconds: float

    @property
    def is_interruption(self) -> bool:
        """Whether the participant began before the psychiatrist finished."""
        return self.seconds < 0.0


def response_latencies(
    turns: Sequence[Turn],
    *,
    speaker_role: str = ROLE_PARTICIPANT,
    prior_role: str = ROLE_PSYCHIATRIST,
) -> tuple[Latency, ...]:
    """Measure every transition from `prior_role` to `speaker_role`.

    Returns every transition, including negative ones. Filtering is the
    caller's decision, so that interruptions are visible rather than quietly
    averaged into response times.
    """
    return tuple(
        Latency(turn_index=later.index, seconds=later.start - earlier.end)
        for earlier, later in pairwise(turns)
        if earlier.role == prior_role and later.role == speaker_role
    )


def usable_latencies(latencies: Sequence[Latency], *, max_latency_s: float) -> tuple[float, ...]:
    """Response latencies that are actually responses.

    Negative values are interruptions, not fast responses. Values beyond
    `max_latency_s` are not responses either: after a long silence the
    participant is starting something, not answering.
    """
    return tuple(
        latency.seconds for latency in latencies if 0.0 <= latency.seconds <= max_latency_s
    )


def count_interruptions(latencies: Sequence[Latency]) -> int:
    """How many transitions began before the previous speaker finished."""
    return sum(1 for latency in latencies if latency.is_interruption)


def within_turn_pauses(
    turns: Sequence[Turn],
    spans: Sequence[Span],
    *,
    role: str,
    min_pause_s: float,
) -> tuple[float, ...]:
    """Silences inside one role's own turns.

    Args:
        turns: The turn structure.
        spans: That role's unmerged speech spans, so pauses bridged when the
            turns were built are still visible here.
        role: Which role's turns to look inside.
        min_pause_s: Ignore silences shorter than this, which are articulation
            rather than pausing.

    Returns:
        Pause durations in seconds.
    """
    merged = merge(spans)
    pauses: list[float] = []
    for turn in turns:
        if turn.role != role:
            continue
        inside = [span for span in merged if span.start >= turn.start and span.end <= turn.end]
        pauses.extend(gap.duration for gap in gaps(inside, minimum=min_pause_s))
    return tuple(pauses)


def speaking_ratio(
    by_role: Mapping[str, Sequence[Span]], *, role: str = ROLE_PARTICIPANT
) -> float | None:
    """One role's share of all speech, counting overlap once per speaker.

    Returns None when nobody spoke, rather than zero, which would claim the
    participant was silent while the other person talked.
    """
    totals = {name: covered_duration(spans) for name, spans in by_role.items()}
    everyone = sum(totals.values())
    if everyone <= 0.0:
        return None
    return totals.get(role, 0.0) / everyone


def overlap_ratio(by_role: Mapping[str, Sequence[Span]]) -> float | None:
    """Share of speech time in which both roles were speaking at once."""
    participant = by_role.get(ROLE_PARTICIPANT, ())
    psychiatrist = by_role.get(ROLE_PSYCHIATRIST, ())
    total = covered_duration(list(participant) + list(psychiatrist))
    if total <= 0.0:
        return None
    return overlap_duration(participant, psychiatrist) / total


@dataclass(frozen=True, slots=True)
class SpeakingTimeline:
    """When the participant is speaking, and when they are listening.

    Listening is the other role speaking while the participant is not. Mutual
    silence belongs to neither, so the two never overlap and rarely sum to the
    whole recording. The facial features are summarised separately over each.
    """

    speaking: tuple[Span, ...]
    listening: tuple[Span, ...]

    @property
    def speaking_seconds(self) -> float:
        """Total time the participant is speaking."""
        return covered_duration(self.speaking)

    @property
    def listening_seconds(self) -> float:
        """Total time the participant is listening."""
        return covered_duration(self.listening)


def speaking_timeline(by_role: Mapping[str, Sequence[Span]]) -> SpeakingTimeline:
    """Split the session into participant-speaking and participant-listening."""
    participant = merge(by_role.get(ROLE_PARTICIPANT, ()))
    psychiatrist = merge(by_role.get(ROLE_PSYCHIATRIST, ()))
    return SpeakingTimeline(
        speaking=participant,
        listening=subtract(psychiatrist, participant),
    )


def _mean(values: Sequence[float]) -> float | None:
    """Arithmetic mean, or None for no values."""
    return sum(values) / len(values) if values else None


def _median(values: Sequence[float]) -> float | None:
    """Median, or None for no values."""
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _sd(values: Sequence[float]) -> float | None:
    """Sample standard deviation, or None for fewer than two values."""
    if len(values) < _MIN_FOR_SD:
        return None
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def _per_minute(count: float, seconds: float) -> float | None:
    """A rate per minute, or None when there is no time to divide by."""
    if seconds <= 0.0:
        return None
    return count / (seconds / SECONDS_PER_MINUTE)


def turn_features(
    by_role: Mapping[str, Sequence[Span]],
    *,
    duration_s: float,
    merge_gap_s: float,
    min_pause_s: float,
    max_latency_s: float,
) -> dict[str, float | None]:
    """Compute the session-level turn-taking features.

    Deliberately a small set: with 62 sessions, every added feature costs
    statistical power (docs/decisions/0006). Durations are expressed as ratios
    and per-minute rates rather than absolute seconds, because session length
    varies from 4 to 16 minutes and raw totals would mostly measure that.

    Args:
        by_role: Role to that role's speech spans.
        duration_s: Length of the recording, for the rates.
        merge_gap_s: Silence bridged inside one role's speech when building
            turns.
        min_pause_s: Shortest silence counted as a within-turn pause.
        max_latency_s: Longest gap still counted as a response.

    Returns:
        Feature name to value, with None where a session offers no basis for
        the measure. Names follow the `turns__` convention.
    """
    turns = build_turns(by_role, merge_gap=merge_gap_s)
    latencies = response_latencies(turns)
    usable = usable_latencies(latencies, max_latency_s=max_latency_s)

    participant_turns = [turn.duration for turn in turns if turn.role == ROLE_PARTICIPANT]
    psychiatrist_turns = [turn.duration for turn in turns if turn.role == ROLE_PSYCHIATRIST]
    pauses = within_turn_pauses(
        turns,
        by_role.get(ROLE_PARTICIPANT, ()),
        role=ROLE_PARTICIPANT,
        min_pause_s=min_pause_s,
    )
    participant_speech_s = covered_duration(by_role.get(ROLE_PARTICIPANT, ()))

    return {
        "turns__n_per_minute": _per_minute(len(turns), duration_s),
        "turns__participant_speaking_ratio": speaking_ratio(by_role),
        "turns__overlap_ratio": overlap_ratio(by_role),
        "turns__latency_mean": _mean(usable),
        "turns__latency_median": _median(usable),
        "turns__latency_sd": _sd(usable),
        "turns__interruption_rate": _per_minute(count_interruptions(latencies), duration_s),
        "turns__participant_turn_duration_mean": _mean(participant_turns),
        "turns__participant_turn_duration_sd": _sd(participant_turns),
        "turns__psychiatrist_turn_duration_mean": _mean(psychiatrist_turns),
        "turns__pause_within_mean": _mean(pauses),
        "turns__pause_within_rate": _per_minute(len(pauses), participant_speech_s),
    }


FEATURE_NAMES: Final = tuple(
    turn_features({}, duration_s=1.0, merge_gap_s=0.0, min_pause_s=0.1, max_latency_s=10.0)
)
