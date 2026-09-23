"""Interval arithmetic on time spans.

Both the voice-activity stage and the turn-taking stage are mostly interval
algebra: intersect detected speech with a diarized segment, subtract one
speaker's speech from another's, find the gaps between turns, invert speech into
silence. Getting that arithmetic right is what makes response latency mean
anything, so it lives here as pure functions with no I/O and is tested directly.

Spans are half-open, `[start, end)`. Two spans that merely touch do not overlap,
which keeps adjacent spans from double-counting a boundary instant.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Final

# Spans shorter than this are treated as artifacts of floating-point arithmetic
# rather than as real intervals.
EPSILON: Final = 1e-9


@dataclass(frozen=True, slots=True, order=True)
class Span:
    """A half-open time interval `[start, end)` in seconds."""

    start: float
    end: float

    @property
    def duration(self) -> float:
        """Length in seconds; never negative."""
        return max(0.0, self.end - self.start)

    def overlaps(self, other: Span) -> bool:
        """Whether the two spans share more than an instant."""
        return self.start < other.end - EPSILON and other.start < self.end - EPSILON

    def intersection(self, other: Span) -> Span | None:
        """The shared part of two spans, or None if they do not overlap."""
        start, end = max(self.start, other.start), min(self.end, other.end)
        return Span(start, end) if end - start > EPSILON else None

    def contains(self, moment: float) -> bool:
        """Whether `moment` falls inside the span."""
        return self.start <= moment < self.end

    def shifted(self, offset: float) -> Span:
        """The same span moved later by `offset` seconds."""
        return Span(self.start + offset, self.end + offset)


def normalise(spans: Iterable[Span]) -> tuple[Span, ...]:
    """Sort spans by start time and drop the empty ones."""
    return tuple(sorted((s for s in spans if s.duration > EPSILON), key=lambda s: (s.start, s.end)))


def merge(spans: Iterable[Span], *, gap: float = 0.0) -> tuple[Span, ...]:
    """Combine overlapping spans, and those separated by at most `gap`.

    Args:
        spans: Spans to combine, in any order.
        gap: Bridge separations up to this many seconds. Zero merges only
            spans that overlap or touch.

    Returns:
        Disjoint spans in time order.
    """
    ordered = normalise(spans)
    if not ordered:
        return ()

    merged: list[Span] = [ordered[0]]
    for span in ordered[1:]:
        last = merged[-1]
        if span.start - last.end <= gap + EPSILON:
            merged[-1] = Span(last.start, max(last.end, span.end))
        else:
            merged.append(span)
    return tuple(merged)


def intersect(left: Sequence[Span], right: Sequence[Span]) -> tuple[Span, ...]:
    """Every part of `left` that is also in `right`.

    Both sides are merged first, so the result is disjoint no matter how the
    inputs overlap themselves. This is how detected speech is attributed to a
    diarized segment.
    """
    a, b = merge(left), merge(right)
    result: list[Span] = []
    i = j = 0
    while i < len(a) and j < len(b):
        shared = a[i].intersection(b[j])
        if shared is not None:
            result.append(shared)
        # Advance whichever span ends first; the other may still overlap the next.
        if a[i].end <= b[j].end:
            i += 1
        else:
            j += 1
    return tuple(result)


def subtract(left: Sequence[Span], right: Sequence[Span]) -> tuple[Span, ...]:
    """Every part of `left` that is not in `right`.

    Used to exclude overlapping speech before measuring prosody, and to turn
    "the other person is speaking" into "the participant is listening".
    """
    remaining = merge(left)
    if not remaining:
        return ()

    for cut in merge(right):
        kept: list[Span] = []
        for span in remaining:
            if not span.overlaps(cut):
                kept.append(span)
                continue
            if span.start < cut.start - EPSILON:
                kept.append(Span(span.start, cut.start))
            if cut.end < span.end - EPSILON:
                kept.append(Span(cut.end, span.end))
        remaining = tuple(kept)
        if not remaining:
            break
    return normalise(remaining)


def gaps(spans: Sequence[Span], *, minimum: float = 0.0) -> tuple[Span, ...]:
    """The silences between consecutive spans.

    Args:
        spans: Spans to look between; merged first.
        minimum: Ignore gaps shorter than this, which is how a pause threshold
            distinguishes a pause from ordinary articulation.
    """
    merged = merge(spans)
    return tuple(
        Span(earlier.end, later.start)
        for earlier, later in pairwise(merged)
        if later.start - earlier.end > max(minimum, EPSILON)
    )


def invert(spans: Sequence[Span], extent: Span) -> tuple[Span, ...]:
    """The parts of `extent` not covered by `spans`."""
    return subtract([extent], spans)


def total_duration(spans: Iterable[Span]) -> float:
    """Summed duration, counting overlaps more than once."""
    return sum(span.duration for span in spans)


def covered_duration(spans: Sequence[Span]) -> float:
    """Wall-clock time covered by at least one span, counting overlaps once."""
    return total_duration(merge(spans))


def overlap_duration(left: Sequence[Span], right: Sequence[Span]) -> float:
    """How long both sides are active at once."""
    return total_duration(intersect(left, right))


def clip(spans: Iterable[Span], extent: Span) -> tuple[Span, ...]:
    """Trim spans to `extent`, dropping anything outside it."""
    return tuple(
        clipped for span in normalise(spans) if (clipped := span.intersection(extent)) is not None
    )
