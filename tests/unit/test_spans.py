"""Interval arithmetic.

Response latency is a subtraction between interval endpoints, so an error here
would propagate silently into every timing feature. The operations are checked
against hand-worked examples and against each other.
"""

from __future__ import annotations

import pytest

from vc_multimodal.features.spans import (
    Span,
    clip,
    covered_duration,
    gaps,
    intersect,
    invert,
    merge,
    normalise,
    overlap_duration,
    subtract,
    total_duration,
)


def spans(*pairs: tuple[float, float]) -> list[Span]:
    return [Span(start, end) for start, end in pairs]


def as_pairs(result: tuple[Span, ...]) -> list[tuple[float, float]]:
    return [(span.start, span.end) for span in result]


# ---------------------------------------------------------------------------
# a single span
# ---------------------------------------------------------------------------
def test_duration():
    assert Span(1.0, 3.5).duration == pytest.approx(2.5)


def test_a_reversed_span_has_no_duration_rather_than_a_negative_one():
    assert Span(3.0, 1.0).duration == 0.0


def test_touching_spans_do_not_overlap():
    """Half-open intervals: otherwise adjacent spans double-count a boundary."""
    assert not Span(0.0, 1.0).overlaps(Span(1.0, 2.0))
    assert Span(0.0, 1.0).intersection(Span(1.0, 2.0)) is None


def test_overlapping_spans_intersect():
    assert Span(0.0, 2.0).intersection(Span(1.0, 3.0)) == Span(1.0, 2.0)


def test_containment_uses_a_half_open_interval():
    span = Span(1.0, 2.0)
    assert span.contains(1.0)
    assert span.contains(1.999)
    assert not span.contains(2.0)


def test_shifting():
    assert Span(1.0, 2.0).shifted(0.5) == Span(1.5, 2.5)


def test_spans_sort_by_start_then_end():
    assert sorted([Span(1.0, 5.0), Span(0.0, 9.0), Span(1.0, 2.0)])[0] == Span(0.0, 9.0)


# ---------------------------------------------------------------------------
# normalise and merge
# ---------------------------------------------------------------------------
def test_normalise_sorts_and_drops_empty_spans():
    result = normalise(spans((5.0, 6.0), (1.0, 1.0), (2.0, 3.0)))
    assert as_pairs(result) == [(2.0, 3.0), (5.0, 6.0)]


def test_merge_combines_overlapping_spans():
    assert as_pairs(merge(spans((0.0, 2.0), (1.0, 3.0)))) == [(0.0, 3.0)]


def test_merge_combines_touching_spans():
    assert as_pairs(merge(spans((0.0, 1.0), (1.0, 2.0)))) == [(0.0, 2.0)]


def test_merge_leaves_separated_spans_alone():
    assert len(merge(spans((0.0, 1.0), (2.0, 3.0)))) == 2


def test_merge_bridges_a_gap_within_tolerance():
    """This is how a breath inside one speaker's turn stops ending the turn."""
    assert as_pairs(merge(spans((0.0, 1.0), (1.2, 2.0)), gap=0.3)) == [(0.0, 2.0)]


def test_merge_does_not_bridge_a_gap_beyond_tolerance():
    assert len(merge(spans((0.0, 1.0), (1.4, 2.0)), gap=0.3)) == 2


def test_merge_absorbs_a_contained_span():
    assert as_pairs(merge(spans((0.0, 10.0), (2.0, 3.0)))) == [(0.0, 10.0)]


def test_merge_of_nothing():
    assert merge([]) == ()


def test_merge_is_idempotent():
    once = merge(spans((0.0, 2.0), (1.0, 3.0), (5.0, 6.0)))
    assert merge(once) == once


# ---------------------------------------------------------------------------
# intersect
# ---------------------------------------------------------------------------
def test_intersect_keeps_only_shared_time():
    result = intersect(spans((0.0, 2.0), (3.0, 5.0)), spans((1.0, 4.0)))
    assert as_pairs(result) == [(1.0, 2.0), (3.0, 4.0)]


def test_intersect_with_nothing_is_nothing():
    assert intersect(spans((0.0, 1.0)), []) == ()
    assert intersect([], spans((0.0, 1.0))) == ()


def test_intersect_of_disjoint_sets_is_empty():
    assert intersect(spans((0.0, 1.0)), spans((2.0, 3.0))) == ()


def test_intersect_is_symmetric():
    left, right = spans((0.0, 2.0), (3.0, 5.0)), spans((1.0, 4.0))
    assert intersect(left, right) == intersect(right, left)


def test_intersect_handles_one_span_covering_many():
    result = intersect(spans((0.0, 10.0)), spans((1.0, 2.0), (3.0, 4.0), (5.0, 6.0)))
    assert as_pairs(result) == [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]


def test_intersect_merges_self_overlapping_inputs_first():
    """So a doubled input cannot inflate the result."""
    doubled = spans((0.0, 2.0), (0.0, 2.0))
    assert total_duration(intersect(doubled, spans((0.0, 2.0)))) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# subtract
# ---------------------------------------------------------------------------
def test_subtract_removes_the_middle():
    assert as_pairs(subtract(spans((0.0, 5.0)), spans((2.0, 3.0)))) == [(0.0, 2.0), (3.0, 5.0)]


def test_subtract_trims_the_start_and_the_end():
    assert as_pairs(subtract(spans((0.0, 5.0)), spans((0.0, 1.0)))) == [(1.0, 5.0)]
    assert as_pairs(subtract(spans((0.0, 5.0)), spans((4.0, 9.0)))) == [(0.0, 4.0)]


def test_subtracting_everything_leaves_nothing():
    assert subtract(spans((1.0, 2.0)), spans((0.0, 5.0))) == ()


def test_subtracting_nothing_changes_nothing():
    assert as_pairs(subtract(spans((1.0, 2.0)), [])) == [(1.0, 2.0)]


def test_subtract_handles_several_cuts():
    result = subtract(spans((0.0, 10.0)), spans((1.0, 2.0), (4.0, 5.0), (8.0, 9.0)))
    assert as_pairs(result) == [(0.0, 1.0), (2.0, 4.0), (5.0, 8.0), (9.0, 10.0)]


def test_subtract_is_how_listening_is_defined():
    """Listening is the other speaking while the participant is not."""
    psychiatrist = spans((0.0, 10.0))
    participant = spans((4.0, 6.0))
    assert as_pairs(subtract(psychiatrist, participant)) == [(0.0, 4.0), (6.0, 10.0)]


# ---------------------------------------------------------------------------
# gaps and invert
# ---------------------------------------------------------------------------
def test_gaps_finds_the_silences_between_spans():
    assert as_pairs(gaps(spans((0.0, 1.0), (2.0, 3.0), (5.0, 6.0)))) == [(1.0, 2.0), (3.0, 5.0)]


def test_gaps_ignores_short_silences():
    """A pause threshold separates pausing from ordinary articulation."""
    result = gaps(spans((0.0, 1.0), (1.1, 2.0), (3.0, 4.0)), minimum=0.2)
    assert as_pairs(result) == [(2.0, 3.0)]


def test_a_single_span_has_no_gaps():
    assert gaps(spans((0.0, 1.0))) == ()


def test_overlapping_spans_have_no_gaps():
    assert gaps(spans((0.0, 2.0), (1.0, 3.0))) == ()


def test_invert_returns_the_uncovered_parts_of_an_extent():
    assert as_pairs(invert(spans((1.0, 2.0)), Span(0.0, 3.0))) == [(0.0, 1.0), (2.0, 3.0)]


def test_inverting_full_coverage_gives_nothing():
    assert invert(spans((0.0, 3.0)), Span(0.0, 3.0)) == ()


def test_inverting_nothing_gives_the_whole_extent():
    assert as_pairs(invert([], Span(0.0, 3.0))) == [(0.0, 3.0)]


# ---------------------------------------------------------------------------
# durations
# ---------------------------------------------------------------------------
def test_total_duration_counts_overlap_twice():
    assert total_duration(spans((0.0, 2.0), (1.0, 3.0))) == pytest.approx(4.0)


def test_covered_duration_counts_overlap_once():
    assert covered_duration(spans((0.0, 2.0), (1.0, 3.0))) == pytest.approx(3.0)


def test_overlap_duration_is_the_shared_time():
    assert overlap_duration(spans((0.0, 2.0)), spans((1.0, 3.0))) == pytest.approx(1.0)


def test_the_three_durations_agree():
    left, right = spans((0.0, 3.0)), spans((2.0, 5.0))
    both = left + right
    assert total_duration(both) - covered_duration(both) == pytest.approx(
        overlap_duration(left, right)
    )


def test_durations_of_nothing_are_zero():
    assert total_duration([]) == 0.0
    assert covered_duration([]) == 0.0
    assert overlap_duration([], []) == 0.0


# ---------------------------------------------------------------------------
# clip
# ---------------------------------------------------------------------------
def test_clip_trims_to_an_extent():
    result = clip(spans((-1.0, 1.0), (2.0, 3.0), (9.0, 12.0)), Span(0.0, 10.0))
    assert as_pairs(result) == [(0.0, 1.0), (2.0, 3.0), (9.0, 10.0)]


def test_clip_drops_spans_outside_the_extent():
    assert clip(spans((20.0, 30.0)), Span(0.0, 10.0)) == ()


def test_clip_is_how_a_truncated_recording_is_handled():
    """Nothing may claim speech beyond the audio that actually decoded."""
    decoded = Span(0.0, 46.3)
    assert as_pairs(clip(spans((40.0, 400.0)), decoded)) == [(40.0, 46.3)]
