"""Tests for the role-assignment rule.

No audio, no video, no model: just the decision. The properties that matter
most are the ones about *not* deciding - a small margin, a third speaker, or a
speaker with almost no speech all have to survive contact with this rule.
"""

from __future__ import annotations

from typing import Final

import pytest

from vc_multimodal.features.assign_math import (
    AGREE,
    DISAGREE,
    FLAG_CLIP_DISAGREEMENT,
    FLAG_EXTRA_SPEAKER,
    FLAG_INSUFFICIENT_SPEECH,
    FLAG_LOW_MARGIN,
    FLAG_UNMAPPED,
    ROLE_PARTICIPANT,
    ROLE_PSYCHIATRIST,
    ROLE_UNKNOWN,
    UNAVAILABLE,
    SpeakerScore,
    agreement,
    clip_choices,
    decide,
    role_from_tiles,
)

CLIPS = ("psy_a", "psy_b")


def score(
    speaker: str,
    *,
    psy_a: float,
    psy_b: float | None = None,
    speech_s: float = 200.0,
    embedded_s: float = 120.0,
) -> SpeakerScore:
    similarities = {"psy_a": psy_a}
    if psy_b is not None:
        similarities["psy_b"] = psy_b
    return SpeakerScore(
        speaker=speaker, speech_s=speech_s, embedded_s=embedded_s, similarities=similarities
    )


def assign(*scores: SpeakerScore, **kwargs: object):
    defaults: dict[str, object] = {
        "decisive_clips": ["psy_a"],
        "all_clips": list(CLIPS),
        "min_margin": 0.10,
        "min_speech_s": 5.0,
    }
    defaults.update(kwargs)
    return decide(list(scores), **defaults)  # type: ignore[arg-type]


class TestTheRankingDecides:
    def test_the_closest_voice_is_the_psychiatrist(self) -> None:
        result = assign(
            score("SPEAKER_00", psy_a=0.84, psy_b=0.77),
            score("SPEAKER_01", psy_a=0.22, psy_b=0.19),
        )
        assert result.psychiatrist == "SPEAKER_00"
        assert result.participant == "SPEAKER_01"
        assert result.by_speaker["SPEAKER_00"] == ROLE_PSYCHIATRIST
        assert result.by_speaker["SPEAKER_01"] == ROLE_PARTICIPANT

    def test_the_margin_is_the_gap_to_the_next_speaker(self) -> None:
        result = assign(
            score("SPEAKER_00", psy_a=0.80),
            score("SPEAKER_01", psy_a=0.30),
        )
        assert result.margin == pytest.approx(0.50)

    def test_order_of_the_inputs_does_not_matter(self) -> None:
        first = assign(score("A", psy_a=0.3), score("B", psy_a=0.8))
        second = assign(score("B", psy_a=0.8), score("A", psy_a=0.3))
        assert first.psychiatrist == second.psychiatrist == "B"

    def test_there_is_no_absolute_threshold(self) -> None:
        # The property that makes this work across both waves: winter
        # participants score 0.46-0.60 against the reference where summer
        # participants score 0.18-0.27, so only the ranking can be trusted.
        # Two high scores still produce an assignment.
        result = assign(
            score("SPEAKER_00", psy_a=0.75),
            score("SPEAKER_01", psy_a=0.55),
        )
        assert result.is_complete
        assert result.psychiatrist == "SPEAKER_00"
        assert result.margin == pytest.approx(0.20)

    def test_two_low_scores_still_produce_an_assignment(self) -> None:
        result = assign(score("A", psy_a=0.31), score("B", psy_a=0.12))
        assert result.psychiatrist == "A"
        assert FLAG_LOW_MARGIN not in result.flags


class TestWeakEvidenceIsFlaggedNotHidden:
    def test_a_small_margin_is_flagged(self) -> None:
        result = assign(score("A", psy_a=0.60), score("B", psy_a=0.55))
        assert FLAG_LOW_MARGIN in result.flags
        # Flagged, not refused: weak evidence is not wrong evidence.
        assert result.is_complete

    def test_the_margin_threshold_is_configurable(self) -> None:
        scores = (score("A", psy_a=0.60), score("B", psy_a=0.40))
        assert FLAG_LOW_MARGIN not in assign(*scores, min_margin=0.10).flags
        assert FLAG_LOW_MARGIN in assign(*scores, min_margin=0.30).flags

    def test_a_speaker_with_too_little_speech_gets_no_role(self) -> None:
        result = assign(
            score("A", psy_a=0.80),
            score("B", psy_a=0.30),
            score("C", psy_a=0.75, embedded_s=2.0),
        )
        assert result.by_speaker["C"] == ROLE_UNKNOWN
        assert result.psychiatrist == "A"

    def test_one_usable_speaker_means_no_assignment_at_all(self) -> None:
        result = assign(
            score("A", psy_a=0.80),
            score("B", psy_a=0.30, embedded_s=1.0),
        )
        assert not result.is_complete
        assert result.psychiatrist is None
        assert FLAG_INSUFFICIENT_SPEECH in result.flags
        assert set(result.by_speaker.values()) == {ROLE_UNKNOWN}

    def test_no_speakers_at_all_is_not_a_crash(self) -> None:
        result = assign()
        assert not result.is_complete
        assert FLAG_INSUFFICIENT_SPEECH in result.flags

    def test_a_missing_similarity_does_not_win(self) -> None:
        # A speaker who could not be embedded has no score, and must not be
        # ranked above one who could.
        unembedded = SpeakerScore("B", speech_s=300.0, embedded_s=0.0, similarities={})
        result = assign(score("A", psy_a=0.40), unembedded)
        assert not result.is_complete  # B is ineligible, so there is no comparison


class TestAThirdSpeaker:
    def test_the_participant_is_the_one_who_talked_most(self) -> None:
        # Not the one most similar to anything: a sliver of a third voice must
        # not be handed the participant's role.
        result = assign(
            score("A", psy_a=0.85, speech_s=250.0),
            score("B", psy_a=0.30, speech_s=40.0, embedded_s=40.0),
            score("C", psy_a=0.25, speech_s=200.0, embedded_s=120.0),
        )
        assert result.psychiatrist == "A"
        assert result.participant == "C"
        assert result.by_speaker["B"] == ROLE_UNKNOWN

    def test_a_third_speaker_is_flagged(self) -> None:
        result = assign(
            score("A", psy_a=0.85),
            score("B", psy_a=0.30, speech_s=200.0),
            score("C", psy_a=0.25, speech_s=100.0),
        )
        assert FLAG_EXTRA_SPEAKER in result.flags

    def test_two_speakers_raise_no_extra_flag(self) -> None:
        result = assign(score("A", psy_a=0.85), score("B", psy_a=0.30))
        assert FLAG_EXTRA_SPEAKER not in result.flags


class TestReferenceClips:
    def test_every_clip_votes(self) -> None:
        scores = [
            score("A", psy_a=0.80, psy_b=0.20),
            score("B", psy_a=0.30, psy_b=0.90),
        ]
        assert clip_choices(scores, CLIPS) == {"psy_a": "A", "psy_b": "B"}

    def test_disagreement_between_clips_is_flagged(self) -> None:
        result = assign(
            score("A", psy_a=0.80, psy_b=0.20),
            score("B", psy_a=0.30, psy_b=0.90),
        )
        assert FLAG_CLIP_DISAGREEMENT in result.flags
        assert result.clips_agree is False

    def test_agreement_between_clips_is_recorded(self) -> None:
        result = assign(
            score("A", psy_a=0.80, psy_b=0.75),
            score("B", psy_a=0.30, psy_b=0.25),
        )
        assert result.clips_agree is True
        assert FLAG_CLIP_DISAGREEMENT not in result.flags

    def test_the_decisive_clip_decides_even_when_another_disagrees(self) -> None:
        # psy_b prefers B, but this session is mapped to psy_a.
        result = assign(
            score("A", psy_a=0.80, psy_b=0.20),
            score("B", psy_a=0.30, psy_b=0.90),
            decisive_clips=["psy_a"],
        )
        assert result.psychiatrist == "A"
        assert result.clip_choices["psy_b"] == "B"

    def test_one_clip_means_nothing_to_agree_about(self) -> None:
        result = assign(score("A", psy_a=0.80), score("B", psy_a=0.20), all_clips=["psy_a"])
        assert result.clips_agree is None

    def test_an_unmapped_session_uses_the_best_across_all_clips(self) -> None:
        result = assign(
            score("A", psy_a=0.40, psy_b=0.88),
            score("B", psy_a=0.35, psy_b=0.30),
            decisive_clips=list(CLIPS),
            mapped=False,
        )
        assert result.psychiatrist == "A"
        assert result.margin == pytest.approx(0.88 - 0.35)
        assert FLAG_UNMAPPED in result.flags

    def test_a_mapped_session_is_not_flagged_as_unmapped(self) -> None:
        result = assign(score("A", psy_a=0.8), score("B", psy_a=0.2))
        assert FLAG_UNMAPPED not in result.flags

    def test_best_clip_names_the_closest_reference(self) -> None:
        entry = score("A", psy_a=0.60, psy_b=0.85)
        assert entry.best_clip_over(CLIPS) == "psy_b"
        assert entry.best_clip_over(["psy_a"]) == "psy_a"
        assert entry.best_clip_over(["missing"]) is None


SIDES: Final = {"left_tile": "left", "right_tile": "right"}


class TestTheBridgeToTheVisualEvidence:
    def test_the_speaker_on_the_psychiatrists_side_is_identified(self) -> None:
        found = role_from_tiles(
            {"A": "left_tile", "B": "right_tile"},
            psychiatrist_side="left",
            side_of_tile=SIDES,
        )
        assert found == "A"

    def test_the_other_side_identifies_the_other_speaker(self) -> None:
        found = role_from_tiles(
            {"A": "left_tile", "B": "right_tile"},
            psychiatrist_side="right",
            side_of_tile=SIDES,
        )
        assert found == "B"

    def test_an_unknown_side_identifies_nobody(self) -> None:
        assert (
            role_from_tiles({"A": "left_tile"}, psychiatrist_side=None, side_of_tile=SIDES) is None
        )

    def test_two_speakers_on_one_side_identifies_nobody(self) -> None:
        # Both voices correlating with the same face is not evidence about
        # which of them is the psychiatrist.
        assert (
            role_from_tiles(
                {"A": "left_tile", "B": "left_tile"},
                psychiatrist_side="left",
                side_of_tile=SIDES,
            )
            is None
        )

    def test_a_speaker_matched_to_no_tile_is_ignored(self) -> None:
        found = role_from_tiles(
            {"A": "left_tile", "B": None}, psychiatrist_side="left", side_of_tile=SIDES
        )
        assert found == "A"

    def test_an_unknown_tile_name_identifies_nobody(self) -> None:
        assert (
            role_from_tiles({"A": "middle_tile"}, psychiatrist_side="left", side_of_tile=SIDES)
            is None
        )


class TestAgreement:
    def test_the_same_choice_agrees(self) -> None:
        assert agreement("SPEAKER_00", "SPEAKER_00") == AGREE

    def test_a_different_choice_disagrees(self) -> None:
        assert agreement("SPEAKER_00", "SPEAKER_01") == DISAGREE

    @pytest.mark.parametrize(("first", "second"), [(None, "A"), ("A", None), (None, None)])
    def test_a_missing_opinion_is_unavailable_not_agreement(
        self, first: str | None, second: str | None
    ) -> None:
        # An absent cross-check leaves the embedding uncorroborated, which is
        # not the same as corroborated.
        assert agreement(first, second) == UNAVAILABLE
