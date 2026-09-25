"""Turning similarity scores into a role assignment.

Kept separate from the audio and video handling so the rule that decides who
the psychiatrist is can be read, and tested, on its own. Nothing here touches
a file.

The rule in one sentence: **among speakers with enough speech to judge, the
psychiatrist is the one whose voice is most similar to a reference recording of
the psychiatrist, and how far ahead of the next speaker they are is reported.**

What the rule deliberately does not do:

* **No absolute threshold.** Measured on this cohort, a participant scores
  0.18-0.27 against the psychiatrist reference in the summer wave and 0.46-0.60
  in the winter wave, because the recording channels differ. Any fixed cutoff
  would therefore mean something different in each wave. Only the ranking and
  the gap carry information.
* **No tie-breaking by another method.** Where the cross-checks disagree with
  the embedding, that is recorded as a disagreement and flagged. Silently
  preferring whichever method agrees with the answer we expected would make the
  cross-check worthless.
* **No assignment on thin evidence.** A speaker with almost no speech gets the
  role `unknown`, which excludes their audio from the features, rather than a
  role inferred from a few seconds.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final

ROLE_PARTICIPANT: Final = "participant"
ROLE_PSYCHIATRIST: Final = "psychiatrist"
ROLE_UNKNOWN: Final = "unknown"

FLAG_LOW_MARGIN: Final = "speakers_low_margin"
FLAG_CLIP_DISAGREEMENT: Final = "speakers_reference_clips_disagree"
FLAG_EXTRA_SPEAKER: Final = "speakers_extra_speaker"
FLAG_INSUFFICIENT_SPEECH: Final = "speakers_insufficient_speech"
FLAG_UNMAPPED: Final = "speakers_no_mapped_reference"
FLAG_MOUTH_DISAGREEMENT: Final = "speakers_mouth_disagrees"
FLAG_MOUTH_UNUSABLE: Final = "speakers_mouth_evidence_weak"
FLAG_OCR_DISAGREEMENT: Final = "speakers_ocr_disagrees"

AGREE: Final = "agree"
DISAGREE: Final = "disagree"
UNAVAILABLE: Final = "unavailable"

#: Roles can only be assigned by comparison, so two speakers are the minimum.
_MIN_SPEAKERS: Final = 2


@dataclass(frozen=True, slots=True)
class SpeakerScore:
    """One diarized speaker, and how much they sound like each reference clip.

    Attributes:
        speaker: The diarization label, e.g. `SPEAKER_00`.
        speech_s: Total diarized speech for this speaker.
        embedded_s: How much of it was actually embedded, after short segments
            were left out and the per-speaker cap applied.
        similarities: Reference clip id to cosine similarity.
    """

    speaker: str
    speech_s: float
    embedded_s: float
    similarities: Mapping[str, float]

    def score_over(self, clips: Sequence[str]) -> float:
        """The best similarity across `clips`, or -1.0 if none are present."""
        present = [self.similarities[clip] for clip in clips if clip in self.similarities]
        return max(present) if present else -1.0

    def best_clip_over(self, clips: Sequence[str]) -> str | None:
        """Which of `clips` this speaker matches best."""
        present = {clip: self.similarities[clip] for clip in clips if clip in self.similarities}
        if not present:
            return None
        return max(present, key=lambda clip: present[clip])


@dataclass(frozen=True, slots=True)
class Decision:
    """The assignment, and everything a reader needs to judge it.

    Attributes:
        by_speaker: Speaker to role, covering every speaker seen.
        psychiatrist: The speaker assigned that role, if any.
        participant: The speaker assigned that role, if any.
        margin: Gap between the best and second-best similarity, or None when
            there was nothing to compare.
        decisive_clips: The clips the decision was taken over.
        clip_choices: What each clip would have chosen on its own.
        flags: Everything worth knowing about, in order.
    """

    by_speaker: Mapping[str, str]
    psychiatrist: str | None
    participant: str | None
    margin: float | None
    decisive_clips: tuple[str, ...]
    clip_choices: Mapping[str, str]
    flags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_complete(self) -> bool:
        """Whether both roles were assigned."""
        return self.psychiatrist is not None and self.participant is not None

    @property
    def clips_agree(self) -> bool | None:
        """Whether every clip would have chosen the same speaker.

        None with fewer than two clips, where there is nothing to agree about.
        """
        if len(self.clip_choices) < _MIN_SPEAKERS:
            return None
        return len(set(self.clip_choices.values())) == 1


def clip_choices(scores: Sequence[SpeakerScore], clips: Sequence[str]) -> dict[str, str]:
    """Which speaker each reference clip would pick as the psychiatrist.

    Every clip votes, including clips not mapped to this session, because a
    clip that disagrees is information whether or not it was expected to apply.
    """
    choices: dict[str, str] = {}
    for clip in clips:
        ranked = [score for score in scores if clip in score.similarities]
        if not ranked:
            continue
        choices[clip] = max(ranked, key=lambda score: score.similarities[clip]).speaker
    return choices


def decide(
    scores: Sequence[SpeakerScore],
    *,
    decisive_clips: Sequence[str],
    all_clips: Sequence[str],
    min_margin: float,
    min_speech_s: float,
    mapped: bool = True,
) -> Decision:
    """Assign roles from similarity scores.

    Args:
        scores: One entry per diarized speaker.
        decisive_clips: The clips that decide, normally the one mapped to this
            session. Where several are given, the best match across them wins.
        all_clips: Every configured clip, used for the agreement check.
        min_margin: Below this gap the assignment is flagged, not refused: a
            small margin means weak evidence, not wrong evidence.
        min_speech_s: Least embedded speech for a speaker to be assignable.
        mapped: Whether this session had a mapped reference clip.

    Returns:
        The decision, with flags.
    """
    flags: list[str] = []
    if not mapped:
        flags.append(FLAG_UNMAPPED)

    eligible = sorted(
        (score for score in scores if score.embedded_s >= min_speech_s),
        key=lambda score: score.score_over(decisive_clips),
        reverse=True,
    )
    choices = clip_choices(scores, all_clips)
    by_speaker: dict[str, str] = {score.speaker: ROLE_UNKNOWN for score in scores}

    if len(eligible) < _MIN_SPEAKERS:
        flags.append(FLAG_INSUFFICIENT_SPEECH)
        return Decision(
            by_speaker=by_speaker,
            psychiatrist=None,
            participant=None,
            margin=None,
            decisive_clips=tuple(decisive_clips),
            clip_choices=choices,
            flags=tuple(flags),
        )

    psychiatrist = eligible[0]
    margin = psychiatrist.score_over(decisive_clips) - eligible[1].score_over(decisive_clips)
    if margin < min_margin:
        flags.append(FLAG_LOW_MARGIN)

    # The participant is the remaining speaker who did most of the talking.
    # Using speech rather than similarity keeps a diarization artifact - a
    # sliver of a third voice - from being handed the participant's role.
    rest = [score for score in eligible if score.speaker != psychiatrist.speaker]
    participant = max(rest, key=lambda score: score.speech_s)

    by_speaker[psychiatrist.speaker] = ROLE_PSYCHIATRIST
    by_speaker[participant.speaker] = ROLE_PARTICIPANT

    extra = [score.speaker for score in rest if score.speaker != participant.speaker]
    if extra:
        flags.append(FLAG_EXTRA_SPEAKER)

    if len(choices) >= _MIN_SPEAKERS and len(set(choices.values())) > 1:
        flags.append(FLAG_CLIP_DISAGREEMENT)

    return Decision(
        by_speaker=by_speaker,
        psychiatrist=psychiatrist.speaker,
        participant=participant.speaker,
        margin=margin,
        decisive_clips=tuple(decisive_clips),
        clip_choices=choices,
        flags=tuple(flags),
    )


def role_from_tiles(
    speaker_tiles: Mapping[str, str | None],
    *,
    psychiatrist_side: str | None,
    side_of_tile: Mapping[str, str],
) -> str | None:
    """Who the psychiatrist is, going by which tile each speaker occupies.

    This is the only bridge between the visual evidence and the acoustic
    evidence. Label OCR says which *side of the frame* the psychiatrist sits
    on; diarization gives *anonymous voices*. Neither can be compared with the
    other until something says which voice belongs to which tile, and that is
    what the mouth-movement check provides.

    Args:
        speaker_tiles: Speaker to the tile it was matched to, or None.
        psychiatrist_side: The side the psychiatrist is on, from OCR or the
            configured assumption. None means unknown.
        side_of_tile: Tile name to the side of the frame it occupies.

    Returns:
        The speaker on that side, or None if the evidence does not identify
        exactly one.
    """
    if psychiatrist_side is None:
        return None
    on_side = [
        speaker
        for speaker, tile in speaker_tiles.items()
        if tile is not None and side_of_tile.get(tile) == psychiatrist_side
    ]
    return on_side[0] if len(on_side) == 1 else None


def agreement(embedding_choice: str | None, other_choice: str | None) -> str:
    """Whether two methods picked the same psychiatrist.

    `unavailable` is not a soft `agree`: it means the second method produced no
    opinion, so the embedding stands uncorroborated.
    """
    if embedding_choice is None or other_choice is None:
        return UNAVAILABLE
    return AGREE if embedding_choice == other_choice else DISAGREE
