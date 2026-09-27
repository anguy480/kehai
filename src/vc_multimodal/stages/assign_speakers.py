"""Decide which diarized speaker is the participant.

Everything downstream depends on this. Prosody is measured on the
participant's speech only; the speaking and listening windows for the facial
features are defined by who is talking. Swap the two speakers in one session
and that session's features describe the wrong person while looking completely
normal - no contract fails, no metric complains.

So this stage gathers three independent lines of evidence and reports all of
them rather than producing a bare answer:

1. **Speaker embeddings against reference clips of the psychiatrist.** The
   primary method, and the decisive one. It is the only method that identifies
   the psychiatrist *directly* rather than by position or by inference.
2. **Mouth movement per video tile.** Correlating each speaker's speech
   timeline against jaw movement in every tile says which voice belongs to
   which tile. This is also the only bridge between the acoustic evidence and
   the visual evidence: see `_ocr_cross_check`.
3. **Zoom name labels, from `vc verify-layout`.** OCR knows which *side of the
   frame* the psychiatrist sits on. It knows nothing about voices, so it can
   only corroborate an assignment through (2).

Where these disagree, the disagreement is flagged and the embedding stands.
Preferring whichever cross-check agreed with the expected answer would make the
cross-checks worthless, so they are recorded, not consulted.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import pandas as pd

from vc_multimodal.config import AppConfig, CropBox
from vc_multimodal.embeddings import (
    EmbeddingError,
    SpeakerEmbedder,
    cosine_similarity,
    get_embedder,
)
from vc_multimodal.faces import FaceError
from vc_multimodal.faces import get_backend as get_face_backend
from vc_multimodal.features.assign_math import (
    AGREE,
    DISAGREE,
    FLAG_MOUTH_DISAGREEMENT,
    FLAG_MOUTH_UNUSABLE,
    FLAG_OCR_DISAGREEMENT,
    UNAVAILABLE,
    Decision,
    SpeakerScore,
    agreement,
    decide,
    role_from_tiles,
)
from vc_multimodal.features.sampling import SamplingError, resolve_sampling
from vc_multimodal.features.spans import Span, covered_duration, merge
from vc_multimodal.ffmpeg import FfmpegError, FfmpegTools, parse_media_info
from vc_multimodal.io_utils import read_csv, write_csv
from vc_multimodal.logging_setup import get_logger
from vc_multimodal.paths import DataRoots, RawSession, discover_sessions, select_sessions
from vc_multimodal.roles import write_role_mapping
from vc_multimodal.runner import StageReport, run_sessions
from vc_multimodal.session_tables import ID_COLUMN, carry_forward, combine
from vc_multimodal.stages import diarize as diarize_stage
from vc_multimodal.stages import extract_audio as extract_audio_stage
from vc_multimodal.stages import verify_layout as layout_stage
from vc_multimodal.stages.vad import read_mono_wav, segments_by_speaker

logger = get_logger(__name__)

STAGE: Final = "assign-speakers"
SPEAKERS_FILENAME: Final = "speakers.csv"
REFERENCE_DIRNAME: Final = "reference"

#: Correlation is undefined for a constant series, and a tile where the face
#: was never found produces one.
_MIN_VARIANCE: Final = 1e-9

#: Fewest points a correlation can be computed over.
_MIN_SERIES: Final = 2

#: Longest list of session ids printed for an unremarkable outcome.
_MAX_LISTED: Final = 12


class AssignError(RuntimeError):
    """Raised when speaker assignment cannot run at all."""


# ---------------------------------------------------------------------------
# Reference clips
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ReferenceVoice:
    """One psychiatrist reference clip, embedded."""

    clip_id: str
    filename: str
    duration_s: float
    embedding: np.ndarray


@dataclass(frozen=True, slots=True)
class ReferenceReport:
    """How the reference clips relate to each other.

    The question this answers is whether the two recurring Zoom labels that OCR
    found are two psychiatrists or one who renamed themselves. It is answered
    with controls rather than with a bare number: `within_clip` compares two
    halves of the *same* clip, which is the best similarity this model achieves
    on this material, and `between_clips` is read against that.
    """

    voices: tuple[ReferenceVoice, ...]
    within_clip: Mapping[str, float]
    between_clips: Mapping[tuple[str, str], float]

    def report_lines(self) -> list[str]:
        """The comparison, for the stage summary."""
        lines = ["reference clips:"]
        for voice in self.voices:
            control = self.within_clip.get(voice.clip_id)
            shown = f"{control:+.3f}" if control is not None else "n/a"
            lines.append(
                f"  {voice.clip_id}: {voice.duration_s:.1f}s, own two halves agree at {shown}"
            )
        for (first, second), value in sorted(self.between_clips.items()):
            lines.append(f"  {first} vs {second}: {value:+.3f}")
        if self.between_clips and self.within_clip:
            worst_control = min(self.within_clip.values())
            highest_pair = max(self.between_clips.values())
            if highest_pair >= worst_control:
                lines.append(
                    "  the clips resemble each other as closely as either resembles "
                    "itself, which is what one person recorded twice looks like"
                )
            else:
                lines.append(
                    "  the clips resemble each other less closely than either "
                    "resembles itself, which is what two people look like"
                )
        return lines


def reference_dir(roots: DataRoots) -> Path:
    """Where the psychiatrist reference clips live."""
    return roots.work / REFERENCE_DIRNAME


def load_references(
    config: AppConfig, roots: DataRoots, embedder: SpeakerEmbedder
) -> ReferenceReport:
    """Embed every configured reference clip, with a within-clip control.

    Raises:
        AssignError: if no clip is configured or a configured clip is missing.
    """
    clips = config.speakers.reference_clips
    if not clips:
        msg = (
            "no psychiatrist reference clips are configured. Add them under "
            "speakers.reference_clips and place the audio in "
            "$VC_WORK_ROOT/reference/."
        )
        raise AssignError(msg)

    voices: list[ReferenceVoice] = []
    within: dict[str, float] = {}
    for clip in clips:
        path = reference_dir(roots) / clip.path
        if not path.is_file():
            msg = (
                f"reference clip {clip.path!r} for {clip.psychiatrist_id!r} is not in "
                f"$VC_WORK_ROOT/{REFERENCE_DIRNAME}/"
            )
            raise AssignError(msg)
        try:
            samples, sample_rate = read_mono_wav(path)
        except (OSError, ValueError) as exc:
            msg = f"could not read reference clip {clip.path!r}: {exc}"
            raise AssignError(msg) from exc

        try:
            embedding = embedder.embed(samples, sample_rate)
            half = len(samples) // 2
            control = cosine_similarity(
                embedder.embed(samples[:half], sample_rate),
                embedder.embed(samples[half:], sample_rate),
            )
        except EmbeddingError as exc:
            msg = f"could not embed reference clip {clip.path!r}: {exc}"
            raise AssignError(msg) from exc

        voices.append(
            ReferenceVoice(
                clip_id=clip.psychiatrist_id,
                filename=clip.path,
                duration_s=len(samples) / float(sample_rate),
                embedding=embedding,
            )
        )
        within[clip.psychiatrist_id] = control

    between: dict[tuple[str, str], float] = {}
    for i, first in enumerate(voices):
        for second in voices[i + 1 :]:
            between[first.clip_id, second.clip_id] = cosine_similarity(
                first.embedding, second.embedding
            )
    return ReferenceReport(voices=tuple(voices), within_clip=within, between_clips=between)


def load_psychiatrist_map(config: AppConfig, roots: DataRoots) -> dict[int, str]:
    """Which reference clip is decisive for each session.

    The map names a clip *file*; the decision works in clip ids, so the file
    name is translated here and an unknown file is an error rather than a
    silently ignored row.

    Raises:
        AssignError: if the map is unreadable or names an unconfigured clip.
    """
    name = config.speakers.session_psychiatrist_map
    if not name:
        return {}
    path = roots.work / name
    if not path.is_file():
        logger.warning(
            "%s: no session-to-psychiatrist map at %s; every session will be decided "
            "by its best match across all reference clips",
            STAGE,
            name,
        )
        return {}

    try:
        frame = read_csv(path)
    except (OSError, ValueError) as exc:
        msg = f"could not read {name}: {exc}"
        raise AssignError(msg) from exc

    columns = {str(c).strip().lower(): str(c) for c in frame.columns}
    if "session_id" not in columns:
        msg = f"{name} needs a session_id column; it has {list(frame.columns)}"
        raise AssignError(msg)
    accepted = ("reference_clip", "psychiatrist_id", "clip", "psychiatrist")
    clip_column = next((columns[name] for name in accepted if name in columns), None)
    if clip_column is None:
        msg = (
            f"{name} needs a column naming the reference clip (reference_clip or "
            f"psychiatrist_id); it has {list(frame.columns)}"
        )
        raise AssignError(msg)

    by_filename = {clip.path: clip.psychiatrist_id for clip in config.speakers.reference_clips}
    by_id = {clip.psychiatrist_id: clip.psychiatrist_id for clip in config.speakers.reference_clips}
    known = {**by_filename, **by_id}

    mapping: dict[int, str] = {}
    unknown: set[str] = set()
    for session_id, value in zip(frame[columns["session_id"]], frame[clip_column], strict=True):
        text = str(value).strip()
        clip_id = known.get(text) or known.get(Path(text).name)
        if clip_id is None:
            unknown.add(text)
            continue
        mapping[int(session_id)] = clip_id
    if unknown:
        msg = (
            f"{name} names reference clip(s) {sorted(unknown)} that are not configured "
            f"under speakers.reference_clips ({sorted(by_id)})"
        )
        raise AssignError(msg)
    logger.info("%s: %d session(s) have a mapped reference clip", STAGE, len(mapping))
    return mapping


# ---------------------------------------------------------------------------
# Per-speaker samples
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class SpeakerSample:
    """The audio used to represent one speaker."""

    speaker: str
    speech_s: float
    audio: np.ndarray
    sample_rate: int
    n_segments: int

    @property
    def embedded_s(self) -> float:
        """How much audio was actually taken."""
        return len(self.audio) / float(self.sample_rate)


def build_samples(
    samples: np.ndarray,
    sample_rate: int,
    by_speaker: Mapping[str, Sequence[Span]],
    config: AppConfig,
) -> list[SpeakerSample]:
    """Collect each speaker's speech into one stretch of audio.

    Longest segments first, up to the configured cap. Short segments are left
    out entirely: the diarizer partitions time and cannot represent overlap, so
    its briefest segments are the ones most likely to contain the other
    person's voice, and they are also the least informative about a voice.
    """
    settings = config.speakers.embedding
    built: list[SpeakerSample] = []
    limit = int(settings.max_seconds_per_speaker * sample_rate)

    for speaker, spans in sorted(by_speaker.items()):
        usable = sorted(
            (span for span in spans if span.duration >= settings.min_segment_s),
            key=lambda span: span.duration,
            reverse=True,
        )
        pieces: list[np.ndarray] = []
        taken = 0
        for span in usable:
            start = max(0, int(span.start * sample_rate))
            end = min(len(samples), int(span.end * sample_rate))
            if end <= start:
                continue
            piece = samples[start:end]
            pieces.append(piece[: max(0, limit - taken)] if taken + len(piece) > limit else piece)
            taken += len(pieces[-1])
            if taken >= limit:
                break
        audio = (
            np.concatenate(pieces)
            if pieces
            else np.zeros(0, dtype=samples.dtype if samples.size else np.float32)
        )
        built.append(
            SpeakerSample(
                speaker=speaker,
                speech_s=covered_duration(list(spans)),
                audio=audio,
                sample_rate=sample_rate,
                n_segments=len(usable),
            )
        )
    return built


def score_speakers(
    speaker_samples: Sequence[SpeakerSample],
    references: ReferenceReport,
    embedder: SpeakerEmbedder,
) -> list[SpeakerScore]:
    """Embed each speaker and compare against every reference clip.

    Every clip is scored, not only the mapped one: once a speaker is embedded,
    comparing against another clip is a dot product, and cross-clip agreement
    is then free evidence.
    """
    scored: list[SpeakerScore] = []
    for sample in speaker_samples:
        similarities: dict[str, float] = {}
        if sample.audio.size:
            try:
                embedding = embedder.embed(sample.audio, sample.sample_rate)
            except EmbeddingError as exc:
                logger.warning("speaker %s could not be embedded: %s", sample.speaker, exc)
            else:
                similarities = {
                    voice.clip_id: cosine_similarity(embedding, voice.embedding)
                    for voice in references.voices
                }
        scored.append(
            SpeakerScore(
                speaker=sample.speaker,
                speech_s=sample.speech_s,
                embedded_s=sample.embedded_s if similarities else 0.0,
                similarities=similarities,
            )
        )
    return scored


# ---------------------------------------------------------------------------
# Mouth movement: which voice belongs to which tile
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class MouthEvidence:
    """Which tile each speaker's voice appears to come from.

    Attributes:
        by_speaker: Speaker to the tile it correlates with, or None where the
            evidence was too weak to choose.
        correlations: Speaker to tile to correlation.
        frames_measured: Tile to the number of frames a face was found in.
        usable: Whether any speaker was matched to a tile.
        reason: Why the evidence was unusable, empty when it was.
    """

    by_speaker: Mapping[str, str | None]
    correlations: Mapping[str, Mapping[str, float]]
    frames_measured: Mapping[str, int]
    usable: bool
    reason: str = ""


def _speech_indicator(spans: Sequence[Span], timestamps: np.ndarray) -> np.ndarray:
    """1.0 where a speaker was talking at each sampled timestamp."""
    times = np.asarray(timestamps, dtype=np.float64)
    indicator = np.zeros(times.shape, dtype=np.float64)
    for span in spans:
        indicator[(times >= span.start) & (times < span.end)] = 1.0
    return indicator


def _correlate(mouth: np.ndarray, speech: np.ndarray) -> float | None:
    """Pearson correlation, or None where either series is constant.

    A tile in which no face was ever found gives a constant mouth series, and a
    speaker who talked throughout gives a constant speech series. Neither is a
    correlation of zero; both are an absence of evidence.
    """
    if mouth.size != speech.size or mouth.size < _MIN_SERIES:
        return None
    if float(np.var(mouth)) < _MIN_VARIANCE or float(np.var(speech)) < _MIN_VARIANCE:
        return None
    return float(np.corrcoef(mouth, speech)[0, 1])


def _tile_crops(config: AppConfig, tiles: Sequence[str]) -> dict[str, CropBox]:
    """The configured crop for each named tile."""
    return {name: config.video.tiles[name] for name in tiles if name in config.video.tiles}


def mouth_evidence(
    session: RawSession,
    *,
    config: AppConfig,
    roots: DataRoots,
    by_speaker: Mapping[str, Sequence[Span]],
    tools: FfmpegTools,
) -> MouthEvidence:
    """Correlate each speaker's speech against jaw movement in every tile.

    Correlating against *both* tiles rather than one is what makes this a real
    check: a correct assignment shows each speaker correlating with exactly one
    tile, and that pattern is much harder to produce by chance than a single
    positive correlation.

    Never raises: a missing video, an unavailable backend or a frame rate the
    sampling cannot divide all mean the same thing here, which is that this
    line of evidence is unavailable for this session.
    """
    settings = config.speakers.mouth_crosscheck
    tiles = list(settings.tiles or [name for name, _ in config.video.tiles_left_to_right()])
    crops = _tile_crops(config, tiles)
    if not crops:
        return MouthEvidence({}, {}, {}, usable=False, reason="no tiles are configured")

    try:
        info = parse_media_info(tools.probe(session.path))
        sampling = resolve_sampling(info.fps, settings.sample_fps)
    except (FfmpegError, SamplingError, ValueError) as exc:
        return MouthEvidence({}, {}, {}, usable=False, reason=str(exc))

    # A backend per session, not one shared: the landmarker is cached on the
    # instance and is not safe to call from several threads at once.
    backend = get_face_backend(config, roots)
    if not backend.available():
        return MouthEvidence({}, {}, {}, usable=False, reason=backend.unavailable_reason())

    per_tile: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    frames_measured: dict[str, int] = {}
    try:
        for tile, crop in crops.items():
            try:
                measures = backend.measure_session(
                    session, config=config, crop=crop, sampling=sampling
                )
            except FaceError as exc:
                return MouthEvidence({}, {}, {}, usable=False, reason=str(exc))
            detected = [m for m in measures if m.detected and m.jaw is not None]
            frames_measured[tile] = len(detected)
            if len(detected) < _MIN_SERIES:
                continue
            per_tile[tile] = (
                np.asarray([m.timestamp_s for m in detected], dtype=np.float64),
                np.asarray([float(m.jaw or 0.0) for m in detected], dtype=np.float64),
            )
    finally:
        backend.close()

    if not per_tile:
        return MouthEvidence(
            {},
            {},
            frames_measured,
            usable=False,
            reason="no face was found in any tile",
        )

    correlations: dict[str, dict[str, float]] = {}
    chosen: dict[str, str | None] = {}
    for speaker, spans in sorted(by_speaker.items()):
        per_speaker: dict[str, float] = {}
        for tile, (times, jaw) in per_tile.items():
            value = _correlate(jaw, _speech_indicator(spans, times))
            if value is not None:
                per_speaker[tile] = value
        correlations[speaker] = per_speaker
        chosen[speaker] = _choose_tile(
            per_speaker, settings.min_correlation, settings.min_separation
        )

    return MouthEvidence(
        by_speaker=chosen,
        correlations=correlations,
        frames_measured=frames_measured,
        usable=any(tile is not None for tile in chosen.values()),
        reason="" if any(chosen.values()) else "no tile correlated strongly enough",
    )


def _choose_tile(
    correlations: Mapping[str, float], min_correlation: float, min_separation: float
) -> str | None:
    """The tile a speaker belongs to, or None where the evidence is too weak.

    Two conditions, because either alone is easy to satisfy by accident: the
    best correlation has to be positive enough to mean something, and it has to
    be clearly ahead of the next tile. A speaker correlating equally with both
    faces has told us nothing about which one they are.
    """
    if not correlations:
        return None
    ordered = sorted(correlations.items(), key=lambda item: item[1], reverse=True)
    best_tile, best = ordered[0]
    if best < min_correlation:
        return None
    if len(ordered) > 1 and best - ordered[1][1] < min_separation:
        return None
    return best_tile


# ---------------------------------------------------------------------------
# The OCR bridge
# ---------------------------------------------------------------------------
def load_layout(roots: DataRoots) -> dict[int, str]:
    """Which side OCR put the psychiatrist on, per session.

    Read from what `vc verify-layout` wrote. Sessions it could not settle are
    left out, so an absent session means "no opinion" rather than a default.
    """
    path = roots.out_path(layout_stage.LAYOUT_FILENAME, create_parent=False)
    if not path.exists():
        return {}
    try:
        frame = read_csv(path)
    except (OSError, ValueError):  # pragma: no cover - defensive
        logger.warning("%s: could not read %s; OCR evidence unavailable", STAGE, path.name)
        return {}
    # `ocr_side` is what OCR itself concluded. `decided_side` is deliberately
    # not used: it falls back to the configured assumption, so reading it would
    # let an assumption masquerade as corroboration by the labels.
    if layout_stage.OCR_SIDE_COLUMN not in frame.columns:
        logger.warning(
            "%s: %s has no %s column; OCR evidence unavailable",
            STAGE,
            path.name,
            layout_stage.OCR_SIDE_COLUMN,
        )
        return {}
    sides: dict[int, str] = {}
    for session_id, side in zip(
        frame["session_id"], frame[layout_stage.OCR_SIDE_COLUMN], strict=True
    ):
        text = str(side).strip().lower()
        if text in {"left", "right"}:
            sides[int(session_id)] = text
    return sides


@dataclass(frozen=True, slots=True)
class CrossChecks:
    """What the two corroborating methods said.

    Attributes:
        mouth_agreement: agree, disagree, or unavailable.
        ocr_agreement: agree, disagree, or unavailable, where OCR's side is
            bridged to a voice by the mouth evidence.
        ocr_side: The side OCR settled on, or None.
        side_source: Whether the side came from `ocr` or from the configured
            assumption.
        independent_choice: Who the bridged evidence says the psychiatrist is.
        flags: Flags raised by the cross-checks.
    """

    mouth_agreement: str
    ocr_agreement: str
    ocr_side: str | None
    side_source: str
    independent_choice: str | None
    flags: tuple[str, ...]


def _ocr_cross_check(
    decision: Decision,
    mouth: MouthEvidence,
    *,
    config: AppConfig,
    ocr_side: str | None,
) -> CrossChecks:
    """Compare the embedding assignment against the visual evidence.

    Two distinct comparisons come out of this:

    * **Mouth agreement.** Does the tile each speaker was matched to imply the
      same psychiatrist as the embedding did, using the side we believe the
      psychiatrist sits on?
    * **OCR agreement.** The same question with the side taken specifically
      from OCR rather than from the configured assumption.

    The second is what corroborates the labels, and it is only computable where
    the mouth evidence exists, because OCR speaks about sides of the frame and
    diarization speaks about anonymous voices. With no bridge between them the
    two are not comparable, and this reports `unavailable` rather than treating
    an absent check as agreement.
    """
    flags: list[str] = []
    side_of_tile = {
        name: side
        for name in config.video.tiles
        if (side := config.video.side_of_tile(name)) is not None
    }

    if not mouth.usable:
        if config.speakers.mouth_crosscheck.enabled:
            flags.append(FLAG_MOUTH_UNUSABLE)
        return CrossChecks(
            mouth_agreement=UNAVAILABLE,
            ocr_agreement=UNAVAILABLE,
            ocr_side=ocr_side,
            side_source="ocr" if ocr_side else "none",
            independent_choice=None,
            flags=tuple(flags),
        )

    assumed = config.speakers.assumed_psychiatrist_side
    side = ocr_side or assumed
    source = "ocr" if ocr_side else "assumed"

    by_assumed = role_from_tiles(
        mouth.by_speaker, psychiatrist_side=side, side_of_tile=side_of_tile
    )
    mouth_agreement = agreement(decision.psychiatrist, by_assumed)
    if mouth_agreement == DISAGREE:
        flags.append(FLAG_MOUTH_DISAGREEMENT)

    if ocr_side is None:
        ocr_agreement = UNAVAILABLE
    else:
        by_ocr = role_from_tiles(
            mouth.by_speaker, psychiatrist_side=ocr_side, side_of_tile=side_of_tile
        )
        ocr_agreement = agreement(decision.psychiatrist, by_ocr)
        if ocr_agreement == DISAGREE:
            flags.append(FLAG_OCR_DISAGREEMENT)

    return CrossChecks(
        mouth_agreement=mouth_agreement,
        ocr_agreement=ocr_agreement,
        ocr_side=ocr_side,
        side_source=source,
        independent_choice=by_assumed,
        flags=tuple(flags),
    )


# ---------------------------------------------------------------------------
# One session
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class SessionAssignment:
    """Everything this stage established about one session."""

    session: RawSession
    scores: tuple[SpeakerScore, ...]
    decision: Decision
    checks: CrossChecks
    mouth: MouthEvidence

    @property
    def flags(self) -> tuple[str, ...]:
        """Every flag, decision and cross-check together."""
        return (*self.decision.flags, *self.checks.flags)

    def summary(self) -> str:
        """One line for the per-session log."""
        if not self.decision.is_complete:
            return f"{len(self.scores)} speaker(s), NOT assigned"
        margin = self.decision.margin or 0.0
        return (
            f"psychiatrist={self.decision.psychiatrist} margin={margin:+.3f} "
            f"ocr={self.checks.ocr_agreement} mouth={self.checks.mouth_agreement}"
        )


def _decisive_clips(
    references: ReferenceReport, mapped_clip: str | None
) -> tuple[tuple[str, ...], bool]:
    """Which clips decide this session, and whether it was mapped.

    An unmapped session is decided by its best match across every clip. That is
    the honest reading of "we do not know which psychiatrist this was": the
    session is compared with all of them and the closest wins.
    """
    every = tuple(voice.clip_id for voice in references.voices)
    if mapped_clip is None:
        return every, False
    return (mapped_clip,), True


def assign_one(
    session: RawSession,
    *,
    config: AppConfig,
    roots: DataRoots,
    references: ReferenceReport,
    embedder: SpeakerEmbedder,
    mapped_clip: str | None,
    ocr_side: str | None,
    tools: FfmpegTools,
) -> SessionAssignment:
    """Work out the roles for one session.

    Raises:
        AssignError: if the session's audio or diarization is missing, which is
            a missing prerequisite rather than a weak result.
    """
    audio_file = extract_audio_stage.audio_path(roots, session.session_id)
    if not audio_file.exists():
        msg = f"no extracted audio for session {session.session_id}; run `vc extract-audio`"
        raise AssignError(msg)
    segments_file = diarize_stage.segments_path(roots, session.session_id)
    if not segments_file.exists():
        msg = f"no diarization for session {session.session_id}; run `vc diarize`"
        raise AssignError(msg)

    try:
        samples, sample_rate = read_mono_wav(audio_file)
    except (OSError, ValueError) as exc:
        msg = f"could not read the audio for session {session.session_id}: {exc}"
        raise AssignError(msg) from exc

    frame = pd.read_parquet(segments_file)
    by_speaker = {
        speaker: merge(list(spans)) for speaker, spans in segments_by_speaker(frame).items()
    }

    speaker_samples = build_samples(samples, sample_rate, by_speaker, config)
    scores = score_speakers(speaker_samples, references, embedder)

    decisive, mapped = _decisive_clips(references, mapped_clip)
    decision = decide(
        scores,
        decisive_clips=decisive,
        all_clips=[voice.clip_id for voice in references.voices],
        min_margin=config.speakers.min_margin,
        min_speech_s=config.speakers.embedding.min_speech_s,
        mapped=mapped,
    )

    mouth = (
        mouth_evidence(session, config=config, roots=roots, by_speaker=by_speaker, tools=tools)
        if config.speakers.mouth_crosscheck.enabled
        else MouthEvidence({}, {}, {}, usable=False, reason="the mouth cross-check is disabled")
    )
    checks = _ocr_cross_check(decision, mouth, config=config, ocr_side=ocr_side)

    if decision.is_complete:
        write_role_mapping(
            roots.work,
            session.session_id,
            decision.by_speaker,
            evidence=_evidence(decision, checks, scores, embedder),
        )
    else:
        logger.warning(
            "session %s: roles not assigned (%s); downstream stages will refuse this "
            "session rather than guess",
            session.session_id,
            ", ".join(decision.flags) or "no reason recorded",
        )

    return SessionAssignment(
        session=session, scores=tuple(scores), decision=decision, checks=checks, mouth=mouth
    )


def _evidence(
    decision: Decision,
    checks: CrossChecks,
    scores: Sequence[SpeakerScore],
    embedder: SpeakerEmbedder,
) -> dict[str, Any]:
    """What is recorded beside the role mapping.

    Written so that a role assignment can be audited later without rerunning
    anything: the numbers that produced it, what the cross-checks said, and
    which model version was used.
    """
    return {
        "method": "speaker_embedding",
        "embedder": embedder.version(),
        "decisive_clips": list(decision.decisive_clips),
        "margin": decision.margin,
        "clip_choices": dict(decision.clip_choices),
        "clips_agree": decision.clips_agree,
        "similarities": {
            score.speaker: {clip: round(value, 6) for clip, value in score.similarities.items()}
            for score in scores
        },
        "embedded_seconds": {score.speaker: round(score.embedded_s, 3) for score in scores},
        "mouth_agreement": checks.mouth_agreement,
        "ocr_agreement": checks.ocr_agreement,
        "ocr_side": checks.ocr_side,
        "side_source": checks.side_source,
        "flags": list(decision.flags) + list(checks.flags),
    }


# ---------------------------------------------------------------------------
# The stage
# ---------------------------------------------------------------------------
SCORES_FILENAME: Final = "speaker_scores.csv"


@dataclass(frozen=True, slots=True)
class AssignResult:
    """What the stage produced."""

    report: StageReport
    frame: pd.DataFrame
    scores: pd.DataFrame
    path: Path
    scores_path: Path
    references: ReferenceReport


def speakers_path(roots: DataRoots) -> Path:
    """Where the per-session QC table is written."""
    return roots.out_path(SPEAKERS_FILENAME)


def scores_path(roots: DataRoots) -> Path:
    """Where the per-speaker evidence table is written."""
    return roots.out_path(SCORES_FILENAME)


def _tiles_note(mouth: MouthEvidence) -> str:
    """Which tile each speaker was matched to, as one field."""
    return ";".join(
        f"{speaker}={tile or 'none'}" for speaker, tile in sorted(mouth.by_speaker.items())
    )


def _best_correlation(mouth: MouthEvidence) -> float | None:
    """The strongest speaker-to-tile correlation seen."""
    values = [value for per in mouth.correlations.values() for value in per.values()]
    return max(values) if values else None


def _session_row(assignment: SessionAssignment, mapped_clip: str | None) -> dict[str, object]:
    """One row of the per-session QC table."""
    decision = assignment.decision
    by_speaker = {score.speaker: score for score in assignment.scores}
    psychiatrist = by_speaker.get(decision.psychiatrist or "")
    participant = by_speaker.get(decision.participant or "")
    decisive = list(decision.decisive_clips)

    return {
        "session_id": assignment.session.session_id,
        "wave": assignment.session.wave,
        "n_speakers": len(assignment.scores),
        "assigned": decision.is_complete,
        "psychiatrist_speaker": decision.psychiatrist or "",
        "participant_speaker": decision.participant or "",
        "mapped_clip": mapped_clip or "",
        "decisive_clips": ";".join(decisive),
        "psychiatrist_similarity": (
            psychiatrist.score_over(decisive) if psychiatrist is not None else None
        ),
        "participant_similarity": (
            participant.score_over(decisive) if participant is not None else None
        ),
        "best_clip": (psychiatrist.best_clip_over(decisive) or "") if psychiatrist else "",
        "margin": decision.margin,
        "clip_choices": ";".join(
            f"{clip}={who}" for clip, who in sorted(decision.clip_choices.items())
        ),
        "clips_agree": decision.clips_agree,
        "psychiatrist_speech_s": psychiatrist.speech_s if psychiatrist is not None else None,
        "participant_speech_s": participant.speech_s if participant is not None else None,
        "ocr_side": assignment.checks.ocr_side or "",
        "side_source": assignment.checks.side_source,
        "ocr_agreement": assignment.checks.ocr_agreement,
        "mouth_agreement": assignment.checks.mouth_agreement,
        "mouth_tiles": _tiles_note(assignment.mouth),
        "mouth_best_correlation": _best_correlation(assignment.mouth),
        "mouth_reason": assignment.mouth.reason,
        "qc__flags": ";".join(assignment.flags),
    }


def _score_rows(assignment: SessionAssignment, clips: Sequence[str]) -> list[dict[str, object]]:
    """One row per speaker: the evidence behind the assignment."""
    rows: list[dict[str, object]] = []
    for score in assignment.scores:
        row: dict[str, object] = {
            "session_id": assignment.session.session_id,
            "speaker": score.speaker,
            "role": assignment.decision.by_speaker.get(score.speaker, "unknown"),
            "speech_s": round(score.speech_s, 3),
            "embedded_s": round(score.embedded_s, 3),
            "tile": assignment.mouth.by_speaker.get(score.speaker) or "",
        }
        for clip in clips:
            row[f"sim__{clip}"] = score.similarities.get(clip)
        for tile, value in sorted(assignment.mouth.correlations.get(score.speaker, {}).items()):
            row[f"corr__{tile}"] = value
        rows.append(row)
    return rows


#: Column order of the per-session table, named once so the merge and the
#: builder cannot disagree about the shape.
SPEAKER_COLUMN_ORDER: Final = (
    "session_id",
    "wave",
    "n_speakers",
    "assigned",
    "psychiatrist_speaker",
    "participant_speaker",
    "mapped_clip",
    "decisive_clips",
    "psychiatrist_similarity",
    "participant_similarity",
    "best_clip",
    "margin",
    "clip_choices",
    "clips_agree",
    "psychiatrist_speech_s",
    "participant_speech_s",
    "ocr_side",
    "side_source",
    "ocr_agreement",
    "mouth_agreement",
    "mouth_tiles",
    "mouth_best_correlation",
    "mouth_reason",
    "qc__flags",
)


def _build_frame(rows: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    """Assemble a table with stable columns even when empty."""
    columns = list(SPEAKER_COLUMN_ORDER)
    frame = pd.DataFrame(list(rows), columns=columns)
    if not frame.empty:
        frame["session_id"] = frame["session_id"].astype("int64")
        frame["n_speakers"] = frame["n_speakers"].astype("int64")
    return frame.sort_values("session_id", ignore_index=True)


def _merge_scores(target: Path, fresh: pd.DataFrame, computed: set[int]) -> pd.DataFrame:
    """The per-speaker evidence table, keeping rows for sessions not rerun.

    Several rows per session here, so this cannot go through `combine`, which
    is keyed by session. The rule is the same: drop the sessions recomputed,
    keep the rest.
    """
    if not target.exists():
        return fresh
    try:
        existing = read_csv(target)
    except (OSError, ValueError):
        logger.warning("%s: could not read %s; writing this run's rows only", STAGE, target.name)
        return fresh
    if ID_COLUMN not in existing.columns:
        logger.warning(
            "%s: %s has no %s column; writing this run's rows only",
            STAGE,
            target.name,
            ID_COLUMN,
        )
        return fresh
    numeric = pd.to_numeric(existing[ID_COLUMN], errors="coerce")
    kept = existing.loc[numeric.notna() & ~numeric.isin(list(computed))]
    if kept.empty:
        return fresh
    logger.info(
        "%s: keeping %d existing evidence row(s) for session(s) not in this run",
        STAGE,
        len(kept),
    )
    merged = pd.concat([kept, fresh], ignore_index=True)
    return merged.sort_values([ID_COLUMN, "speaker"], ignore_index=True)


def run(
    config: AppConfig,
    roots: DataRoots,
    *,
    session_ids: Sequence[int] | None = None,
    workers: int | None = None,
    force: bool = False,
    tools: FfmpegTools | None = None,
) -> AssignResult:
    """Assign roles across the selected sessions.

    Args:
        config: Resolved configuration.
        roots: Data roots.
        session_ids: Sessions to process, or None for all discovered.
        workers: Parallel workers. Threads, because the work is dominated by
            model inference that releases the interpreter lock.
        force: Unused; the assignment is always recomputed, being cheap next to
            the evidence it rests on and the only thing every later stage
            depends on.
        tools: ffmpeg binaries, discovered when not supplied.

    Returns:
        The stage report and both evidence tables.

    Raises:
        AssignError: if the reference clips or the map cannot be used.
    """
    binaries = tools or FfmpegTools.discover()
    embedder = get_embedder(config, roots.work)
    if not embedder.available():
        msg = (
            f"speaker embeddings are unavailable, so roles cannot be assigned: "
            f"{embedder.unavailable_reason()}"
        )
        raise AssignError(msg)

    references = load_references(config, roots, embedder)
    for line in references.report_lines():
        logger.info("%s: %s", STAGE, line)

    mapping = load_psychiatrist_map(config, roots)
    sides = load_layout(roots)

    discovery = discover_sessions(roots.data, config.dataset)
    selected, missing = select_sessions(discovery.sessions, session_ids)
    notes = [f"requested session(s) not found: {sorted(missing)}"] if missing else []
    if not sides:
        notes.append("no settled label OCR, so nothing corroborates the labels")

    assignments: dict[int, SessionAssignment] = {}

    def assign(session: RawSession) -> str:
        assignment = assign_one(
            session,
            config=config,
            roots=roots,
            references=references,
            embedder=embedder,
            mapped_clip=mapping.get(session.session_id),
            ocr_side=sides.get(session.session_id),
            tools=binaries,
        )
        assignments[session.session_id] = assignment
        return assignment.summary()

    report = run_sessions(STAGE, selected, assign, workers=workers, backend="threads", notes=notes)

    clips = [voice.clip_id for voice in references.voices]
    rows = [
        _session_row(assignment, mapping.get(session_id))
        for session_id, assignment in sorted(assignments.items())
    ]
    score_rows = [
        row
        for _, assignment in sorted(assignments.items())
        for row in _score_rows(assignment, clips)
    ]

    # Rows for sessions outside this run are kept rather than deleted.
    target = speakers_path(roots)
    carried = carry_forward(
        target,
        computed=set(assignments),
        columns=list(SPEAKER_COLUMN_ORDER),
        stage=STAGE,
        force=force,
    )
    fresh = {int(cast("int", row["session_id"])): row for row in rows}
    frame = _build_frame(combine(fresh, carried))
    evidence_target = scores_path(roots)
    scores = _merge_scores(evidence_target, pd.DataFrame(score_rows), set(assignments))
    write_csv(target, frame)
    write_csv(evidence_target, scores)
    logger.info("%s: wrote %s and %s", STAGE, target.name, evidence_target.name)

    return AssignResult(
        report=report.with_notes(carried.notes(STAGE)),
        frame=frame,
        scores=scores,
        path=target,
        scores_path=evidence_target,
        references=references,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _ids(frame: pd.DataFrame, mask: pd.Series) -> list[int]:
    """Session ids matching `mask`, for a report."""
    return [int(v) for v in frame.loc[mask, "session_id"]]


def _agreement_lines(frame: pd.DataFrame, column: str, label: str) -> list[str]:
    """Agree / disagree / unavailable, with the session ids.

    Disagreement is listed in full however long the list: it is the one outcome
    that needs acting on, and a truncated list of problems is a list of
    problems someone will assume they have seen the end of.
    """
    lines = [f"{label}:"]
    for outcome in (AGREE, DISAGREE, UNAVAILABLE):
        matching = _ids(frame, frame[column] == outcome)
        if not matching:
            continue
        shown = (
            f" {matching}"
            if outcome == DISAGREE or len(matching) <= _MAX_LISTED
            else f" {matching[:_MAX_LISTED]} ..."
        )
        lines.append(f"  {outcome:<12} {len(matching):>3} session(s){shown}")
    return lines


def _margin_lines(frame: pd.DataFrame, config: AppConfig) -> list[str]:
    """How decisive the assignments were."""
    margins = pd.to_numeric(frame["margin"], errors="coerce").dropna()
    if margins.empty:
        return ["margins: none computed"]
    low = _ids(frame, pd.to_numeric(frame["margin"], errors="coerce") < config.speakers.min_margin)
    lines = [
        f"margins: min {margins.min():+.3f}  median {margins.median():+.3f}  "
        f"max {margins.max():+.3f}",
    ]
    if low:
        lines.append(f"  below the {config.speakers.min_margin:+.2f} threshold, flagged: {low}")
    else:
        lines.append(f"  every session is clear of the {config.speakers.min_margin:+.2f} threshold")
    return lines


def _unmapped_lines(frame: pd.DataFrame) -> list[str]:
    """What the sessions with no mapped reference clip matched.

    Reported separately because these are the sessions where this stage may
    settle something that OCR could not.
    """
    unmapped = frame.loc[frame["mapped_clip"].astype(str).str.len() == 0]
    if unmapped.empty:
        return []
    lines = [f"sessions with no mapped reference clip ({len(unmapped)}):"]
    for _, row in unmapped.iterrows():
        margin = row["margin"]
        margin_text = f"{float(margin):+.3f}" if pd.notna(margin) else "n/a"
        lines.append(
            f"  session {int(row['session_id']):>3}: best match {row['best_clip'] or 'none'}, "
            f"margin {margin_text}, psychiatrist {row['psychiatrist_speaker'] or 'unassigned'}, "
            f"choices {row['clip_choices'] or 'none'}"
        )
    return lines


def summarise(result: AssignResult, config: AppConfig) -> list[str]:
    """What the CLI prints.

    Counts, session ids and similarity numbers only. No recognised label text
    and no audio ever reaches this.
    """
    frame = result.frame
    lines: list[str] = []
    lines.extend(result.references.report_lines())
    lines.append("")

    assigned = _ids(frame, frame["assigned"].astype(bool))
    unassigned = _ids(frame, ~frame["assigned"].astype(bool))
    lines.append(f"assigned: {len(assigned)} session(s)")
    if unassigned:
        lines.append(f"NOT assigned: {len(unassigned)} session(s): {unassigned}")

    # Which side the embedding put the psychiatrist on is not knowable without
    # the mouth link, so the headline check is between the reference clips.
    disagreeing = _ids(frame, frame["clips_agree"] == False)  # noqa: E712 - a nullable column
    lines.append(
        f"reference clips agree on the psychiatrist in "
        f"{len(frame) - len(disagreeing)} of {len(frame)} session(s)"
    )
    if disagreeing:
        lines.append(f"  clips disagree, flagged: {disagreeing}")

    lines.append("")
    lines.extend(_margin_lines(frame, config))
    lines.append("")
    lines.extend(
        _agreement_lines(
            frame, "ocr_agreement", "embedding vs label OCR (bridged by mouth movement)"
        )
    )
    lines.append("")
    lines.extend(_agreement_lines(frame, "mouth_agreement", "embedding vs mouth movement"))

    unavailable = _ids(frame, frame["mouth_agreement"] == UNAVAILABLE)
    if unavailable:
        lines.append(
            "  OCR can only corroborate an assignment through the mouth link: it knows "
            "which side of the frame the psychiatrist is on, and diarization knows only "
            "anonymous voices. Where the mouth evidence is missing the two are not "
            "comparable, which is what unavailable means here."
        )

    unmapped = _unmapped_lines(frame)
    if unmapped:
        lines.append("")
        lines.extend(unmapped)

    flagged = frame.loc[frame["qc__flags"].astype(str).str.len() > 0]
    if not flagged.empty:
        lines.append("")
        lines.append(f"flags across {len(flagged)} session(s):")
        counted: dict[str, list[int]] = {}
        for _, row in flagged.iterrows():
            for flag in str(row["qc__flags"]).split(";"):
                if flag:
                    counted.setdefault(flag, []).append(int(row["session_id"]))
        for flag, ids in sorted(counted.items()):
            lines.append(f"  {flag}: {len(ids)} session(s) {ids}")

    lines.append("")
    lines.append(f"per-speaker evidence: {result.scores_path}")
    return lines
