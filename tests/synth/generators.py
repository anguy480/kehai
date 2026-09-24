"""Synthetic data generators for the test suite.

No test touches real recordings. Everything is built here from a single shared
speaking schedule, so the audio, the mouth movement in the video and the
diarization transcript all agree with each other. That is what makes the
end-to-end test meaningful rather than merely green: a stage that mismatches
speakers, misplaces a turn boundary or correlates the wrong video tile produces
a wrong answer against a known ground truth.

Conventions used throughout:

* `SPK_A` is the psychiatrist and speaks first, from the LEFT video tile.
* `SPK_B` is the participant and speaks from the RIGHT tile.
* Tones differ in pitch per speaker, so pitch features have something to find.
"""

from __future__ import annotations

import subprocess
import wave
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import pandas as pd

PSYCHIATRIST = "SPEAKER_00"
PARTICIPANT = "SPEAKER_01"

# Distinct pitches, roughly an octave apart, so F0 is measurable per speaker.
SPEAKER_TONE_HZ: dict[str, float] = {PSYCHIATRIST: 110.0, PARTICIPANT: 220.0}
SPEAKER_TILE: dict[str, str] = {PSYCHIATRIST: "left", PARTICIPANT: "right"}

# Zoom-style name labels drawn in the bottom-left of each tile, so the label-OCR
# path can be exercised. Obviously fake, Latin-only: the Hershey fonts OpenCV
# ships cannot render Japanese, and these only have to be readable and
# consistent, not realistic. The psychiatrist's label is the same in every
# session and each participant's is unique, which is the structure that lets
# `vc verify-layout` identify the psychiatrist without being told any name.
PSYCHIATRIST_LABEL = "DR SATO"


def participant_label(session_id: int) -> str:
    """The unique label drawn in the participant's tile."""
    return f"GUEST {session_id:03d}"


@dataclass(frozen=True, slots=True)
class Utterance:
    """One stretch of speech by one speaker."""

    speaker: str
    start: float
    end: float
    text: str = "..."

    @property
    def duration(self) -> float:
        """Length in seconds."""
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class SyntheticSession:
    """A complete synthetic session: what is said, by whom, and when."""

    session_id: int
    utterances: tuple[Utterance, ...]
    duration: float
    width: int = 320
    height: int = 180
    fps: float = 10.0
    sample_rate: int = 16000
    speakers: tuple[str, str] = (PSYCHIATRIST, PARTICIPANT)
    draw_labels: bool = True
    swap_tiles: bool = False

    def tile_of(self, speaker: str) -> str:
        """Which tile a speaker occupies, honouring `swap_tiles`."""
        tile = SPEAKER_TILE[speaker]
        if not self.swap_tiles:
            return tile
        return "right" if tile == "left" else "left"

    def label_of(self, speaker: str) -> str:
        """The name label drawn in a speaker's tile."""
        if speaker == PSYCHIATRIST:
            return PSYCHIATRIST_LABEL
        return participant_label(self.session_id)

    def spans(self, speaker: str) -> tuple[tuple[float, float], ...]:
        """Speaking spans for one speaker."""
        return tuple((u.start, u.end) for u in self.utterances if u.speaker == speaker)

    def speaking_at(self, speaker: str, t: float) -> bool:
        """Whether `speaker` is speaking at time `t`."""
        return any(start <= t < end for start, end in self.spans(speaker))

    def total_speech(self, speaker: str) -> float:
        """Total seconds of speech by one speaker."""
        return sum(u.duration for u in self.utterances if u.speaker == speaker)


def alternating_session(
    session_id: int,
    *,
    n_turns: int = 6,
    turn_s: float = 1.5,
    gap_s: float = 0.5,
    lead_in_s: float = 0.5,
    duration: float | None = None,
    **geometry: float | int,
) -> SyntheticSession:
    """Build a session of clean alternating turns separated by fixed gaps.

    Every gap is `gap_s`, so response latency has an exact expected value and
    the turn/latency mathematics can be checked against it.

    Args:
        session_id: Session identifier.
        n_turns: Total number of turns, alternating psychiatrist first.
        turn_s: Length of each turn.
        gap_s: Silence between consecutive turns.
        lead_in_s: Silence before the first turn.
        duration: Total length. Defaults to just past the final turn.
        **geometry: Frame geometry overrides for `SyntheticSession`, e.g.
            `width`, `height`, `fps`, `sample_rate`.

    Returns:
        The synthetic session.
    """
    utterances: list[Utterance] = []
    cursor = lead_in_s
    for turn in range(n_turns):
        speaker = PSYCHIATRIST if turn % 2 == 0 else PARTICIPANT
        utterances.append(
            Utterance(speaker=speaker, start=cursor, end=cursor + turn_s, text=f"turn {turn}")
        )
        cursor += turn_s + gap_s
    return SyntheticSession(
        session_id=session_id,
        utterances=tuple(utterances),
        duration=cursor if duration is None else duration,
        **geometry,
    )


def overlapping_session(
    session_id: int, *, duration: float = 10.0, **geometry: float | int
) -> SyntheticSession:
    """A session containing a deliberate overlap, for overlap-exclusion tests."""
    utterances = (
        Utterance(PSYCHIATRIST, 0.5, 3.0, "question"),
        # Starts before the psychiatrist has finished: 0.5 s of overlap.
        Utterance(PARTICIPANT, 2.5, 6.0, "answer"),
        Utterance(PSYCHIATRIST, 7.0, 8.5, "follow up"),
    )
    return SyntheticSession(
        session_id=session_id,
        utterances=utterances,
        duration=duration,
        **geometry,
    )


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------
def _tone(frequency: float, n_samples: int, sample_rate: int, amplitude: float) -> np.ndarray:
    """A sine tone with a short fade in and out, to avoid clicks."""
    t = np.arange(n_samples, dtype=np.float64) / sample_rate
    wave_data = amplitude * np.sin(2.0 * np.pi * frequency * t)
    fade = min(int(0.01 * sample_rate), max(n_samples // 4, 1))
    if fade > 0:
        ramp = np.linspace(0.0, 1.0, fade)
        wave_data[:fade] *= ramp
        wave_data[-fade:] *= ramp[::-1]
    return wave_data


def session_waveform(
    session: SyntheticSession,
    *,
    speakers: Sequence[str] | None = None,
    amplitude: float = 0.35,
    noise: float = 0.001,
) -> np.ndarray:
    """Render a session to a float waveform in [-1, 1].

    Args:
        session: The session to render.
        speakers: Which speakers to include. Defaults to all of them, i.e. a
            single mixed stream.
        amplitude: Tone amplitude.
        noise: Amplitude of background noise, so the signal is not pure
            silence, which some voice-activity detectors treat specially.

    Returns:
        A 1-D float array of `duration * sample_rate` samples.
    """
    wanted = set(speakers) if speakers is not None else {u.speaker for u in session.utterances}
    n = round(session.duration * session.sample_rate)
    rng = np.random.default_rng(session.session_id)
    signal = rng.normal(0.0, noise, size=n) if noise > 0 else np.zeros(n)

    for utterance in session.utterances:
        if utterance.speaker not in wanted:
            continue
        start = round(utterance.start * session.sample_rate)
        end = min(round(utterance.end * session.sample_rate), n)
        if end <= start:
            continue
        frequency = SPEAKER_TONE_HZ.get(utterance.speaker, 150.0)
        signal[start:end] += _tone(frequency, end - start, session.sample_rate, amplitude)

    return np.clip(signal, -1.0, 1.0)


def voiced_signal(
    seconds: float,
    *,
    sample_rate: int = 16000,
    f0: float = 120.0,
    syllable_hz: float = 4.0,
    seed: int = 0,
) -> np.ndarray:
    """Synthesise something speech-like enough for a voice activity detector.

    Silero is trained on speech and correctly rejects the pure tones the rest
    of these generators use, so a tone-based recording cannot exercise it. This
    builds a harmonic source with jitter, three formant-ish resonances and
    syllable-rate amplitude modulation.

    It is only marginally speech-like: Silero finds the onset exactly but ends
    the span early. Tests therefore assert onset placement, which is enough to
    prove the wiring, and use a stub detector for everything about the stage's
    own logic.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(sample_rate * seconds)) / sample_rate
    if t.size == 0:
        return np.zeros(0, dtype=np.float64)

    wander = rng.normal(0.0, 1.0, t.size).cumsum() / max(t.size**0.5, 1.0)
    jitter = 1.0 + 0.02 * np.sin(2.0 * np.pi * 3.1 * t) + 0.01 * wander
    signal = sum(
        (1.0 / harmonic) * np.sin(2.0 * np.pi * f0 * harmonic * jitter * t)
        for harmonic in range(1, 16)
    )
    for centre, bandwidth, amplitude in (
        (700.0, 120.0, 1.0),
        (1200.0, 160.0, 0.6),
        (2600.0, 250.0, 0.3),
    ):
        noise = rng.normal(0.0, 1.0, t.size)
        carrier = np.sin(2.0 * np.pi * centre * t + 6.0 * np.cumsum(noise) / sample_rate)
        signal = signal + amplitude * carrier * (
            0.3 + 0.7 * np.abs(np.sin(2.0 * np.pi * bandwidth / 100.0 * t))
        )
    envelope = 0.35 + 0.65 * np.clip(np.sin(2.0 * np.pi * syllable_hz * t), 0.0, None) ** 0.6
    peak = float(np.max(np.abs(signal))) or 1.0
    return signal / peak * envelope * 0.5


def voiced_session_waveform(session: SyntheticSession, *, amplitude: float = 0.5) -> np.ndarray:
    """Render a session using the speech-like signal instead of tones.

    Each speaker gets a different fundamental, so they remain distinguishable.
    """
    n = round(session.duration * session.sample_rate)
    out = np.zeros(n, dtype=np.float64)
    pitches = {PSYCHIATRIST: 110.0, PARTICIPANT: 190.0}
    for index, utterance in enumerate(session.utterances):
        start = round(utterance.start * session.sample_rate)
        end = min(round(utterance.end * session.sample_rate), n)
        if end <= start:
            continue
        chunk = voiced_signal(
            (end - start) / session.sample_rate,
            sample_rate=session.sample_rate,
            f0=pitches.get(utterance.speaker, 150.0),
            seed=index,
        )
        out[start : start + chunk.size] += chunk[: end - start] * amplitude
    return np.clip(out, -1.0, 1.0)


def write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> Path:
    """Write a 16-bit PCM WAV using only the standard library.

    Accepts mono `(n,)` or stereo `(n, 2)` samples, so tests can build the
    stereo layouts the left/right probe has to tell apart.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    channels = 1 if samples.ndim == 1 else samples.shape[1]
    pcm = np.clip(samples * 32767.0, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        # Stereo frames are stored interleaved, which is what C-order gives.
        handle.writeframes(pcm.tobytes())
    return path


# How the two speakers are placed across the stereo field.
#   "mono"       one signal duplicated into both channels: no separation
#   "per_speaker" psychiatrist hard left, participant hard right: total
#                 separation, the best case a panned recording could approach
#   "panned"     each speaker mostly on one side: partial separation, which is
#                what Zoom panning would actually look like
StereoLayout = Literal["mono", "per_speaker", "panned"]

# Fraction of a speaker's signal that leaks into the other channel when panned.
PANNED_BLEED = 0.3


def session_stereo(
    session: SyntheticSession,
    layout: StereoLayout = "mono",
    **kwargs: object,
) -> np.ndarray:
    """Render a session to a stereo waveform shaped `(n, 2)`.

    Args:
        session: The session to render.
        layout: How to place the speakers across the channels.
        **kwargs: Passed through to `session_waveform`.

    Returns:
        A two-channel float array.
    """
    if layout == "mono":
        mixed = session_waveform(session, **kwargs)  # type: ignore[arg-type]
        return np.stack([mixed, mixed], axis=1)

    psychiatrist = session_waveform(session, speakers=[PSYCHIATRIST], **kwargs)  # type: ignore[arg-type]
    participant = session_waveform(session, speakers=[PARTICIPANT], **kwargs)  # type: ignore[arg-type]

    if layout == "per_speaker":
        return np.stack([psychiatrist, participant], axis=1)

    bleed = PANNED_BLEED
    left = (1.0 - bleed) * psychiatrist + bleed * participant
    right = bleed * psychiatrist + (1.0 - bleed) * participant
    return np.stack([left, right], axis=1)


def write_session_wav(
    path: Path, session: SyntheticSession, *, speakers: Sequence[str] | None = None
) -> Path:
    """Render and write a session's audio in one call."""
    samples = session_waveform(session, speakers=speakers)
    return write_wav(path, samples, session.sample_rate)


# ---------------------------------------------------------------------------
# Video
# ---------------------------------------------------------------------------
def write_letterboxed_face_video(
    path: Path,
    session: SyntheticSession,
    *,
    bar: int = 180,
    detectable: bool = True,
) -> Path:
    """Write a gallery-view video with letterbox bars and face-like tiles.

    Matches the real recordings' geometry: 1280x720 with bars top and bottom,
    so the content is two 16:9 tiles side by side. Each tile holds a crude
    frontal face, drawn large enough that a landmarker has a chance of finding
    it, with a mouth that opens while that speaker talks.

    `detectable=False` draws flat grey tiles instead, for testing the
    dropped-frame path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1280, 720
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), session.fps, (width, height)
    )
    if not writer.isOpened():  # pragma: no cover - depends on local codecs
        msg = f"OpenCV could not open a writer for {path}"
        raise RuntimeError(msg)

    content_top, content_height = bar, height - 2 * bar
    half = width // 2
    n_frames = round(session.duration * session.fps)

    try:
        for index in range(n_frames):
            t = index / session.fps
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[content_top : content_top + content_height, :] = 205

            for speaker in session.speakers:
                tile = session.tile_of(speaker)
                cx = (half // 2) if tile == "left" else (half + half // 2)
                cy = content_top + content_height // 2
                if not detectable:
                    continue

                # A frontal face: head, brows, eyes, nose, mouth.
                cv2.ellipse(frame, (cx, cy), (95, 125), 0, 0, 360, (212, 184, 164), -1)
                for sign in (-1, 1):
                    eye_x = cx + sign * 38
                    cv2.ellipse(frame, (eye_x, cy - 30), (17, 10), 0, 0, 360, (250, 250, 250), -1)
                    cv2.circle(frame, (eye_x, cy - 30), 7, (35, 35, 45), -1)
                    cv2.ellipse(frame, (eye_x, cy - 52), (20, 7), 0, 180, 360, (70, 50, 40), 3)
                cv2.line(frame, (cx, cy - 20), (cx, cy + 18), (180, 150, 135), 3)
                speaking = session.speaking_at(speaker, t)
                openness = 0.5 + 0.5 * float(np.sin(2.0 * np.pi * 3.0 * t)) if speaking else 0.0
                mouth_h = max(4, round(6 + openness * 22))
                cv2.ellipse(frame, (cx, cy + 52), (30, mouth_h), 0, 0, 360, (90, 55, 55), -1)
            writer.write(frame)
    finally:
        writer.release()
    return path


def write_session_video(path: Path, session: SyntheticSession) -> Path:
    """Write a synthetic two-tile "gallery view" video.

    Each half of the frame holds a face: a filled circle with a mouth whose
    height grows while that speaker is talking. The mouth is what makes the
    speaker/tile cross-check testable, because tile brightness in the mouth
    region correlates with that speaker's speech and with nothing else.

    Args:
        path: Output video path.
        session: Session defining the speaking schedule and frame geometry.

    Returns:
        `path`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        session.fps,
        (session.width, session.height),
    )
    if not writer.isOpened():  # pragma: no cover - depends on local codecs
        msg = f"OpenCV could not open a writer for {path}"
        raise RuntimeError(msg)

    half = session.width // 2
    centres = {"left": half // 2, "right": half + half // 2}
    n_frames = round(session.duration * session.fps)

    try:
        for index in range(n_frames):
            t = index / session.fps
            frame = np.full((session.height, session.width, 3), 30, dtype=np.uint8)
            # A visible seam between tiles, so a wrong crop is obvious by eye.
            frame[:, half - 1 : half + 1] = 90

            for speaker in session.speakers:
                tile = session.tile_of(speaker)
                cx, cy = centres[tile], session.height // 2
                cv2.circle(frame, (cx, cy), session.height // 4, (200, 180, 160), -1)
                speaking = session.speaking_at(speaker, t)
                # Oscillating mouth while speaking, nearly closed otherwise.
                openness = 0.5 + 0.5 * float(np.sin(2.0 * np.pi * 3.0 * t)) if speaking else 0.0
                mouth_h = max(1, round(2 + openness * session.height * 0.12))
                mouth_w = session.height // 6
                cv2.rectangle(
                    frame,
                    (cx - mouth_w // 2, cy + session.height // 10),
                    (cx + mouth_w // 2, cy + session.height // 10 + mouth_h),
                    (40, 20, 20),
                    -1,
                )

                if session.draw_labels:
                    # Bottom-left of the tile, as Zoom draws it.
                    tile_left = 0 if tile == "left" else half
                    cv2.putText(
                        frame,
                        session.label_of(speaker),
                        (tile_left + 6, session.height - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (250, 250, 250),
                        1,
                        cv2.LINE_AA,
                    )
            writer.write(frame)
    finally:
        writer.release()
    return path


# ---------------------------------------------------------------------------
# Muxing, to produce something ffprobe sees as a real recording
# ---------------------------------------------------------------------------
def mux(
    out_path: Path,
    video: Path,
    audio_streams: Sequence[Path],
    *,
    ffmpeg: str = "ffmpeg",
) -> Path:
    """Mux a video file and one or more WAV files into an mp4.

    Several audio streams are supported so the inventory stage's handling of the
    unresolved "one mixed stream or two" question can be tested both ways.

    Args:
        out_path: Destination mp4.
        video: Source video file.
        audio_streams: One or more WAV files, each becoming its own stream.
        ffmpeg: ffmpeg executable.

    Returns:
        `out_path`.

    Raises:
        RuntimeError: if ffmpeg fails.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    command = [ffmpeg, "-y", "-loglevel", "error", "-i", str(video)]
    for stream in audio_streams:
        command += ["-i", str(stream)]
    command += ["-map", "0:v:0"]
    for index in range(len(audio_streams)):
        command += ["-map", f"{index + 1}:a:0"]
    # +faststart moves the moov atom to the front, as a Zoom recording has it.
    # It also makes truncation realistic: cutting the tail leaves the metadata
    # intact and the media data short, which is how a partially copied file
    # behaves, rather than making the file unreadable.
    command += [
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        "-movflags",
        "+faststart",
        str(out_path),
    ]

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        msg = f"ffmpeg failed muxing {out_path.name}: {result.stderr.strip()[:400]}"
        raise RuntimeError(msg)
    return out_path


def write_session_mp4(
    out_path: Path,
    session: SyntheticSession,
    *,
    tmp_dir: Path,
    per_speaker_audio: bool = False,
    stereo_layout: StereoLayout | None = None,
    ffmpeg: str = "ffmpeg",
) -> Path:
    """Build a complete synthetic recording: video plus audio, muxed to mp4.

    Args:
        out_path: Destination mp4, normally `<session_id>.mp4`.
        session: The session to render.
        tmp_dir: Scratch directory for the intermediate video and WAV files.
        per_speaker_audio: Write one audio stream per speaker instead of a
            single mixed stream.
        stereo_layout: Render the single stream in stereo with this layout,
            matching the real recordings, which carry one stereo stream. None
            writes mono.
        ffmpeg: ffmpeg executable.

    Returns:
        `out_path`.
    """
    video = write_session_video(tmp_dir / f"{session.session_id}_video.mp4", session)
    if per_speaker_audio:
        streams = [
            write_session_wav(
                tmp_dir / f"{session.session_id}_{speaker}.wav", session, speakers=[speaker]
            )
            for speaker in session.speakers
        ]
    elif stereo_layout is not None:
        streams = [
            write_wav(
                tmp_dir / f"{session.session_id}_stereo.wav",
                session_stereo(session, stereo_layout),
                session.sample_rate,
            )
        ]
    else:
        streams = [write_session_wav(tmp_dir / f"{session.session_id}_mixed.wav", session)]
    return mux(out_path, video, streams, ffmpeg=ffmpeg)


# ---------------------------------------------------------------------------
# Diarization output
# ---------------------------------------------------------------------------
def _srt_timestamp(seconds: float) -> str:
    """Format seconds as an SRT timestamp (`HH:MM:SS,mmm`)."""
    if seconds < 0:
        msg = f"negative timestamp: {seconds}"
        raise ValueError(msg)
    total_ms = round(seconds * 1000)
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def srt_text(session: SyntheticSession, *, speaker_prefix: str = "") -> str:
    """Render a session as whisper-diarization style SRT.

    Speaker labels are carried in the cue text as `SPEAKER_00: ...`, which is
    what whisper-diarization emits.

    Args:
        session: Session to render.
        speaker_prefix: Optional prefix, for testing unfamiliar label styles.

    Returns:
        The SRT file contents.
    """
    blocks: list[str] = []
    for index, utterance in enumerate(session.utterances, start=1):
        label = f"{speaker_prefix}{utterance.speaker}"
        blocks.append(
            f"{index}\n"
            f"{_srt_timestamp(utterance.start)} --> {_srt_timestamp(utterance.end)}\n"
            f"{label}: {utterance.text}\n"
        )
    return "\n".join(blocks)


def write_srt(path: Path, session: SyntheticSession, *, speaker_prefix: str = "") -> Path:
    """Write a session's SRT file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(srt_text(session, speaker_prefix=speaker_prefix), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Labels, for the analysis half
# ---------------------------------------------------------------------------
def labels_frame(
    session_ids: Sequence[int],
    *,
    targets: Sequence[str] = ("K6", "SRS2"),
    seed: int = 0,
) -> pd.DataFrame:
    """Build a synthetic labels table as a DataFrame.

    Values are random: these tests check plumbing, validation and the absence of
    leakage, not predictive performance.

    Args:
        session_ids: Sessions to score.
        targets: Target column names.
        seed: Random seed.

    Returns:
        A pandas DataFrame with `session_id` and one column per target.
    """
    rng = np.random.default_rng(seed)
    data: dict[str, object] = {"session_id": list(session_ids)}
    for target in targets:
        data[target] = rng.normal(10.0, 4.0, size=len(session_ids)).round(2)
    return pd.DataFrame(data)
