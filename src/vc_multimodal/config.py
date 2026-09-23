"""Validated configuration.

Every value the pipeline needs is declared here as a pydantic model, loaded from
YAML (`config/default.yaml`), optionally overlaid with a second YAML file, and
optionally overridden by CLI flags. `extra="forbid"` throughout, so a typo in a
config file is an error rather than a silently ignored key.

Models are frozen: a stage receives configuration it cannot mutate, and the
resolved config is snapshotted into every run's output directory so that any
result can be traced back to the exact settings that produced it.

This module deliberately imports nothing else from the package, so it can be the
bottom layer that everything else builds on.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DEFAULT_CONFIG_PATH = Path("config/default.yaml")

# "left" and "right" only describe a layout that has exactly two tiles.
_SIDES_IN_A_TWO_TILE_LAYOUT = 2


class ConfigError(ValueError):
    """Raised when configuration is missing, malformed or self-inconsistent."""


class _Base(BaseModel):
    """Shared model settings: frozen, and strict about unknown keys."""

    model_config = ConfigDict(extra="forbid", frozen=True)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
class CropBox(_Base):
    """A rectangle in fractional frame coordinates.

    Fractions rather than pixels so that one config works across recordings of
    different resolutions. Origin is the top-left of the frame.
    """

    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    width: float = Field(gt=0.0, le=1.0)
    height: float = Field(gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _must_stay_inside_the_frame(self) -> Self:
        if self.x + self.width > 1.0 + 1e-9:
            msg = f"crop extends past the right edge: x={self.x} + width={self.width} > 1"
            raise ValueError(msg)
        if self.y + self.height > 1.0 + 1e-9:
            msg = f"crop extends past the bottom edge: y={self.y} + height={self.height} > 1"
            raise ValueError(msg)
        return self

    def to_pixels(self, frame_width: int, frame_height: int) -> tuple[int, int, int, int]:
        """Convert to integer pixels as `(left, top, width, height)`.

        Rounds the edges rather than the origin and size independently, so that
        adjacent boxes tile without a one-pixel seam or overlap. Guarantees a
        width and height of at least one pixel.
        """
        left = round(self.x * frame_width)
        top = round(self.y * frame_height)
        right = round((self.x + self.width) * frame_width)
        bottom = round((self.y + self.height) * frame_height)
        left = max(0, min(left, frame_width - 1))
        top = max(0, min(top, frame_height - 1))
        right = max(left + 1, min(right, frame_width))
        bottom = max(top + 1, min(bottom, frame_height))
        return left, top, right - left, bottom - top


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class WaveSpec(_Base):
    """One recruitment wave: its date folders and its session-ID range."""

    folders: tuple[str, ...]
    id_min: int = Field(ge=0)
    id_max: int = Field(ge=0)

    @model_validator(mode="after")
    def _range_must_be_ordered(self) -> Self:
        if self.id_min > self.id_max:
            msg = f"id_min={self.id_min} exceeds id_max={self.id_max}"
            raise ValueError(msg)
        return self

    def contains(self, session_id: int) -> bool:
        """Whether `session_id` falls in this wave's ID range."""
        return self.id_min <= session_id <= self.id_max


class DurationChecks(_Base):
    """Bounds used to flag recordings of unexpected length.

    Sessions are nominally 10-12 minutes. Both an absolute window and a robust
    (median absolute deviation) test are applied; either can raise a flag.
    """

    min_seconds: float = Field(gt=0.0)
    max_seconds: float = Field(gt=0.0)
    mad_k: float = Field(gt=0.0)

    @model_validator(mode="after")
    def _window_must_be_ordered(self) -> Self:
        if self.min_seconds >= self.max_seconds:
            msg = f"min_seconds={self.min_seconds} must be below max_seconds={self.max_seconds}"
            raise ValueError(msg)
        return self


class DatasetConfig(_Base):
    """What the raw data tree is expected to look like."""

    expected_sessions: int = Field(gt=0)
    media_glob: str
    waves: Mapping[str, WaveSpec]
    duration: DurationChecks
    known_short_sessions: tuple[int, ...] = ()

    @field_validator("waves")
    @classmethod
    def _waves_must_not_be_empty(cls, value: Mapping[str, WaveSpec]) -> Mapping[str, WaveSpec]:
        if not value:
            msg = "at least one wave must be defined"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def _folders_and_ranges_must_not_overlap(self) -> Self:
        seen_folders: dict[str, str] = {}
        for wave, spec in self.waves.items():
            for folder in spec.folders:
                if folder in seen_folders:
                    msg = (
                        f"folder {folder!r} is claimed by both "
                        f"{seen_folders[folder]!r} and {wave!r}"
                    )
                    raise ValueError(msg)
                seen_folders[folder] = wave
        waves = list(self.waves.items())
        for i, (name_a, a) in enumerate(waves):
            for name_b, b in waves[i + 1 :]:
                if a.id_min <= b.id_max and b.id_min <= a.id_max:
                    msg = f"session-ID ranges of {name_a!r} and {name_b!r} overlap"
                    raise ValueError(msg)
        return self

    @property
    def folders(self) -> tuple[str, ...]:
        """Every date folder, in wave order."""
        return tuple(folder for spec in self.waves.values() for folder in spec.folders)

    def wave_of_folder(self, folder: str) -> str | None:
        """Wave owning `folder`, or None if the folder is not configured."""
        for wave, spec in self.waves.items():
            if folder in spec.folders:
                return wave
        return None

    def wave_of_id(self, session_id: int) -> str | None:
        """Wave whose ID range contains `session_id`, or None."""
        for wave, spec in self.waves.items():
            if spec.contains(session_id):
                return wave
        return None


# ---------------------------------------------------------------------------
# Media handling
# ---------------------------------------------------------------------------
class VideoConfig(_Base):
    """Video layout and the tiles to crop.

    `layout` is a declaration to be confirmed by eye with `vc preview`, not an
    assumption baked into the code. `tiles` maps a tile name to a crop box;
    `participant_tile` and `psychiatrist_tile` name which tile is whose.
    """

    layout: Literal["gallery", "active_speaker", "unknown"]
    tiles: Mapping[str, CropBox]
    participant_tile: str
    psychiatrist_tile: str
    preview_times_seconds: tuple[float, ...]
    preview_max_width: int = Field(gt=0)
    preview_format: Literal["jpg", "png"]

    def tiles_left_to_right(self) -> tuple[tuple[str, CropBox], ...]:
        """Tiles ordered by position, so a "side" means what it looks like."""
        return tuple(sorted(self.tiles.items(), key=lambda item: (item[1].x, item[1].y)))

    @property
    def is_two_tile(self) -> bool:
        """Whether "left" and "right" sides are meaningful for this layout."""
        return len(self.tiles) == _SIDES_IN_A_TWO_TILE_LAYOUT

    def tile_on_side(self, side: str) -> str | None:
        """Name of the tile on `side`, or None if sides are not meaningful.

        Sides are physical positions in the frame, independent of what the tiles
        happen to be called in config.
        """
        if not self.is_two_tile:
            return None
        ordered = self.tiles_left_to_right()
        if side == "left":
            return ordered[0][0]
        if side == "right":
            return ordered[1][0]
        return None

    def side_of_tile(self, tile: str) -> str | None:
        """Which side `tile` sits on, or None if sides are not meaningful."""
        if not self.is_two_tile:
            return None
        ordered = [name for name, _ in self.tiles_left_to_right()]
        if tile not in ordered:
            return None
        return "left" if ordered.index(tile) == 0 else "right"

    @model_validator(mode="after")
    def _roles_must_name_real_tiles(self) -> Self:
        for role, tile in (
            ("participant_tile", self.participant_tile),
            ("psychiatrist_tile", self.psychiatrist_tile),
        ):
            if tile not in self.tiles:
                msg = f"{role}={tile!r} is not defined in tiles ({sorted(self.tiles)})"
                raise ValueError(msg)
        if self.participant_tile == self.psychiatrist_tile:
            msg = "participant_tile and psychiatrist_tile must differ"
            raise ValueError(msg)
        if not self.preview_times_seconds:
            msg = "preview_times_seconds must list at least one timestamp"
            raise ValueError(msg)
        return self


class AudioConfig(_Base):
    """Audio extraction settings.

    `stream_layout: auto` means the inventory stage decides per file whether a
    recording carries one mixed stream or one per speaker.
    """

    sample_rate: int = Field(gt=0)
    stream_layout: Literal["auto", "mixed", "per_speaker"]
    codec: str


# ---------------------------------------------------------------------------
# Diarization
# ---------------------------------------------------------------------------
class PyannoteConfig(_Base):
    """Local pyannote diarization. Requires the `pyannote` optional extra."""

    model: str
    revision: str | None
    num_speakers: int = Field(gt=0)
    token_env: str


class WhisperDiarizationConfig(_Base):
    """External whisper-diarization run, invoked as a subprocess.

    Intentionally not a Python dependency of this project: the upstream repo has
    heavy requirements that do not install cleanly on Apple silicon. It is run
    in its own environment and its output is read by the `import` backend.
    """

    command: tuple[str, ...]
    language: str
    output_dir: str | None
    timeout_seconds: float = Field(gt=0.0)


class DiarizationConfig(_Base):
    """Which diarization source to use, and how to read it."""

    backend: Literal["import", "pyannote", "whisper_diarization"]
    import_dir: str | None
    import_patterns: tuple[str, ...]
    keep_text: bool
    pyannote: PyannoteConfig
    whisper_diarization: WhisperDiarizationConfig

    def require_import_dir(self) -> str:
        """Return `import_dir`, or explain what the user must configure.

        Checked when the diarize stage runs rather than at load time: the
        directory is legitimately unknown until the original diarization output
        is available, and earlier stages must still be able to load config.

        Raises:
            ConfigError: if the import backend is selected without a directory.
        """
        if not self.import_dir:
            msg = (
                "diarization.backend='import' needs diarization.import_dir: the "
                "folder holding the whisper-diarization output. Keep it under "
                "$VC_WORK_ROOT."
            )
            raise ConfigError(msg)
        return self.import_dir


# ---------------------------------------------------------------------------
# Speaker assignment
# ---------------------------------------------------------------------------
class ReferenceClip(_Base):
    """A clean voice sample for one psychiatrist.

    Path is relative to `$VC_WORK_ROOT/reference/`. Several clips are supported
    because it is not yet confirmed that the same psychiatrist ran every
    session, particularly across the two recruitment waves.
    """

    psychiatrist_id: str
    path: str


class MouthCrosscheckConfig(_Base):
    """Cross-check speaker assignment against visible mouth movement.

    In gallery view both faces are visible, so each diarized speaker's speech
    timeline is correlated against *every* configured tile, not just the
    participant's. The expected pattern is a clean assignment: each speaker
    correlates with exactly one tile. That is a far stronger check than
    correlating against one tile alone.
    """

    enabled: bool
    tiles: tuple[str, ...] | None
    sample_fps: float = Field(gt=0.0)
    min_correlation: float = Field(ge=0.0, le=1.0)
    min_separation: float = Field(ge=0.0, le=1.0)


class LabelOcrConfig(_Base):
    """On-device OCR of the Zoom name label in each video tile.

    Used by `vc verify-layout` to determine which side the psychiatrist is on.
    Recognised text is compared in memory and never printed, logged or written:
    the labels are people's names.

    The psychiatrist is identified WITHOUT being named, by the fact that their
    label recurs across sessions while participants' labels do not. An explicit
    pattern can be supplied through the environment for the cases where that is
    not enough, but never through a committed config file.
    """

    enabled: bool
    backend: Literal["apple_vision", "tesseract", "none"]
    languages: tuple[str, ...]
    # Region within a tile holding the name label, in coordinates relative to
    # that tile. Null means search the whole tile.
    label_region: CropBox | None
    sample_times_seconds: tuple[float, ...]
    min_confidence: float = Field(ge=0.0, le=1.0)
    # A label must appear in at least this fraction of sessions to be treated as
    # the recurring one, i.e. the person present in every session.
    min_recurrence: float = Field(gt=0.0, le=1.0)
    # Name of an environment variable holding optional explicit label patterns,
    # comma-separated. Never a value: a real name does not belong in this repo.
    psychiatrist_label_env: str

    @model_validator(mode="after")
    def _needs_at_least_one_timestamp(self) -> Self:
        if self.enabled and not self.sample_times_seconds:
            msg = "label_ocr.sample_times_seconds must list at least one timestamp"
            raise ValueError(msg)
        if self.enabled and not self.languages:
            msg = "label_ocr.languages must list at least one language"
            raise ValueError(msg)
        return self


class SpeakerAssignConfig(_Base):
    """How diarized speaker labels become roles.

    Embedding similarity against a reference clip is primary; mouth-movement
    correlation is an independent cross-check. Agreement between the two, and
    the embedding margin, are recorded as QC columns, and disagreement or a
    small margin raises a flag rather than silently choosing.
    """

    embedding_model: str
    embedding_token_env: str
    # Which side the psychiatrist is expected to be on. A default derived from
    # the sessions checked by hand, not a guarantee: it is used only where OCR
    # is unavailable or inconclusive, and any disagreement with OCR is recorded
    # as a QC flag rather than silently overriding it.
    assumed_psychiatrist_side: Literal["left", "right"]
    label_ocr: LabelOcrConfig
    reference_clips: tuple[ReferenceClip, ...]
    session_psychiatrist_map: str | None
    min_margin: float = Field(ge=0.0)
    mouth_crosscheck: MouthCrosscheckConfig


# ---------------------------------------------------------------------------
# Speech timing and features
# ---------------------------------------------------------------------------
class VadConfig(_Base):
    """Silero VAD parameters, applied inside diarized segments."""

    threshold: float = Field(gt=0.0, lt=1.0)
    min_speech_ms: int = Field(gt=0)
    min_silence_ms: int = Field(gt=0)
    speech_pad_ms: int = Field(ge=0)


class TurnsConfig(_Base):
    """Turn construction and timing thresholds."""

    merge_same_speaker_gap_s: float = Field(ge=0.0)
    min_pause_s: float = Field(gt=0.0)
    max_latency_s: float = Field(gt=0.0)
    min_overlap_s: float = Field(gt=0.0)


class OpenSmileConfig(_Base):
    """Extension point for openSMILE eGeMAPS. Off until a binary is available."""

    enabled: bool
    feature_set: str
    executable: str | None


class ProsodyConfig(_Base):
    """Prosodic analysis of participant speech only, with overlaps excluded."""

    f0_floor_hz: float = Field(gt=0.0)
    f0_ceiling_hz: float = Field(gt=0.0)
    semitone_reference: Literal["speaker_median"]
    exclude_overlap: bool
    min_analysis_s: float = Field(gt=0.0)
    opensmile: OpenSmileConfig

    @model_validator(mode="after")
    def _f0_range_must_be_ordered(self) -> Self:
        if self.f0_floor_hz >= self.f0_ceiling_hz:
            msg = f"f0_floor_hz={self.f0_floor_hz} must be below f0_ceiling_hz={self.f0_ceiling_hz}"
            raise ValueError(msg)
        return self


class MediaPipeConfig(_Base):
    """MediaPipe Face Landmarker settings and the pinned model asset."""

    model_asset: str
    model_url: str
    model_sha256: str | None
    blendshapes: tuple[str, ...]
    head_pose: bool


class OpenFaceConfig(_Base):
    """Importer for OpenFace 2.0 CSV output produced on a lab machine."""

    csv_dir: str | None
    confidence_column: str
    success_column: str


class FaceConfig(_Base):
    """Frame sampling, cropping and the face-landmark backend."""

    backend: Literal["mediapipe", "openface"]
    sample_fps: float = Field(gt=0.0)
    min_confidence: float = Field(ge=0.0, le=1.0)
    max_dropped_fraction: float = Field(ge=0.0, le=1.0)
    mediapipe: MediaPipeConfig
    openface: OpenFaceConfig


class AggregateConfig(_Base):
    """Per-session summarisation.

    The feature count is kept deliberately modest: with 62 sessions, a large
    feature set overfits. See docs/decisions/0006.
    """

    stats: tuple[str, ...]
    min_speaking_s: float = Field(gt=0.0)
    min_listening_s: float = Field(gt=0.0)
    max_features: int = Field(gt=0)


# ---------------------------------------------------------------------------
# Handoff and analysis
# ---------------------------------------------------------------------------
class HandoffConfig(_Base):
    """Bundle construction."""

    bundle_dir: str
    allow_dirty: bool


class ModelConfig(_Base):
    """The analysis the label holder runs.

    `grouping='session'` treats each session as one participant, which is the
    documented structure of this dataset (62 participants, 62 sessions, unique
    IDs). Supplying `participant_map` overrides that with an explicit
    session-to-participant mapping, so the assumption is configurable rather
    than hardcoded.
    """

    grouping: Literal["session", "participant_map"]
    participant_map: str | None
    targets: tuple[str, ...]
    feature_sets: Mapping[str, tuple[str, ...]]
    models: tuple[Literal["elastic_net", "random_forest"], ...]
    n_permutations: int = Field(ge=0)
    text_features: str | None

    @model_validator(mode="after")
    def _explicit_grouping_needs_a_map(self) -> Self:
        if self.grouping == "participant_map" and not self.participant_map:
            msg = "model.grouping='participant_map' requires model.participant_map"
            raise ValueError(msg)
        if not self.models:
            msg = "at least one model must be configured"
            raise ValueError(msg)
        if not self.targets:
            msg = "at least one target must be configured"
            raise ValueError(msg)
        return self


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
class RuntimeConfig(_Base):
    """Execution settings shared by every stage."""

    workers: int | None
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"]
    seed: int
    pilot_sessions: tuple[int, ...]

    @field_validator("workers")
    @classmethod
    def _workers_must_be_positive(cls, value: int | None) -> int | None:
        if value is not None and value < 1:
            msg = "runtime.workers must be at least 1, or null to choose automatically"
            raise ValueError(msg)
        return value


class AppConfig(_Base):
    """The complete resolved configuration for a run."""

    dataset: DatasetConfig
    video: VideoConfig
    audio: AudioConfig
    diarization: DiarizationConfig
    speakers: SpeakerAssignConfig
    vad: VadConfig
    turns: TurnsConfig
    prosody: ProsodyConfig
    face: FaceConfig
    aggregate: AggregateConfig
    handoff: HandoffConfig
    model: ModelConfig
    runtime: RuntimeConfig

    def snapshot(self) -> dict[str, Any]:
        """Plain-data view of the resolved config, for the run manifest."""
        return self.model_dump(mode="json")

    def to_yaml(self) -> str:
        """The resolved config as YAML, for the config snapshot file."""
        return yaml.safe_dump(self.snapshot(), sort_keys=True, allow_unicode=True)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge `overlay` onto `base`, returning a new dict.

    Mappings merge key by key; every other value, including lists, is replaced
    wholesale. Replacing lists is intentional: an overlay that sets
    `runtime.pilot_sessions` means exactly those sessions, not those appended to
    the defaults.
    """
    merged = dict(base)
    for key, value in overlay.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def read_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping from `path`.

    Raises:
        ConfigError: if the file is missing, unreadable, or not a mapping.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        msg = f"config file not found: {path}"
        raise ConfigError(msg) from exc
    except OSError as exc:  # pragma: no cover - unreadable file
        msg = f"could not read config file {path}: {exc}"
        raise ConfigError(msg) from exc
    try:
        loaded = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        msg = f"invalid YAML in {path}: {exc}"
        raise ConfigError(msg) from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, Mapping):
        msg = f"config file {path} must contain a mapping at the top level"
        raise ConfigError(msg)
    return dict(loaded)


def set_by_dotted_path(data: dict[str, Any], dotted: str, value: Any) -> dict[str, Any]:
    """Return a copy of `data` with `dotted` (e.g. `face.sample_fps`) set.

    Raises:
        ConfigError: if the path is empty or traverses a non-mapping.
    """
    parts = [part for part in dotted.split(".") if part]
    if not parts:
        msg = f"empty config override path: {dotted!r}"
        raise ConfigError(msg)
    result = dict(data)
    cursor = result
    for part in parts[:-1]:
        existing = cursor.get(part)
        if existing is not None and not isinstance(existing, Mapping):
            msg = f"cannot set {dotted!r}: {part!r} is not a section"
            raise ConfigError(msg)
        child = dict(existing) if isinstance(existing, Mapping) else {}
        cursor[part] = child
        cursor = child
    cursor[parts[-1]] = value
    return result


def load_config(
    path: Path | None = None,
    *,
    overlays: Iterable[Path] = (),
    overrides: Mapping[str, Any] | None = None,
) -> AppConfig:
    """Load, merge and validate configuration.

    Precedence, lowest to highest: `path`, each overlay in order, then
    `overrides` (CLI flags, as dotted paths).

    Args:
        path: Base config file. Defaults to `config/default.yaml`.
        overlays: Additional config files merged onto the base, e.g. a pilot
            config.
        overrides: Dotted-path values from the command line.

    Returns:
        The validated configuration.

    Raises:
        ConfigError: if any file is missing or the merged result is invalid.
    """
    base_path = DEFAULT_CONFIG_PATH if path is None else path
    data = read_yaml(base_path)
    for overlay in overlays:
        data = deep_merge(data, read_yaml(overlay))
    for dotted, value in (overrides or {}).items():
        data = set_by_dotted_path(data, dotted, value)
    try:
        return AppConfig.model_validate(data)
    except Exception as exc:
        sources = ", ".join(str(p) for p in [base_path, *overlays])
        msg = f"invalid configuration from {sources}:\n{exc}"
        raise ConfigError(msg) from exc
