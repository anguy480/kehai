"""Frame sampling arithmetic.

Every recording runs at a constant 25 fps, and a configured sample rate that
does not divide it exactly cannot be honoured: 10 fps at 25 fps native means
alternating 2- and 3-frame steps. Uneven spacing quietly distorts anything
derived from differences between frames, and it does so without any error to
notice, so a rate that does not divide the native rate is rejected rather than
rounded.

Pure arithmetic: no video decoding, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

# How far from a whole number a computed step may sit and still count as exact.
# Frame rates arrive as ratios such as 30000/1001, so an exact equality test on
# floats would reject rates that are exact in practice.
STEP_TOLERANCE: Final = 1e-6

# Candidate rates offered in an error message, as divisors of the native rate.
_SUGGESTED_DIVISORS: Final = (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25)


class SamplingError(ValueError):
    """Raised when a configured sample rate cannot be honoured exactly."""


@dataclass(frozen=True, slots=True)
class FrameSampling:
    """A resolved sampling plan for one recording.

    Attributes:
        native_fps: The recording's own frame rate.
        requested_fps: The rate asked for in configuration.
        step: Take every `step`-th frame.
        effective_fps: The rate actually achieved, `native_fps / step`. Equal to
            `requested_fps` by construction, and recorded in QC so the
            configured value can never silently misdescribe what happened.
    """

    native_fps: float
    requested_fps: float
    step: int
    effective_fps: float

    def frame_indices(self, n_frames: int) -> range:
        """Indices of the frames to sample from a recording of `n_frames`."""
        return range(0, max(n_frames, 0), self.step)

    def n_sampled(self, n_frames: int) -> int:
        """How many frames would be sampled from `n_frames`."""
        return len(self.frame_indices(n_frames))

    def timestamp(self, frame_index: int) -> float:
        """Time in seconds of a frame, by its index in the original recording."""
        return frame_index / self.native_fps if self.native_fps else 0.0


def _suggestions(native_fps: float) -> list[float]:
    """Sample rates that do divide `native_fps` exactly, for an error message."""
    rates = []
    for divisor in _SUGGESTED_DIVISORS:
        rate = native_fps / divisor
        if rate >= 1.0:
            rates.append(round(rate, 4))
    return rates


def resolve_sampling(native_fps: float | None, requested_fps: float) -> FrameSampling:
    """Work out the integer frame step for a requested sample rate.

    Args:
        native_fps: The recording's frame rate. None or non-positive is an
            error: the step cannot be known, and guessing one would produce
            timestamps that silently disagree with the recording.
        requested_fps: The configured sample rate.

    Returns:
        The resolved sampling plan.

    Raises:
        SamplingError: if the native rate is unknown, the requested rate is not
            positive, exceeds the native rate, or does not divide it exactly.
    """
    if native_fps is None or native_fps <= 0.0:
        msg = (
            "cannot resolve frame sampling without the recording's frame rate; "
            "run `vc inventory` first, and check it for a variable-frame-rate flag"
        )
        raise SamplingError(msg)
    if requested_fps <= 0.0:
        msg = f"the sample rate must be positive, got {requested_fps}"
        raise SamplingError(msg)
    if requested_fps > native_fps + STEP_TOLERANCE:
        msg = (
            f"cannot sample at {requested_fps} fps from a {native_fps} fps recording: "
            f"there are not that many frames"
        )
        raise SamplingError(msg)

    exact_step = native_fps / requested_fps
    step = round(exact_step)
    if step < 1 or abs(exact_step - step) > STEP_TOLERANCE:
        msg = (
            f"a sample rate of {requested_fps} fps does not divide the recording's "
            f"{native_fps} fps: it would need every {exact_step:.4f}th frame, so the "
            f"spacing would alternate rather than be even. Rates that divide "
            f"{native_fps} exactly: {_suggestions(native_fps)}"
        )
        raise SamplingError(msg)

    return FrameSampling(
        native_fps=native_fps,
        requested_fps=requested_fps,
        step=step,
        effective_fps=native_fps / step,
    )
