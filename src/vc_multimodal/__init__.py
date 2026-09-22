"""Multimodal feature extraction for clinical Zoom sessions.

The package is split into two halves that never share a machine:

* Extraction (`inventory` ... `handoff`) turns raw video into a per-session
  feature table. It runs with no questionnaire labels present anywhere.
* Analysis (`model`) joins that feature table with privately held labels and
  reports cross-validated metrics only.

See `docs/decisions/0001-handoff-split-no-labels-on-student-machine.md`.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
