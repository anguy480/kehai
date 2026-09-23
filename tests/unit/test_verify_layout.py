"""The layout check: label handling, the cohort rule, and what may be output.

The decision logic is pure and tested directly. The stage is tested with an
injected fake OCR backend, so nothing here depends on a platform OCR engine and
CI never needs one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tests.conftest import (
    SUMMER_FOLDER,
    WINTER_FOLDER,
    PackageLogCapture,
)
from tests.synth import generators as gen
from tests.synth.fake_ocr import FakeOcr, SideScriptedOcr
from vc_multimodal.config import AppConfig, CropBox, load_config
from vc_multimodal.contracts import LAYOUT_SCHEMA, validate
from vc_multimodal.features.geometry import resolve_regions
from vc_multimodal.ffmpeg import FfmpegTools
from vc_multimodal.ocr import get_backend
from vc_multimodal.paths import DataRoots, discover_sessions
from vc_multimodal.stages import verify_layout as stage

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT = REPO_ROOT / "config" / "default.yaml"

DOCTOR = "drsato"
GUEST = "guest028"


# ---------------------------------------------------------------------------
# label normalisation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Dr Sato", "drsato"),
        ("DR. SATO", "drsato"),
        ("  dr   sato  ", "drsato"),
        ("Dr-Sato", "drsato"),
        ("Dr_Sato", "drsato"),
        ("佐藤　太郎", "佐藤太郎"),
        ("佐藤・太郎", "佐藤太郎"),
        ("佐藤（医師）", "佐藤医師"),
        ("ＤＲ　ＳＡＴＯ", "drsato"),
        ("GUEST 028", "guest028"),
    ],
)
def test_labels_normalise_to_a_comparable_key(raw: str, expected: str):
    assert stage.normalise_label(raw) == expected


def test_variants_of_one_name_share_a_key():
    """OCR of the same label differs across sessions; the key must not."""
    keys = {
        stage.normalise_label(variant)
        for variant in ("Dr Sato", "dr. sato", "DR  SATO", "Ｄｒ．Ｓａｔｏ")
    }
    assert len(keys) == 1


def test_different_names_do_not_collide():
    assert stage.normalise_label("Dr Sato") != stage.normalise_label("Dr Suto")


@pytest.mark.parametrize("raw", ["", "   ", "()", "（）", "・・・", "---", "　"])
def test_text_with_nothing_comparable_yields_an_empty_key(raw: str):
    assert stage.normalise_label(raw) == ""


# ---------------------------------------------------------------------------
# the recurrence rule: identifying the psychiatrist without naming them
# ---------------------------------------------------------------------------
def test_the_label_present_in_every_session_is_the_recurring_one():
    per_session = {sid: (DOCTOR, f"guest{sid:03d}") for sid in range(1, 11)}
    assert stage.find_recurring_labels(per_session, min_recurrence=0.5) == frozenset({DOCTOR})


def test_a_label_in_half_the_sessions_still_recurs():
    per_session = {
        sid: ((DOCTOR,) if sid <= 5 else ()) + (f"guest{sid:03d}",) for sid in range(1, 11)
    }
    assert DOCTOR in stage.find_recurring_labels(per_session, min_recurrence=0.5)


def test_a_label_below_the_threshold_does_not_recur():
    per_session = {
        sid: ((DOCTOR,) if sid <= 2 else ()) + (f"guest{sid:03d}",) for sid in range(1, 11)
    }
    assert stage.find_recurring_labels(per_session, min_recurrence=0.5) == frozenset()


def test_two_psychiatrists_across_two_waves_both_recur():
    """It is not confirmed that one psychiatrist ran every session."""
    per_session = {sid: ("drsato", f"guest{sid:03d}") for sid in range(1, 6)}
    per_session.update({sid: ("drsuzuki", f"guest{sid:03d}") for sid in range(102, 107)})
    recurring = stage.find_recurring_labels(per_session, min_recurrence=0.4)
    assert recurring == frozenset({"drsato", "drsuzuki"})


def test_a_repeated_label_within_one_session_counts_once():
    """Reading the same label at three timestamps is not three sessions."""
    per_session = {1: (DOCTOR, DOCTOR, DOCTOR), 2: ("other",), 3: ("other2",)}
    assert stage.find_recurring_labels(per_session, min_recurrence=0.6) == frozenset()


def test_recurrence_needs_at_least_two_sessions():
    assert stage.find_recurring_labels({1: (DOCTOR,)}, min_recurrence=0.5) == frozenset()


def test_no_sessions_yields_nothing():
    assert stage.find_recurring_labels({}, min_recurrence=0.5) == frozenset()


# ---------------------------------------------------------------------------
# per-session side decision
# ---------------------------------------------------------------------------
def test_the_recurring_label_on_the_left_means_psychiatrist_left():
    decision = stage.decide_side([DOCTOR], [GUEST], recurring=frozenset({DOCTOR}))
    assert decision.side == stage.SIDE_LEFT
    assert decision.flags == ()


def test_the_recurring_label_on_the_right_means_psychiatrist_right():
    decision = stage.decide_side([GUEST], [DOCTOR], recurring=frozenset({DOCTOR}))
    assert decision.side == stage.SIDE_RIGHT


def test_no_recurring_label_anywhere_is_inconclusive():
    decision = stage.decide_side(["a"], ["b"], recurring=frozenset({DOCTOR}))
    assert decision.side == stage.SIDE_INCONCLUSIVE
    assert stage.FLAG_INCONCLUSIVE in decision.flags


def test_the_recurring_label_on_both_sides_is_inconclusive_and_flagged():
    decision = stage.decide_side([DOCTOR], [DOCTOR], recurring=frozenset({DOCTOR}))
    assert decision.side == stage.SIDE_INCONCLUSIVE
    assert stage.FLAG_BOTH_SIDES in decision.flags


def test_no_labels_at_all_is_inconclusive():
    assert stage.decide_side([], [], recurring=frozenset({DOCTOR})).side == stage.SIDE_INCONCLUSIVE


def test_explicit_patterns_take_precedence_over_recurrence():
    """A configured pattern expresses direct knowledge, so it wins."""
    decision = stage.decide_side(
        [GUEST], ["drsuzuki"], recurring=frozenset({GUEST}), patterns=["suzuki"]
    )
    assert decision.side == stage.SIDE_RIGHT


def test_patterns_match_a_fragment_of_the_label():
    decision = stage.decide_side(["drsatotaro"], [GUEST], patterns=["sato"])
    assert decision.side == stage.SIDE_LEFT


def test_an_empty_pattern_matches_nothing():
    assert not stage.matches_any_pattern(DOCTOR, [""])


# ---------------------------------------------------------------------------
# combining OCR with the assumption
# ---------------------------------------------------------------------------
def test_agreement_records_the_ocr_finding():
    side, method, matches, flags = stage.resolve(stage.SideDecision(stage.SIDE_LEFT), "left")
    assert (side, method, matches) == ("left", stage.METHOD_OCR, True)
    assert flags == []


def test_disagreement_keeps_the_ocr_finding_and_flags_it():
    """The assumption must never silently override what OCR found."""
    side, method, matches, flags = stage.resolve(stage.SideDecision(stage.SIDE_RIGHT), "left")
    assert side == "right"
    assert method == stage.METHOD_OCR
    assert matches is False
    assert stage.FLAG_MISMATCH in flags


def test_an_inconclusive_result_falls_back_to_the_assumption():
    side, method, matches, flags = stage.resolve(
        stage.SideDecision(stage.SIDE_INCONCLUSIVE, (stage.FLAG_INCONCLUSIVE,)), "left"
    )
    assert side == "left"
    assert method == stage.METHOD_ASSUMED
    assert matches is None
    assert stage.FLAG_INCONCLUSIVE in flags


def test_the_fallback_follows_the_configured_side():
    side, _, _, _ = stage.resolve(stage.SideDecision(stage.SIDE_INCONCLUSIVE), "right")
    assert side == "right"


# ---------------------------------------------------------------------------
# rows built from a cohort
# ---------------------------------------------------------------------------
def _labels(session_id: int, left: tuple[str, ...], right: tuple[str, ...]) -> stage.SessionLabels:
    return stage.SessionLabels(
        session_id=session_id,
        by_side={
            stage.SIDE_LEFT: stage.TileLabels(left, 0.9 if left else 0.0),
            stage.SIDE_RIGHT: stage.TileLabels(right, 0.9 if right else 0.0),
        },
    )


def test_rows_capture_the_decision_and_the_evidence():
    labels = {
        28: _labels(28, (DOCTOR,), ("guest028",)),
        3: _labels(3, ("guest003",), (DOCTOR,)),
    }
    rows = stage.build_rows(
        labels,
        {28: "winter", 3: "winter"},
        recurring=frozenset({DOCTOR}),
        assumed_side="left",
    )
    by_id = {int(row["session_id"]): row for row in rows}

    assert by_id[28]["decided_side"] == "left"
    assert by_id[28]["matches_assumed"] is True
    assert by_id[28]["flags"] == ""

    assert by_id[3]["decided_side"] == "right"
    assert by_id[3]["matches_assumed"] is False
    assert stage.FLAG_MISMATCH in str(by_id[3]["flags"])


def test_rows_validate_against_the_contract():
    labels = {28: _labels(28, (DOCTOR,), ("guest028",))}
    frame = stage.build_frame(
        stage.build_rows(labels, {28: "winter"}, recurring=frozenset({DOCTOR}), assumed_side="left")
    )
    validate(frame, LAYOUT_SCHEMA)


def test_the_layout_table_has_no_text_column():
    """Recognised labels are people's names and must not reach any file."""
    assert "text" not in LAYOUT_SCHEMA.columns
    assert "label" not in LAYOUT_SCHEMA.columns
    labels = {28: _labels(28, (DOCTOR,), ("guest028",))}
    frame = stage.build_frame(
        stage.build_rows(labels, {28: "winter"}, recurring=frozenset({DOCTOR}), assumed_side="left")
    )
    dumped = frame.to_csv(index=False)
    assert DOCTOR not in dumped
    assert "guest028" not in dumped


def test_a_session_flagged_during_reading_stays_inconclusive():
    labels = {
        28: stage.SessionLabels(28, {}, (stage.FLAG_OCR_UNAVAILABLE, stage.FLAG_INCONCLUSIVE))
    }
    rows = stage.build_rows(labels, {28: "winter"}, assumed_side="left")
    assert rows[0]["ocr_side"] == stage.SIDE_INCONCLUSIVE
    assert rows[0]["method"] == stage.METHOD_ASSUMED
    assert stage.FLAG_OCR_UNAVAILABLE in str(rows[0]["flags"])


# ---------------------------------------------------------------------------
# explicit patterns come from the environment, never from config
# ---------------------------------------------------------------------------
def test_patterns_are_read_from_the_configured_environment_variable(
    default_config: AppConfig, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("VC_PSYCHIATRIST_LABELS", "Dr Sato, 佐藤")
    assert stage.explicit_patterns(default_config) == ("drsato", "佐藤")


def test_no_patterns_when_the_variable_is_unset(
    default_config: AppConfig, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("VC_PSYCHIATRIST_LABELS", raising=False)
    assert stage.explicit_patterns(default_config) == ()


def test_the_shipped_config_contains_no_name(default_config: AppConfig):
    """A real person's name must never be committed to this repository."""
    text = (REPO_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    assert default_config.speakers.label_ocr.psychiatrist_label_env == "VC_PSYCHIATRIST_LABELS"
    # The config names the variable, and gives no value for it.
    assert "VC_PSYCHIATRIST_LABELS" in text
    assert 'psychiatrist_label_env: "VC_PSYCHIATRIST_LABELS"' in text


# ---------------------------------------------------------------------------
# cropping
# ---------------------------------------------------------------------------
def test_the_label_region_is_cropped_from_the_bottom_of_a_tile():
    tile = np.zeros((100, 200, 3), dtype=np.uint8)
    tile[82:, :120] = 255
    region = CropBox(x=0.0, y=0.82, width=0.6, height=0.18)
    patch = stage.crop_label_region(tile, region)
    assert patch.shape[:2] == (18, 120)
    assert patch.mean() == 255


def test_no_region_means_the_whole_tile():
    tile = np.zeros((100, 200, 3), dtype=np.uint8)
    assert stage.crop_label_region(tile, None).shape == tile.shape


# ---------------------------------------------------------------------------
# summary output
# ---------------------------------------------------------------------------
def _frame(**sides: str) -> Any:
    labels = {}
    for session_id_text, side in sides.items():
        session_id = int(session_id_text.lstrip("s"))
        if side == "left":
            labels[session_id] = _labels(session_id, (DOCTOR,), (f"guest{session_id:03d}",))
        elif side == "right":
            labels[session_id] = _labels(session_id, (f"guest{session_id:03d}",), (DOCTOR,))
        else:
            labels[session_id] = _labels(session_id, (), ())
    waves = dict.fromkeys(labels, "winter")
    return stage.build_frame(
        stage.build_rows(labels, waves, recurring=frozenset({DOCTOR}), assumed_side="left")
    )


def test_summary_counts_each_side_with_session_ids(default_config: AppConfig):
    frame = _frame(s1="left", s2="left", s3="right", s4="none")
    text = "\\n".join(stage.summarise(frame, default_config))
    assert "checked 4 session(s)" in text
    assert "left" in text and "[1, 2]" in text
    assert "[3]" in text
    assert "inconclusive" in text and "[4]" in text


def test_summary_reports_a_mismatch_as_flagged_not_applied(default_config: AppConfig):
    text = "\\n".join(stage.summarise(_frame(s1="left", s2="right"), default_config))
    assert "DISAGREES" in text
    assert "[2]" in text
    assert "not overridden" in text


def test_summary_confirms_the_assumption_when_every_session_agrees(default_config: AppConfig):
    text = "\\n".join(stage.summarise(_frame(s1="left", s2="left", s3="left"), default_config))
    assert "All 3 session(s) confirmed" in text
    assert "Later stages can rely on it" in text


def test_summary_says_the_assumption_is_untested_when_nothing_was_settled(
    default_config: AppConfig,
):
    text = "\\n".join(stage.summarise(_frame(s1="none", s2="none"), default_config))
    assert "untested" in text


def test_summary_does_not_claim_confirmation_when_some_sessions_disagree(
    default_config: AppConfig,
):
    text = "\\n".join(stage.summarise(_frame(s1="left", s2="right"), default_config))
    assert "confirmed" not in text
    assert "Resolve those" in text


def test_summary_never_contains_recognised_text(default_config: AppConfig):
    text = "\\n".join(stage.summarise(_frame(s1="left", s2="right"), default_config))
    assert DOCTOR not in text
    assert "guest" not in text


def test_summary_of_nothing(default_config: AppConfig):
    assert stage.summarise(pd.DataFrame(), default_config) == ["no sessions were checked"]


# ---------------------------------------------------------------------------
# running the stage, with an injected fake backend
# ---------------------------------------------------------------------------
@pytest.fixture
def cohort(make_real_media: Any) -> list[int]:
    """Three recordings with Zoom-style name labels drawn in each tile."""
    make_real_media(28, folder=WINTER_FOLDER, duration=12.0)
    make_real_media(3, folder="January 17 2026", duration=12.0)
    make_real_media(210, folder=SUMMER_FOLDER, duration=12.0)
    return [3, 28, 210]


@pytest.mark.slow
def test_run_decides_every_session_and_writes_the_table(
    roots: DataRoots, default_config: AppConfig, cohort: list[int]
):
    backend = SideScriptedOcr(left="Dr Sato", right="Guest")

    result = stage.run(default_config, roots, workers=1, backend=backend)

    assert result.report.ok
    assert result.path == roots.out / "layout.csv"
    assert sorted(result.frame["session_id"]) == cohort
    validate(result.frame, LAYOUT_SCHEMA)
    # "Dr Sato" is read in every session and "Guest" also recurs, but only the
    # left side carries the label that appears first; both recur, so this
    # exercises the both-sides path being flagged rather than guessed.
    assert set(result.frame["method"]) <= {stage.METHOD_OCR, stage.METHOD_ASSUMED}


@pytest.mark.slow
def test_a_unique_participant_label_lets_the_psychiatrist_be_identified(
    roots: DataRoots, default_config: AppConfig, cohort: list[int]
):
    """The real structure: one recurring label, a different one each session."""
    sides = {sid: SideScriptedOcr(left="Dr Sato", right=f"Guest {sid:03d}") for sid in cohort}

    # The stage takes one backend for the whole run, so the per-session reads
    # are driven directly here. Everything after that is the same code the
    # stage calls.
    tools = FfmpegTools.discover()
    discovered = {
        session.session_id: session
        for session in discover_sessions(roots.data, default_config.dataset).sessions
    }
    labels = {
        sid: stage.read_session_labels(
            discovered[sid],
            config=default_config,
            backend=sides[sid],
            tools=tools,
            scratch=roots.work_path("tmp", "test", str(sid)),
        )[0]
        for sid in cohort
    }

    recurring = stage.find_recurring_labels(
        {sid: result.all_labels for sid, result in labels.items()}, min_recurrence=0.5
    )
    assert recurring == frozenset({"drsato"})

    rows = stage.build_rows(
        labels, dict.fromkeys(cohort, "winter"), recurring=recurring, assumed_side="left"
    )
    assert {row["decided_side"] for row in rows} == {"left"}
    assert all(row["matches_assumed"] is True for row in rows)


@pytest.mark.slow
def test_unavailable_ocr_falls_back_to_the_assumed_side(
    roots: DataRoots, default_config: AppConfig, cohort: list[int]
):
    backend = FakeOcr(is_available=False)

    result = stage.run(default_config, roots, workers=1, backend=backend)

    assert result.report.ok
    assert set(result.frame["method"]) == {stage.METHOD_ASSUMED}
    assert set(result.frame["decided_side"]) == {"left"}
    assert result.frame["matches_assumed"].isna().all()
    assert all(stage.FLAG_OCR_UNAVAILABLE in flags for flags in result.frame["flags"])
    assert any("OCR unavailable" in note for note in result.report.notes)
    assert backend.calls == 0


@pytest.mark.slow
def test_low_confidence_lines_are_ignored(
    roots: DataRoots, default_config: AppConfig, cohort: list[int]
):
    config = load_config(DEFAULT, overrides={"speakers.label_ocr.min_confidence": 0.8})
    backend = SideScriptedOcr(left="Dr Sato", right="Guest", confidence=0.2)

    result = stage.run(config, roots, workers=1, backend=backend)

    assert backend.calls > 0
    assert set(result.frame["ocr_side"]) == {stage.SIDE_INCONCLUSIVE}
    assert (result.frame["n_labels_left"] == 0).all()


@pytest.mark.slow
def test_a_failing_ocr_read_does_not_fail_the_session(
    roots: DataRoots, default_config: AppConfig, cohort: list[int]
):
    backend = FakeOcr(fail_after=0)

    result = stage.run(default_config, roots, workers=1, backend=backend)

    assert result.report.ok  # the stage completes
    assert set(result.frame["ocr_side"]) == {stage.SIDE_INCONCLUSIVE}
    assert all(stage.FLAG_OCR_ERROR in flags for flags in result.frame["flags"])


@pytest.mark.slow
def test_a_layout_that_is_not_two_tile_is_reported_as_such(roots: DataRoots, cohort: list[int]):
    """Sides are meaningless for a three-tile layout, so it is not guessed."""
    config = load_config(
        DEFAULT,
        overrides={
            "video.tiles": {
                "left": {"x": 0.0, "y": 0.0, "width": 0.34, "height": 1.0},
                "middle": {"x": 0.34, "y": 0.0, "width": 0.32, "height": 1.0},
                "right": {"x": 0.66, "y": 0.0, "width": 0.34, "height": 1.0},
            }
        },
    )
    result = stage.run(config, roots, workers=1, backend=SideScriptedOcr("Dr Sato", "Guest"))

    assert all(stage.FLAG_NOT_TWO_TILE in flags for flags in result.frame["flags"])
    assert set(result.frame["method"]) == {stage.METHOD_ASSUMED}


@pytest.mark.slow
def test_no_extracted_frame_is_left_behind(
    roots: DataRoots, default_config: AppConfig, cohort: list[int]
):
    stage.run(default_config, roots, workers=1, backend=SideScriptedOcr("Dr Sato", "Guest"))
    leftovers = [
        path for path in roots.work.rglob("*") if path.is_file() and path.suffix in {".png", ".jpg"}
    ]
    assert leftovers == []


@pytest.mark.slow
def test_recognised_text_reaches_neither_the_table_nor_the_log(
    roots: DataRoots,
    default_config: AppConfig,
    cohort: list[int],
    package_logs: PackageLogCapture,
):
    """The central safety property of this stage.

    Uses `package_logs` rather than pytest's `caplog`: the package logger does
    not propagate to root once logging is configured, so `caplog.text` would be
    empty and this test would pass without checking anything.
    """
    planted_left, planted_right = "Dr Sato Taro", "Guest Kobayashi 028"
    backend = SideScriptedOcr(left=planted_left, right=planted_right)

    result = stage.run(default_config, roots, workers=1, backend=backend)

    assert backend.calls > 0  # text really was read
    assert package_logs.records, "the stage should log something, or this proves nothing"
    written = result.path.read_text(encoding="utf-8")
    logged = package_logs.text
    messages = " ".join(outcome.message for outcome in result.report.outcomes)

    for fragment in ("Sato", "Taro", "Kobayashi", "drsatotaro"):
        assert fragment not in written
        assert fragment not in logged
        assert fragment not in messages


@pytest.mark.slow
def test_parallel_workers_are_safe(roots: DataRoots, default_config: AppConfig, cohort: list[int]):
    """pyobjc's lazy attribute lookup raced here; guard the parallel path."""
    result = stage.run(
        default_config, roots, workers=3, backend=SideScriptedOcr("Dr Sato", "Guest")
    )
    assert result.report.ok
    assert len(result.frame) == len(cohort)


# ---------------------------------------------------------------------------
# the real backend, on a machine that has it
# ---------------------------------------------------------------------------
def _apple_vision_available() -> bool:
    return get_backend("apple_vision").available()


@pytest.mark.slow
@pytest.mark.skipif(not _apple_vision_available(), reason="the Vision framework is unavailable")
def test_real_ocr_reads_the_drawn_labels_and_catches_a_swapped_tile(
    roots: DataRoots, default_config: AppConfig, tmp_path: Path, ffmpeg_bin: str
):
    """End to end with the platform OCR engine, including a mismatch."""
    scratch = tmp_path / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    plan = [(28, WINTER_FOLDER, False), (3, "January 17 2026", False), (210, SUMMER_FOLDER, True)]
    for session_id, folder, swapped in plan:
        session = gen.alternating_session(
            session_id,
            n_turns=4,
            turn_s=1.5,
            gap_s=0.5,
            duration=12.0,
            width=960,
            height=540,
            swap_tiles=swapped,
        )
        gen.write_session_mp4(
            roots.data / folder / f"{session_id}.mp4",
            session,
            tmp_dir=scratch,
            ffmpeg=ffmpeg_bin,
        )

    result = stage.run(default_config, roots, workers=2)

    sides = dict(zip(result.frame["session_id"], result.frame["ocr_side"], strict=True))
    assert sides[28] == "left"
    assert sides[3] == "left"
    assert sides[210] == "right"

    flags = dict(zip(result.frame["session_id"], result.frame["flags"], strict=True))
    assert stage.FLAG_MISMATCH in flags[210]
    assert flags[28] == ""

    # OCR wins: the assumption does not overwrite what was found.
    decided = dict(zip(result.frame["session_id"], result.frame["decided_side"], strict=True))
    assert decided[210] == "right"


# ---------------------------------------------------------------------------
# the region diagnostic
#
# Label OCR found nothing in any of the 62 recordings and there was no way to
# see why. The cause was geometric: 180px letterbox bars top and bottom, so a
# label region expressed as a fraction of the whole frame landed in the bottom
# bar. This diagnostic is what made that visible.
# ---------------------------------------------------------------------------
def _observation(**overrides: Any) -> stage.RegionObservations:
    geometry = resolve_regions(load_config(DEFAULT), 1280, 720)[0]
    defaults: dict[str, Any] = {
        "session_id": 28,
        "frame_width": 1280,
        "frame_height": 720,
        "content_detected": True,
        "content_box": CropBox(x=0.0, y=0.25, width=1.0, height=0.5),
        "content_bars": (0, 180, 0, 180),
        "geometry": geometry,
        "upscale": 3.0,
        "n_frames_read": 3,
        "n_observations": 3,
        "n_above_confidence": 3,
        "n_usable_labels": 3,
        "max_confidence": 1.0,
        "n_ocr_errors": 0,
    }
    defaults.update(overrides)
    return stage.RegionObservations(**defaults)


def test_the_diagnostic_reports_both_coordinate_systems(default_config: AppConfig):
    lines = stage.debug_report([_observation()], default_config)
    text = "\n".join(lines)
    assert "x=0.0000" in text  # fractional
    assert "at (" in text  # pixels
    assert "640x" in text


def test_the_diagnostic_reports_the_detected_letterbox(default_config: AppConfig):
    text = "\n".join(stage.debug_report([_observation()], default_config))
    assert "bars l=0 t=180 r=0 b=180" in text
    assert "within this content area" in text


def test_the_diagnostic_says_when_no_letterbox_was_found(default_config: AppConfig):
    observation = _observation(content_detected=False, content_bars=(0, 0, 0, 0))
    text = "\n".join(stage.debug_report([observation], default_config))
    assert "none detected" in text


def test_the_diagnostic_reports_observation_counts_per_region(
    default_config: AppConfig,
):
    text = "\n".join(stage.debug_report([_observation()], default_config))
    assert "3 observation(s)" in text
    assert "3 above confidence" in text
    assert "3 usable label(s)" in text
    assert "max confidence 1.00" in text


def test_the_diagnostic_reports_the_settings_that_shaped_the_read(
    default_config: AppConfig,
):
    text = "\n".join(stage.debug_report([_observation()], default_config))
    assert "letterbox detection: auto" in text
    assert "label upscale: 3x" in text


def test_the_diagnostic_never_contains_recognised_text(default_config: AppConfig):
    """The diagnostic exists to be printed, so it must carry no names."""
    text = "\n".join(stage.debug_report([_observation()], default_config))
    for fragment in ("Sato", "sato", "Guest", "guest", DOCTOR):
        assert fragment not in text


def test_nothing_recognised_at_all_points_at_the_region(default_config: AppConfig):
    observation = _observation(n_observations=0, n_above_confidence=0, n_usable_labels=0)
    text = "\n".join(stage.debug_report([observation], default_config))
    assert "points at the region rather than the recogniser" in text
    assert "vc preview --label-regions" in text


def test_text_found_but_unusable_points_at_the_thresholds(default_config: AppConfig):
    observation = _observation(n_observations=4, n_above_confidence=0, n_usable_labels=0)
    text = "\n".join(stage.debug_report([observation], default_config))
    assert "min_confidence" in text or "upscale" in text


def test_ocr_errors_are_reported(default_config: AppConfig):
    text = "\n".join(stage.debug_report([_observation(n_ocr_errors=2)], default_config))
    assert "2 OCR error(s)" in text


def test_the_diagnostic_with_nothing_examined_says_so(default_config: AppConfig):
    text = "\n".join(stage.debug_report([], default_config))
    assert "OCR did not run" in text


# ---------------------------------------------------------------------------
# the machine-readable table
# ---------------------------------------------------------------------------
def test_the_debug_table_has_a_row_per_region_with_both_coordinate_systems():
    frame = stage.debug_frame([_observation(), _observation(session_id=3)])

    assert len(frame) == 2
    for column in ("tile_x", "tile_px_left", "label_x", "label_px_left", "content_x"):
        assert column in frame.columns
    assert list(frame["session_id"]) == [3, 28]


def test_the_debug_table_has_no_text_column():
    """By construction: there is nowhere for a name to be written."""
    frame = stage.debug_frame([_observation()])
    assert not any("text" in column or "label_value" in column for column in frame.columns)
    assert DOCTOR not in frame.to_csv(index=False)


def test_the_debug_table_records_the_bar_thickness():
    row = stage.debug_frame([_observation()]).iloc[0]
    assert row["bar_top"] == 180
    assert row["bar_bottom"] == 180
    assert bool(row["content_detected"])


def test_an_empty_debug_table_still_has_its_columns():
    frame = stage.debug_frame([])
    assert list(frame.columns) == list(stage.DEBUG_COLUMN_ORDER)
    assert frame.empty


# ---------------------------------------------------------------------------
# upscaling
# ---------------------------------------------------------------------------
def test_a_label_patch_is_enlarged_before_recognition():
    patch = np.zeros((20, 100, 3), dtype=np.uint8)
    assert stage.upscaled(patch, 3.0).shape[:2] == (60, 300)


def test_an_upscale_of_one_leaves_the_patch_alone():
    patch = np.zeros((20, 100, 3), dtype=np.uint8)
    assert stage.upscaled(patch, 1.0) is patch


def test_upscaling_an_empty_patch_is_safe():
    patch = np.zeros((0, 0, 3), dtype=np.uint8)
    assert stage.upscaled(patch, 3.0).size == 0


# ---------------------------------------------------------------------------
# the stage produces the diagnostic
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_the_stage_writes_the_debug_table(
    roots: DataRoots, default_config: AppConfig, cohort: list[int]
):
    result = stage.run(
        default_config, roots, workers=1, backend=SideScriptedOcr("Dr Sato", "Guest")
    )

    assert result.debug_path is not None
    assert result.debug_path.exists()
    # Two regions per session.
    assert len(result.debug) == 2 * len(cohort)
    assert result.observations


@pytest.mark.slow
def test_the_recorded_geometry_matches_what_was_read(
    roots: DataRoots, default_config: AppConfig, cohort: list[int]
):
    """The diagnostic is only useful if it reports the regions actually used."""
    result = stage.run(
        default_config, roots, workers=1, backend=SideScriptedOcr("Dr Sato", "Guest")
    )

    row = result.debug.iloc[0]
    assert row["n_frames_read"] > 0
    assert row["label_px_width"] > 0
    assert row["upscale"] == default_config.speakers.label_ocr.upscale


@pytest.mark.slow
def test_letterbox_detection_can_be_switched_off(roots: DataRoots, cohort: list[int]):
    config = load_config(DEFAULT, overrides={"video.letterbox_detection": "off"})
    result = stage.run(config, roots, workers=1, backend=SideScriptedOcr("Dr Sato", "Guest"))
    assert not result.debug["content_detected"].any()
