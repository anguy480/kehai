"""The exploratory LLM face pipeline, on synthetic frames and a fake model server."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import pytest

from vc_multimodal.config import AppConfig
from vc_multimodal.exploratory.llm_face import describe, rate, template
from vc_multimodal.io_utils import read_csv, write_parquet
from vc_multimodal.paths import DataRoots
from vc_multimodal.qc_notes import record as record_qc_note
from vc_multimodal.stages import face as face_stage
from vc_multimodal.stages import turns as turns_stage
from vc_multimodal.stages.aggregate import Timeline

STEP = 0.2
REASON = "Synthetic test note: camera confirmed pointed away for most of the recording."


def times(n: int, start: float = 0.0) -> np.ndarray:
    return start + STEP * np.arange(n, dtype=np.float64)


class TestUnitStats:
    def test_counts_episodes_and_their_length(self) -> None:
        values = np.zeros(100)
        values[10:15] = 0.6  # 5 frames, 1.0 s
        values[50:60] = 0.6  # 10 frames, 2.0 s
        stats = template.unit_stats("au12", times(100), values, STEP)
        assert stats.present_fraction == pytest.approx(0.15)
        assert stats.median_episode_s == pytest.approx(1.5)

    def test_a_gap_in_the_frames_ends_an_episode(self) -> None:
        t = np.concatenate([times(5), times(5, start=10.0)])
        stats = template.unit_stats("au12", t, np.full(10, 0.6), STEP)
        assert stats.median_episode_s == pytest.approx(1.0)

    def test_strength_is_the_excess_over_the_threshold(self) -> None:
        threshold = template.PRESENT_ABOVE["au12"]
        values = np.full(20, threshold + 0.5 * (1 - threshold))
        stats = template.unit_stats("au12", times(20), values, STEP)
        assert stats.strength == pytest.approx(0.5)

    def test_trend_compares_the_last_third_with_the_first(self) -> None:
        values = np.zeros(90)
        values[60:] = 0.6
        stats = template.unit_stats("au12", times(90), values, STEP)
        assert stats.trend == pytest.approx(1.0)

    def test_no_trend_from_too_few_frames(self) -> None:
        stats = template.unit_stats("au12", times(9), np.zeros(9), STEP)
        assert stats.trend is None

    def test_each_unit_has_its_own_threshold(self) -> None:
        resting_brow = np.full(20, 0.3)
        assert template.unit_stats("au01", times(20), resting_brow, STEP).present_fraction == 0
        assert template.unit_stats("au12", times(20), resting_brow, STEP).present_fraction == 1


class TestHeadStats:
    def test_a_still_head_has_no_sharp_movements(self) -> None:
        sd, per_min = template.head_stats(times(50), np.zeros(50), np.zeros(50), STEP, 10.0)
        assert sd == 0
        assert per_min == 0

    def test_a_jump_between_consecutive_frames_counts(self) -> None:
        pitch = np.zeros(50)
        pitch[25:] = 10.0
        _, per_min = template.head_stats(times(50), pitch, np.zeros(50), STEP, 60.0)
        assert per_min == pytest.approx(1.0)

    def test_a_jump_across_a_gap_does_not(self) -> None:
        t = np.concatenate([times(25), times(25, start=100.0)])
        pitch = np.concatenate([np.zeros(25), np.full(25, 10.0)])
        _, per_min = template.head_stats(t, pitch, np.zeros(50), STEP, 60.0)
        assert per_min == 0


class TestRendering:
    def stats(self, fraction: float) -> template.WindowStats:
        unit = template.UnitStats("au12", fraction, 0.5, 1.5, 0.0)
        return template.WindowStats("listening", 300.0, 0.9, (unit,), 4.0, 2.0)

    def test_a_window_reads_as_one_paragraph(self) -> None:
        text = template.render_window(self.stats(0.3))
        assert text.startswith("While listening (5.0 min of the face measured, 90% of that time):")
        assert "Lip corner pulling, as in smiling (AU12) appeared often (30% of the time)" in text
        assert "moved their head moderately" in text
        assert "\n" not in text

    def test_an_absent_unit_is_said_to_be_absent(self) -> None:
        assert "essentially never appeared" in template.render_window(self.stats(0.0))


def frames_table(session_id: int, n: int) -> pd.DataFrame:
    rng = np.random.default_rng(session_id)
    frame = pd.DataFrame(
        {
            "session_id": session_id,
            "frame_index": np.arange(n),
            "timestamp_s": times(n),
            "detected": True,
            "confidence": 0.9,
            "head_pitch": rng.normal(0, 3, n),
            "head_yaw": rng.normal(0, 3, n),
        }
    )
    for unit in ("au01", "au02", "au04", "au06", "au12"):
        frame[unit] = rng.uniform(0, 1, n)
    return frame


def timeline_table(seconds: float, *, speaking_until: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "state": ["speaking", "listening"],
            "start_s": [0.0, speaking_until],
            "end_s": [speaking_until, seconds],
        }
    )


class TestDescribeSession:
    def test_two_paragraphs_speaking_first(self, default_config: AppConfig) -> None:
        frames = frames_table(1, 3000)  # 600 s
        timeline = Timeline.from_frame(timeline_table(600.0, speaking_until=300.0))
        paragraphs = describe.describe_session(frames, timeline, default_config).split("\n\n")
        assert [p.split(" (")[0] for p in paragraphs] == ["While speaking", "While listening"]

    def test_too_little_measured_time_is_said_not_described(
        self, default_config: AppConfig
    ) -> None:
        frames = frames_table(1, 3000)
        timeline = Timeline.from_frame(timeline_table(600.0, speaking_until=10.0))
        text = describe.describe_session(frames, timeline, default_config)
        assert text.startswith("While speaking: too little of the face was measured")

    def test_the_same_frames_give_the_same_text(self, default_config: AppConfig) -> None:
        timeline = Timeline.from_frame(timeline_table(600.0, speaking_until=300.0))
        first = describe.describe_session(frames_table(1, 3000), timeline, default_config)
        again = describe.describe_session(frames_table(1, 3000), timeline, default_config)
        assert first == again


class TestDescribeRun:
    @pytest.fixture
    def sessions(self, roots: DataRoots, default_config: AppConfig) -> DataRoots:
        for session_id in (7, 43, 52):
            write_parquet(face_stage.face_path(roots, session_id), frames_table(session_id, 3000))
            write_parquet(
                turns_stage.timeline_path(roots, session_id),
                timeline_table(600.0, speaking_until=300.0),
            )
        notes = roots.work / default_config.qc.notes_path
        record_qc_note(notes, session_id=43, modality="face", status="unavailable", reason=REASON)
        record_qc_note(notes, session_id=52, modality="face", status="degraded", reason=REASON)
        return roots

    def test_unavailable_is_skipped_and_degraded_flagged(
        self, sessions: DataRoots, default_config: AppConfig
    ) -> None:
        result = describe.run(default_config, sessions)
        assert result.n_described == 2
        assert result.skipped == ((43, "face=unavailable"),)
        assert result.flagged == ((52, "face=degraded"),)
        table = read_csv(result.path).set_index("session_id")
        assert pd.isna(table.loc[43, "description"]) or table.loc[43, "description"] == ""
        assert table.loc[52, "qc_flag"] == "face=degraded"

    def test_the_summary_carries_no_description(
        self, sessions: DataRoots, default_config: AppConfig
    ) -> None:
        result = describe.run(default_config, sessions)
        summary = "\n".join(result.report_lines())
        assert "While speaking" not in summary
        assert "AU12" not in summary


class TestRatingSafety:
    @pytest.mark.parametrize(
        "url", ["https://api.openai.com/v1", "https://api.anthropic.com", "http://10.0.0.5:11434"]
    )
    def test_refuses_any_host_but_this_machine(self, url: str) -> None:
        with pytest.raises(rate.RatingError, match="refusing"):
            rate.assert_local(url)

    def test_accepts_loopback(self) -> None:
        rate.assert_local("http://127.0.0.1:11434")
        rate.assert_local("http://localhost:11434")

    def test_refuses_a_model_with_another_digest(self, monkeypatch: pytest.MonkeyPatch) -> None:
        tags = {"models": [{"name": rate.MODEL, "digest": "0" * 64}]}
        monkeypatch.setattr(rate, "_request", lambda url, payload=None: tags)
        with pytest.raises(rate.RatingError, match="not the pinned"):
            rate.check_model()


class TestParsing:
    def good(self) -> dict[str, int]:
        return dict.fromkeys(rate.SCALES, 4)

    def test_accepts_one_integer_per_scale(self) -> None:
        assert rate.parse_scores(json.dumps(self.good())) == self.good()

    @pytest.mark.parametrize(
        "change",
        [{"positive_affect": 8}, {"positive_affect": 0}, {"positive_affect": 3.5}, {"extra": 1}],
    )
    def test_rejects_anything_else(self, change: dict[str, Any]) -> None:
        with pytest.raises(rate.RatingError):
            rate.parse_scores(json.dumps({**self.good(), **change}))

    def test_rejects_non_json(self) -> None:
        with pytest.raises(rate.RatingError, match="JSON"):
            rate.parse_scores("I would rate this a 4.")

    def test_the_prompt_file_builds_two_messages(self) -> None:
        messages = rate.build_messages("A SYNTHETIC DESCRIPTION")
        assert [m["role"] for m in messages] == ["system", "user"]
        assert "A SYNTHETIC DESCRIPTION" in messages[1]["content"]
        assert "{description}" not in messages[1]["content"]
        assert all(scale in messages[1]["content"] for scale in rate.SCALES)


class TestRateRun:
    def test_every_described_session_is_rated_each_run(
        self, roots: DataRoots, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        folder = describe.output_dir(roots)
        pd.DataFrame(
            {
                "session_id": [7, 43, 52],
                "qc_flag": ["", "face=unavailable", "face=degraded"],
                "description": ["text a", "", "text b"],
            }
        ).to_csv(folder / describe.DESCRIPTIONS_FILE, index=False)
        monkeypatch.setattr(rate, "check_model", lambda host=rate.HOST: None)
        scores = dict.fromkeys(rate.SCALES, 5)
        monkeypatch.setattr(rate, "rate_once", lambda text, host=rate.HOST: (scores, "{}"))

        result = rate.run(roots, runs=3)
        table = read_csv(result.path)
        assert result.n_sessions == 2
        assert sorted(table["session_id"].unique()) == [7, 52]
        assert list(table.groupby("session_id").size()) == [3, 3]
        assert result.n_failed == 0
        meta = json.loads((folder / rate.META_FILE).read_text())
        assert meta["model_digest"] == rate.MODEL_DIGEST
        assert meta["scores_sha256"] == result.file_sha256
