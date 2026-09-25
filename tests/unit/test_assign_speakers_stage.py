"""Tests for the speaker assignment stage.

No model is downloaded and no real recording is used. A fake embedder returns
a vector chosen by the caller, so the tests exercise the plumbing - sample
building, the reference comparison, the map, the OCR bridge and the report -
rather than the acoustics, which are tested by the controls in the ADR.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vc_multimodal.config import AppConfig
from vc_multimodal.embeddings.base import EmbeddingError, SpeakerEmbedder, cosine_similarity
from vc_multimodal.features.assign_math import (
    AGREE,
    DISAGREE,
    FLAG_MOUTH_UNUSABLE,
    UNAVAILABLE,
    Decision,
)
from vc_multimodal.features.spans import Span
from vc_multimodal.paths import DataRoots
from vc_multimodal.runner import StageReport
from vc_multimodal.stages import assign_speakers as stage
from vc_multimodal.stages import verify_layout

SAMPLE_RATE = 16_000


class FakeEmbedder(SpeakerEmbedder):
    """Returns a vector looked up by the audio's first sample.

    Encoding the identity in the samples keeps the tests honest about what the
    stage does with audio: if it concatenated the wrong segments or embedded
    the wrong speaker, the vector changes.
    """

    name = "fake"

    def __init__(self, vectors: dict[float, np.ndarray] | None = None) -> None:
        self.vectors = vectors or {}
        self.calls: list[int] = []

    def available(self) -> bool:
        return True

    def unavailable_reason(self) -> str:
        return ""

    def version(self) -> str:
        return "fake:1"

    def embed(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        self.check_rate(sample_rate)
        if audio.size == 0:
            msg = "no audio to embed"
            raise EmbeddingError(msg)
        self.calls.append(audio.size)
        key = round(float(audio[0]), 4)
        return self.vectors.get(key, np.array([key, 1.0 - abs(key), 0.0], dtype=np.float64))


def write_wav(path: Path, samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes((np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes())
    return path


def constant_clip(value: float, seconds: float = 4.0) -> np.ndarray:
    return np.full(int(seconds * SAMPLE_RATE), value, dtype=np.float32)


@pytest.fixture
def references(roots: DataRoots) -> None:
    """Two reference clips, distinguishable by their constant value."""
    write_wav(stage.reference_dir(roots) / "psy_a.wav", constant_clip(0.5))
    write_wav(stage.reference_dir(roots) / "psy_b.wav", constant_clip(0.25))


class TestReferenceClips:
    def test_both_clips_are_embedded_with_a_within_clip_control(
        self, default_config: AppConfig, roots: DataRoots, references: None
    ) -> None:
        report = stage.load_references(default_config, roots, FakeEmbedder())
        assert [voice.clip_id for voice in report.voices] == ["psy_a", "psy_b"]
        # Halves of one clip are identical here, so the control is 1.0.
        assert report.within_clip["psy_a"] == pytest.approx(1.0)

    def test_the_clips_are_compared_against_each_other(
        self, default_config: AppConfig, roots: DataRoots, references: None
    ) -> None:
        report = stage.load_references(default_config, roots, FakeEmbedder())
        assert ("psy_a", "psy_b") in report.between_clips

    def test_a_missing_clip_says_which_one(
        self, default_config: AppConfig, roots: DataRoots
    ) -> None:
        write_wav(stage.reference_dir(roots) / "psy_a.wav", constant_clip(0.5))
        with pytest.raises(stage.AssignError, match=r"psy_b\.wav"):
            stage.load_references(default_config, roots, FakeEmbedder())

    def test_no_configured_clips_says_what_to_do(
        self, default_config: AppConfig, roots: DataRoots
    ) -> None:
        config = default_config.model_copy(
            update={"speakers": default_config.speakers.model_copy(update={"reference_clips": ()})}
        )
        with pytest.raises(stage.AssignError, match="reference_clips"):
            stage.load_references(config, roots, FakeEmbedder())

    def test_the_report_reads_the_comparison_against_the_control(
        self, default_config: AppConfig, roots: DataRoots, references: None
    ) -> None:
        report = stage.load_references(default_config, roots, FakeEmbedder())
        text = "\n".join(report.report_lines())
        assert "own two halves agree at" in text
        assert "psy_a vs psy_b" in text
        assert "one person recorded twice" in text or "two people" in text


class TestThePsychiatristMap:
    def test_a_clip_filename_is_accepted(self, default_config: AppConfig, roots: DataRoots) -> None:
        pd.DataFrame({"session_id": [1, 2], "reference_clip": ["psy_a.wav", "psy_b.wav"]}).to_csv(
            roots.work / "session_psychiatrist_map.csv", index=False
        )
        assert stage.load_psychiatrist_map(default_config, roots) == {1: "psy_a", 2: "psy_b"}

    def test_a_clip_id_is_accepted(self, default_config: AppConfig, roots: DataRoots) -> None:
        pd.DataFrame({"session_id": [7], "psychiatrist_id": ["psy_b"]}).to_csv(
            roots.work / "session_psychiatrist_map.csv", index=False
        )
        assert stage.load_psychiatrist_map(default_config, roots) == {7: "psy_b"}

    def test_an_unconfigured_clip_is_an_error(
        self, default_config: AppConfig, roots: DataRoots
    ) -> None:
        pd.DataFrame({"session_id": [1], "reference_clip": ["psy_z.wav"]}).to_csv(
            roots.work / "session_psychiatrist_map.csv", index=False
        )
        with pytest.raises(stage.AssignError, match="psy_z"):
            stage.load_psychiatrist_map(default_config, roots)

    def test_a_missing_clip_column_says_what_is_needed(
        self, default_config: AppConfig, roots: DataRoots
    ) -> None:
        pd.DataFrame({"session_id": [1], "something": ["x"]}).to_csv(
            roots.work / "session_psychiatrist_map.csv", index=False
        )
        with pytest.raises(stage.AssignError, match="reference_clip"):
            stage.load_psychiatrist_map(default_config, roots)

    def test_an_absent_map_is_not_an_error(
        self, default_config: AppConfig, roots: DataRoots
    ) -> None:
        # Every session is then decided by its best match across all clips.
        assert stage.load_psychiatrist_map(default_config, roots) == {}


class TestSampleBuilding:
    def config(self, default_config: AppConfig, **updates: object) -> AppConfig:
        embedding = default_config.speakers.embedding.model_copy(update=updates)
        speakers = default_config.speakers.model_copy(update={"embedding": embedding})
        return default_config.model_copy(update={"speakers": speakers})

    def test_short_segments_are_left_out(self, default_config: AppConfig) -> None:
        # The diarizer partitions time, so its briefest segments are the ones
        # most likely to hold the other person's voice.
        samples = np.arange(20 * SAMPLE_RATE, dtype=np.float32) / (20 * SAMPLE_RATE)
        by_speaker = {"A": [Span(0.0, 0.5), Span(2.0, 8.0)]}
        built = stage.build_samples(samples, SAMPLE_RATE, by_speaker, default_config)
        assert built[0].n_segments == 1
        assert built[0].embedded_s == pytest.approx(6.0, abs=0.01)

    def test_the_cap_limits_how_much_is_embedded(self, default_config: AppConfig) -> None:
        config = self.config(default_config, max_seconds_per_speaker=3.0)
        samples = np.zeros(30 * SAMPLE_RATE, dtype=np.float32)
        by_speaker = {"A": [Span(0.0, 10.0), Span(11.0, 20.0)]}
        built = stage.build_samples(samples, SAMPLE_RATE, by_speaker, config)
        assert built[0].embedded_s == pytest.approx(3.0, abs=0.01)

    def test_the_longest_segments_are_taken_first(self, default_config: AppConfig) -> None:
        # Marked audio: the long segment holds 0.9, the short one 0.1.
        samples = np.zeros(30 * SAMPLE_RATE, dtype=np.float32)
        samples[: 2 * SAMPLE_RATE] = 0.1
        samples[10 * SAMPLE_RATE : 20 * SAMPLE_RATE] = 0.9
        config = self.config(default_config, max_seconds_per_speaker=5.0)
        by_speaker = {"A": [Span(0.0, 2.0), Span(10.0, 20.0)]}
        built = stage.build_samples(samples, SAMPLE_RATE, by_speaker, config)
        assert built[0].audio[0] == pytest.approx(0.9, abs=0.001)

    def test_speech_seconds_count_all_speech_not_just_what_was_embedded(
        self, default_config: AppConfig
    ) -> None:
        config = self.config(default_config, max_seconds_per_speaker=2.0)
        samples = np.zeros(30 * SAMPLE_RATE, dtype=np.float32)
        by_speaker = {"A": [Span(0.0, 10.0)]}
        built = stage.build_samples(samples, SAMPLE_RATE, by_speaker, config)
        assert built[0].speech_s == pytest.approx(10.0)
        assert built[0].embedded_s == pytest.approx(2.0, abs=0.01)

    def test_a_speaker_with_only_short_segments_yields_no_audio(
        self, default_config: AppConfig
    ) -> None:
        samples = np.zeros(10 * SAMPLE_RATE, dtype=np.float32)
        built = stage.build_samples(
            samples, SAMPLE_RATE, {"A": [Span(0.0, 0.2), Span(1.0, 1.3)]}, default_config
        )
        assert built[0].audio.size == 0
        assert built[0].embedded_s == 0.0


class TestScoring:
    def test_a_speaker_is_scored_against_every_clip(
        self, default_config: AppConfig, roots: DataRoots, references: None
    ) -> None:
        embedder = FakeEmbedder()
        report = stage.load_references(default_config, roots, embedder)
        samples = np.full(10 * SAMPLE_RATE, 0.5, dtype=np.float32)
        built = stage.build_samples(samples, SAMPLE_RATE, {"A": [Span(0.0, 8.0)]}, default_config)
        scored = stage.score_speakers(built, report, embedder)
        assert set(scored[0].similarities) == {"psy_a", "psy_b"}
        # A holds the same constant as psy_a, so it matches that clip exactly.
        assert scored[0].similarities["psy_a"] == pytest.approx(1.0)

    def test_an_unembeddable_speaker_scores_nothing(
        self, default_config: AppConfig, roots: DataRoots, references: None
    ) -> None:
        embedder = FakeEmbedder()
        report = stage.load_references(default_config, roots, embedder)
        built = stage.build_samples(
            np.zeros(SAMPLE_RATE, dtype=np.float32),
            SAMPLE_RATE,
            {"A": [Span(0.0, 0.1)]},
            default_config,
        )
        scored = stage.score_speakers(built, report, embedder)
        assert scored[0].similarities == {}
        assert scored[0].embedded_s == 0.0


class TestMouthCorrelation:
    def test_a_constant_mouth_series_is_no_evidence(self) -> None:
        # A tile where the face was never found gives a constant series; that
        # is an absence of evidence, not a correlation of zero.
        assert stage._correlate(np.ones(10), np.array([0.0, 1.0] * 5)) is None

    def test_a_constant_speech_series_is_no_evidence(self) -> None:
        assert stage._correlate(np.array([0.0, 1.0] * 5), np.ones(10)) is None

    def test_a_matching_series_correlates(self) -> None:
        speech = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 1.0])
        mouth = np.array([0.1, 0.1, 0.8, 0.9, 0.1, 0.7])
        value = stage._correlate(mouth, speech)
        assert value is not None
        assert value > 0.9

    def test_speech_is_sampled_at_the_frame_times(self) -> None:
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        indicator = stage._speech_indicator([Span(1.5, 3.5)], times)
        assert list(indicator) == [0.0, 0.0, 1.0, 1.0, 0.0]

    def test_a_tile_is_chosen_only_when_clearly_ahead(self) -> None:
        assert stage._choose_tile({"l": 0.8, "r": 0.1}, 0.15, 0.10) == "l"
        # Equal correlation with both faces says nothing about which is which.
        assert stage._choose_tile({"l": 0.8, "r": 0.78}, 0.15, 0.10) is None
        # A positive but weak correlation is not evidence either.
        assert stage._choose_tile({"l": 0.05, "r": 0.01}, 0.15, 0.10) is None
        assert stage._choose_tile({}, 0.15, 0.10) is None


class TestTheOcrBridge:
    def decision(self, psychiatrist: str = "A") -> Decision:
        return Decision(
            by_speaker={"A": "psychiatrist", "B": "participant"},
            psychiatrist=psychiatrist,
            participant="B",
            margin=0.5,
            decisive_clips=("psy_a",),
            clip_choices={"psy_a": psychiatrist},
        )

    def mouth(self, **by_speaker: str | None) -> stage.MouthEvidence:
        return stage.MouthEvidence(
            by_speaker=by_speaker,
            correlations={},
            frames_measured={},
            usable=any(v is not None for v in by_speaker.values()),
        )

    def tiles(self, default_config: AppConfig) -> list[str]:
        return [name for name, _ in default_config.video.tiles_left_to_right()]

    def test_ocr_agreement_needs_the_mouth_link(self, default_config: AppConfig) -> None:
        # Without it, OCR speaks about sides and diarization about voices, and
        # the two are simply not comparable.
        checks = stage._ocr_cross_check(
            self.decision(), self.mouth(), config=default_config, ocr_side="left"
        )
        assert checks.ocr_agreement == UNAVAILABLE
        assert checks.mouth_agreement == UNAVAILABLE
        assert FLAG_MOUTH_UNUSABLE in checks.flags

    def test_agreement_when_the_bridged_evidence_matches(self, default_config: AppConfig) -> None:
        left, right = self.tiles(default_config)
        checks = stage._ocr_cross_check(
            self.decision("A"),
            self.mouth(**{"A": left, "B": right}),
            config=default_config,
            ocr_side="left",
        )
        assert checks.ocr_agreement == AGREE
        assert checks.mouth_agreement == AGREE
        assert checks.side_source == "ocr"
        assert not checks.flags

    def test_disagreement_is_flagged_and_the_embedding_stands(
        self, default_config: AppConfig
    ) -> None:
        left, right = self.tiles(default_config)
        decision = self.decision("A")
        checks = stage._ocr_cross_check(
            decision,
            # The mouth evidence puts A on the right, so OCR's "left" implies B.
            self.mouth(**{"A": right, "B": left}),
            config=default_config,
            ocr_side="left",
        )
        assert checks.ocr_agreement == DISAGREE
        assert checks.independent_choice == "B"
        # The decision is untouched: a cross-check is recorded, not consulted.
        assert decision.psychiatrist == "A"

    def test_without_ocr_the_assumed_side_is_used_and_labelled(
        self, default_config: AppConfig
    ) -> None:
        left, right = self.tiles(default_config)
        checks = stage._ocr_cross_check(
            self.decision("A"),
            self.mouth(**{"A": left, "B": right}),
            config=default_config,
            ocr_side=None,
        )
        assert checks.side_source == "assumed"
        assert checks.mouth_agreement == AGREE
        # OCR itself said nothing, so it corroborates nothing.
        assert checks.ocr_agreement == UNAVAILABLE


class TestLayoutInput:
    def write_layout(self, roots: DataRoots, sides: list[str]) -> None:
        """Written through the real column order, not a guess at it.

        An earlier version of this test invented the column name, which let the
        reader look for a column nothing writes: the OCR cross-check then
        reported `unavailable` for every session and looked like a limitation of
        the data rather than a bug.
        """
        rows = [
            dict.fromkeys(verify_layout.COLUMN_ORDER)
            | {"session_id": index + 1, verify_layout.OCR_SIDE_COLUMN: side}
            for index, side in enumerate(sides)
        ]
        verify_layout.build_frame(rows).to_csv(
            roots.out / verify_layout.LAYOUT_FILENAME, index=False
        )

    def test_settled_sides_are_read(self, roots: DataRoots) -> None:
        self.write_layout(roots, ["left", "right", "inconclusive"])
        assert stage.load_layout(roots) == {1: "left", 2: "right"}

    def test_the_column_the_reader_wants_is_the_one_written(self) -> None:
        assert verify_layout.OCR_SIDE_COLUMN in verify_layout.COLUMN_ORDER

    def test_the_decided_side_is_not_used_as_corroboration(self, roots: DataRoots) -> None:
        # decided_side falls back to the configured assumption, so reading it
        # would let an assumption pose as evidence from the labels.
        rows = [
            dict.fromkeys(verify_layout.COLUMN_ORDER) | {"session_id": 1, "decided_side": "left"}
        ]
        verify_layout.build_frame(rows).to_csv(
            roots.out / verify_layout.LAYOUT_FILENAME, index=False
        )
        assert stage.load_layout(roots) == {}

    def test_a_missing_table_means_no_opinion(self, roots: DataRoots) -> None:
        assert stage.load_layout(roots) == {}


class TestTheReport:
    def frame(self, **overrides: object) -> pd.DataFrame:
        base = {
            "session_id": [1, 2, 3],
            "wave": ["winter"] * 3,
            "n_speakers": [2, 2, 2],
            "assigned": [True, True, True],
            "psychiatrist_speaker": ["SPEAKER_00"] * 3,
            "participant_speaker": ["SPEAKER_01"] * 3,
            "mapped_clip": ["psy_a", "psy_a", ""],
            "decisive_clips": ["psy_a", "psy_a", "psy_a;psy_b"],
            "psychiatrist_similarity": [0.8, 0.7, 0.9],
            "participant_similarity": [0.2, 0.6, 0.3],
            "best_clip": ["psy_a", "psy_a", "psy_b"],
            "margin": [0.6, 0.05, 0.6],
            "clip_choices": ["psy_a=SPEAKER_00"] * 3,
            "clips_agree": [True, True, True],
            "psychiatrist_speech_s": [200.0] * 3,
            "participant_speech_s": [180.0] * 3,
            "ocr_side": ["left", "left", ""],
            "side_source": ["ocr", "ocr", "assumed"],
            "ocr_agreement": [AGREE, DISAGREE, UNAVAILABLE],
            "mouth_agreement": [AGREE, DISAGREE, UNAVAILABLE],
            "mouth_tiles": ["SPEAKER_00=left_tile"] * 3,
            "mouth_best_correlation": [0.5, 0.4, None],
            "mouth_reason": ["", "", "disabled"],
            "qc__flags": ["", "speakers_ocr_disagrees", ""],
        }
        base.update(overrides)
        return pd.DataFrame(base)

    def result(self, roots: DataRoots, frame: pd.DataFrame) -> stage.AssignResult:
        return stage.AssignResult(
            report=StageReport(stage=stage.STAGE, outcomes=(), seconds=0.0, notes=()),
            frame=frame,
            scores=pd.DataFrame(),
            path=roots.out / "speakers.csv",
            scores_path=roots.out / "speaker_scores.csv",
            references=stage.ReferenceReport(voices=(), within_clip={}, between_clips={}),
        )

    def test_the_agreement_table_names_the_disagreeing_sessions(
        self, default_config: AppConfig, roots: DataRoots
    ) -> None:
        lines = stage.summarise(self.result(roots, self.frame()), default_config)
        text = "\n".join(lines)
        assert "disagree" in text
        assert "[2]" in text

    def test_the_margin_range_is_reported(
        self, default_config: AppConfig, roots: DataRoots
    ) -> None:
        text = "\n".join(stage.summarise(self.result(roots, self.frame()), default_config))
        assert "margins:" in text
        assert "+0.050" in text

    def test_a_low_margin_session_is_listed(
        self, default_config: AppConfig, roots: DataRoots
    ) -> None:
        text = "\n".join(stage.summarise(self.result(roots, self.frame()), default_config))
        assert "below the +0.10 threshold" in text

    def test_the_unmapped_sessions_are_reported_separately(
        self, default_config: AppConfig, roots: DataRoots
    ) -> None:
        text = "\n".join(stage.summarise(self.result(roots, self.frame()), default_config))
        assert "no mapped reference clip" in text
        assert "best match psy_b" in text

    def test_unavailable_agreement_is_explained_not_glossed(
        self, default_config: AppConfig, roots: DataRoots
    ) -> None:
        text = "\n".join(stage.summarise(self.result(roots, self.frame()), default_config))
        assert "only corroborate an assignment through the mouth link" in text

    def test_flags_are_counted_with_their_sessions(
        self, default_config: AppConfig, roots: DataRoots
    ) -> None:
        text = "\n".join(stage.summarise(self.result(roots, self.frame()), default_config))
        assert "speakers_ocr_disagrees: 1 session(s) [2]" in text

    def test_unassigned_sessions_are_named(
        self, default_config: AppConfig, roots: DataRoots
    ) -> None:
        frame = self.frame(assigned=[True, False, True])
        text = "\n".join(stage.summarise(self.result(roots, frame), default_config))
        assert "NOT assigned: 1 session(s): [2]" in text

    def test_clip_disagreement_is_reported(
        self, default_config: AppConfig, roots: DataRoots
    ) -> None:
        frame = self.frame(clips_agree=[True, False, True])
        text = "\n".join(stage.summarise(self.result(roots, frame), default_config))
        assert "clips disagree, flagged: [2]" in text


class TestCosineSimilarity:
    def test_identical_vectors_score_one(self) -> None:
        v = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_a_zero_vector_is_no_evidence_rather_than_an_error(self) -> None:
        # Silence can produce one, and 0.0 is the honest reading.
        assert cosine_similarity(np.zeros(3), np.array([1.0, 0.0, 0.0])) == 0.0

    def test_opposite_vectors_score_minus_one(self) -> None:
        v = np.array([1.0, 0.0])
        assert cosine_similarity(v, -v) == pytest.approx(-1.0)
