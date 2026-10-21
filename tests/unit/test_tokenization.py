"""The ported token features, against analysis.py itself. Synthetic text only.

`original_token_features` below is analysis.py's per-file loop (Tanaka,
https://github.com/hirokitan1155/virtual_client_SRS2_analysis, ec07010, MIT),
copied as written with the sentence embeddings, agenda calls and word-list
files removed. It is the oracle the port must match: the port is rewritten for
readability, this is not. Its two keyword switches exist for variants (b) and
(c); at their defaults it is the loop as written.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from janome.tokenizer import Tokenizer
from lexicalrichness import LexicalRichness

from vc_multimodal.exploratory.tokenization import run as tokenization
from vc_multimodal.exploratory.tokenization.__main__ import EXIT_REPRODUCTION_FAILED, main
from vc_multimodal.exploratory.tokenization.token_features import (
    FEATURES,
    NORMALIZATION,
    OPPOSITE_FILES,
    ORIGINAL_MIN_LEN,
    STOPWORDS,
    VARIANT_C_EXCLUDED_POS2,
    doctor_speaker,
    transcript_tokens,
)
from vc_multimodal.paths import DataRoots

TOKENIZER = Tokenizer()

# Invented sentences, not from any recording.
DOCTOR = [
    "今日はどのようなことでお困りですか。",
    "それは大変でしたね。どんな時に不安を感じますか。",
    "家族の方とはよく話しますか。",
    "なるほど、仕事の人間関係が負担になっているんですね。",
]
PATIENT = [
    "最近仕事が忙しくて、あまり眠れていません。",
    "人と話すのが苦手だと思います。会議の前は特に緊張します。",
    "母とはよく話しますが、父とはあまり話しません。",
    "職場の同僚に相談したいけれど、迷惑をかけると思ってしまいます。",
]


def transcript(session_id: int) -> str:
    """A diarized transcript in the .txt layout analysis.py reads."""
    doctor = doctor_speaker(session_id)
    patient = 1 - doctor
    lines = []
    for d, p in zip(DOCTOR, PATIENT, strict=True):
        lines.append(f"Speaker {doctor}: {d}")
        lines.append(f"Speaker {patient}: {p}")
    return "\n".join(lines) + "\n"


def write_transcripts(folder: Path, session_ids: list[int]) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for sid in session_ids:
        # analysis.py reads utf-8-sig, so write the BOM it expects to skip.
        (folder / f"{sid}.txt").write_text(transcript(sid), encoding="utf-8-sig")
    return folder


def original_token_features(  # noqa: PLR0912, PLR0915
    file_path: Path,
    *,
    length_rule: bool = True,
    excluded_pos2: frozenset[str] = frozenset({"数"}),
) -> dict[str, float]:
    """analysis.py lines 197-393, minus embeddings, agenda and word-list files.

    Its structure is kept on purpose, so the lint rules it breaks are silenced
    here rather than the code reshaped.
    """
    t = TOKENIZER
    stopwords = set(STOPWORDS)
    normalization = dict(NORMALIZATION)
    opposite_files = set(OPPOSITE_FILES)

    ind_words_spe0 = []
    ind_words_spe1 = []
    word_count = 0
    word_count_doc = 0
    ind_words_patient = []
    ind_words_doctor = []
    patient_turn_count = 0
    doctor_turn_count = 0
    file_id = str(file_path.stem)
    if int(file_id) in opposite_files:  # noqa: SIM108 - as written
        doctor_id = 1
    else:
        doctor_id = 0

    with open(file_path, encoding="utf-8-sig") as f:  # noqa: PTH123 - as written
        for line in f:
            if "Speaker 0" in line or "Speaker 1" in line:
                speaker_id = int(line.split(":")[0].replace("Speaker ", ""))
                result = line.split(":", 1)[1]
                result_space = result.replace(" ", "").replace("　", "")
                if speaker_id == doctor_id:
                    doctor_turn_count += 1
                else:
                    patient_turn_count += 1
                for token in t.tokenize(result_space):
                    pos1, pos2 = token.part_of_speech.split(",")[:2]
                    if pos1 in {"名詞", "動詞", "形容詞"}:  # noqa: SIM102 - as written
                        if pos2 not in excluded_pos2:  # noqa: SIM102 - as written
                            if (
                                token.base_form not in stopwords
                                and token.surface not in stopwords
                                and (not length_rule or len(token.base_form) > 2)
                            ):
                                base = normalization.get(token.base_form, token.base_form)
                                if speaker_id == 0:
                                    ind_words_spe0.append(base)
                                else:
                                    ind_words_spe1.append(base)
                                if speaker_id != doctor_id:
                                    word_count += 1
                                    ind_words_patient.append(base)
                                else:
                                    word_count_doc += 1
                                    ind_words_doctor.append(base)

        counter0 = Counter(ind_words_spe0)
        counter1 = Counter(ind_words_spe1)
        my_set_0 = set(counter0)
        my_set_1 = set(counter1)
        union_op = my_set_1 | my_set_0
        inter_op = my_set_1 & my_set_0
        jaccard = len(inter_op) / len(union_op)
        vocab = sorted(set(counter0) | set(counter1))
        vec0 = np.array([counter0[w] for w in vocab])
        vec1 = np.array([counter1[w] for w in vocab])
        cosine = np.dot(vec0, vec1) / (np.linalg.norm(vec0) * np.linalg.norm(vec1))
        if patient_turn_count > 0:  # noqa: SIM108 - as written
            mean_words_per_turn = word_count / patient_turn_count
        else:
            mean_words_per_turn = np.nan
        patient_text = " ".join(ind_words_patient)
        doctor_text = " ".join(ind_words_doctor)
        try:
            mtld_patient = LexicalRichness(patient_text).mtld()
        except Exception:  # as written
            mtld_patient = np.nan
        try:
            mtld_doctor = LexicalRichness(doctor_text).mtld()
        except Exception:  # as written
            mtld_doctor = np.nan

    return {
        "Jaccard": jaccard,
        "Cosine": cosine,
        "Word_count": word_count,
        "Word_count_doc": word_count_doc,
        "Word_ratio": word_count / (word_count + word_count_doc),
        "MTLD_patient": mtld_patient,
        "MTLD_doctor": mtld_doctor,
        "Mean_words_per_turn": mean_words_per_turn,
    }


SESSIONS = [6, 7, 10, 52, 102]


@pytest.fixture
def transcripts(tmp_path: Path) -> Path:
    return write_transcripts(tmp_path / "diarizations_original", SESSIONS)


def nlp_features(transcripts: Path, path: Path, *, nudge: float = 0.0) -> Path:
    """An nlp_features.csv as analysis.py writes it, from the oracle."""
    rng = np.random.default_rng(3)
    files = sorted(transcripts.glob("*.txt"), key=lambda p: int(p.stem))
    rows = []
    for file_path in files:
        row = {column: float(rng.normal()) for column in tokenization.NLP_COLUMNS}
        row.update(original_token_features(file_path))
        rows.append(row)
    frame = pd.DataFrame(rows, columns=list(tokenization.NLP_COLUMNS))
    frame.loc[1, "Word_ratio"] += nudge
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def command(transcripts: Path, nlp: Path, out: Path) -> int:
    return main(
        [
            "features",
            "--transcripts",
            str(transcripts),
            "--nlp-features",
            str(nlp),
            "--out",
            str(out),
        ]
    )


# ---------------------------------------------------------------------------
# The port against the original
# ---------------------------------------------------------------------------
def test_the_default_rule_reproduces_analysis_py(transcripts: Path) -> None:
    for path in sorted(transcripts.glob("*.txt")):
        ported = transcript_tokens(path, TOKENIZER, ORIGINAL_MIN_LEN).features
        original = original_token_features(path)
        for name in FEATURES:
            assert ported[name] == pytest.approx(original[name], nan_ok=True), (path.name, name)


def test_variants_b_and_c_match_the_original_loop_with_the_same_switches(
    transcripts: Path,
) -> None:
    for path in sorted(transcripts.glob("*.txt")):
        b = transcript_tokens(path, TOKENIZER, None).features
        c = transcript_tokens(path, TOKENIZER, None, VARIANT_C_EXCLUDED_POS2).features
        want_b = original_token_features(path, length_rule=False)
        want_c = original_token_features(
            path, length_rule=False, excluded_pos2=VARIANT_C_EXCLUDED_POS2
        )
        for name in FEATURES:
            assert b[name] == pytest.approx(want_b[name], nan_ok=True), (path.name, name)
            assert c[name] == pytest.approx(want_c[name], nan_ok=True), (path.name, name)


def test_variant_c_drops_only_non_independent_tokens(transcripts: Path) -> None:
    """(c) is (b) less the 非自立 tokens, which the synthetic text must contain."""
    path = transcripts / "7.txt"
    b = transcript_tokens(path, TOKENIZER, None)
    c = transcript_tokens(path, TOKENIZER, None, VARIANT_C_EXCLUDED_POS2)
    dropped = b.lemmas - c.lemmas
    assert dropped, "no 非自立 token in the synthetic text"
    assert not c.lemmas - b.lemmas
    text = path.read_text(encoding="utf-8-sig")
    non_independent = {
        tok.base_form
        for tok in TOKENIZER.tokenize(text)
        if tok.part_of_speech.split(",")[1] == "非自立"
    }
    assert set(dropped) <= non_independent


def test_the_synthetic_text_exercises_the_rule(transcripts: Path) -> None:
    """Without short lemmas to drop, the test above would prove nothing."""
    path = transcripts / "7.txt"
    kept = transcript_tokens(path, TOKENIZER, ORIGINAL_MIN_LEN)
    everything = transcript_tokens(path, TOKENIZER, None)
    recovered = everything.lemmas - kept.lemmas
    assert recovered, "no lemma shorter than three characters in the synthetic text"
    assert all(len(lemma) < ORIGINAL_MIN_LEN for lemma in recovered)
    assert everything.n_lemmas > kept.n_lemmas


def test_removing_the_rule_changes_nothing_else(transcripts: Path) -> None:
    """Every lemma kept under the rule is kept, with the same count, without it."""
    path = transcripts / "6.txt"
    kept = transcript_tokens(path, TOKENIZER, ORIGINAL_MIN_LEN)
    everything = transcript_tokens(path, TOKENIZER, None)
    assert all(everything.lemmas[lemma] == n for lemma, n in kept.lemmas.items())


def test_the_doctor_is_speaker_one_in_the_opposite_files() -> None:
    assert doctor_speaker(6) == 1
    assert doctor_speaker(7) == 0
    assert len(OPPOSITE_FILES) == 27


def test_files_are_ordered_by_session_number_not_by_name(transcripts: Path) -> None:
    ordered = [int(path.stem) for path in tokenization.transcript_files(transcripts)]
    assert ordered == sorted(SESSIONS)


def test_a_transcript_not_named_by_session_is_refused(transcripts: Path) -> None:
    (transcripts / "notes.txt").write_text("x", encoding="utf-8")
    with pytest.raises(tokenization.TokenizationError, match="not named by session"):
        tokenization.transcript_files(transcripts)


# ---------------------------------------------------------------------------
# The reproduction gate
# ---------------------------------------------------------------------------
def test_a_reproduced_table_writes_every_variant(transcripts: Path, tmp_path: Path) -> None:
    nlp = nlp_features(transcripts, tmp_path / "work" / "nlp_features.csv")
    out = tmp_path / "out"
    result = tokenization.run(transcripts, nlp, out, TOKENIZER)

    original = pd.read_csv(nlp, encoding="utf-8-sig")
    new = pd.read_csv(out / tokenization.TEXT_NEW)
    assert list(new.columns) == ["session_id", *tokenization.NLP_COLUMNS]
    assert list(new["session_id"]) == sorted(SESSIONS)
    unchanged = [c for c in tokenization.NLP_COLUMNS if c not in FEATURES]
    assert len(unchanged) == 12
    pd.testing.assert_frame_equal(new[unchanged], original[unchanged])
    assert (new["Word_count"] >= original["Word_count"]).all()

    same = pd.read_csv(out / tokenization.TEXT_ORIGINAL)
    pd.testing.assert_frame_equal(
        same[list(tokenization.NLP_COLUMNS)], original, check_dtype=False, atol=1e-12
    )
    assert len(result.written) == 6
    c = pd.read_csv(out / tokenization.TEXT_NEW_C)
    pd.testing.assert_frame_equal(c[unchanged], original[unchanged])
    assert list(c["session_id"]) == sorted(SESSIONS)


def test_a_value_off_by_more_than_the_tolerance_stops_everything(
    transcripts: Path, tmp_path: Path
) -> None:
    nlp = nlp_features(transcripts, tmp_path / "work" / "nlp_features.csv", nudge=1e-6)
    out = tmp_path / "out"
    with pytest.raises(tokenization.ReproductionError) as caught:
        tokenization.run(transcripts, nlp, out, TOKENIZER)
    assert any(line.startswith("Word_ratio: 1 of 5") for line in caught.value.lines)
    assert not out.exists()


def test_no_variant_is_computed_when_the_gate_fails(
    transcripts: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nlp = nlp_features(transcripts, tmp_path / "work" / "nlp_features.csv", nudge=1e-6)
    computed: list[str] = []
    real = tokenization.compute

    def spy(files: list[Path], tokenizer: Tokenizer, variant: tokenization.Variant) -> object:
        computed.append(variant.key)
        return real(files, tokenizer, variant)

    monkeypatch.setattr(tokenization, "compute", spy)
    with pytest.raises(tokenization.ReproductionError):
        tokenization.run(transcripts, nlp, tmp_path / "out", TOKENIZER)
    assert computed == ["a"]


def test_a_difference_within_the_tolerance_passes(transcripts: Path, tmp_path: Path) -> None:
    nlp = nlp_features(transcripts, tmp_path / "work" / "nlp_features.csv", nudge=1e-12)
    tokenization.run(transcripts, nlp, tmp_path / "out", TOKENIZER)


def test_a_different_number_of_transcripts_is_refused(transcripts: Path, tmp_path: Path) -> None:
    nlp = nlp_features(transcripts, tmp_path / "work" / "nlp_features.csv")
    (transcripts / "200.txt").write_text(transcript(200), encoding="utf-8-sig")
    with pytest.raises(tokenization.TokenizationError, match="not the files"):
        tokenization.run(transcripts, nlp, tmp_path / "out", TOKENIZER)


def test_a_table_with_other_columns_is_refused(transcripts: Path, tmp_path: Path) -> None:
    path = tmp_path / "nlp_features.csv"
    pd.DataFrame({"Jaccard": [0.1]}).to_csv(path, index=False)
    with pytest.raises(tokenization.TokenizationError, match="20 columns"):
        tokenization.run(transcripts, path, tmp_path / "out", TOKENIZER)


# ---------------------------------------------------------------------------
# What is printed
# ---------------------------------------------------------------------------
def test_the_report_is_aggregate_only(transcripts: Path, tmp_path: Path) -> None:
    nlp = nlp_features(transcripts, tmp_path / "work" / "nlp_features.csv")
    result = tokenization.run(transcripts, nlp, tmp_path / "out", TOKENIZER)
    text = "\n".join(result.report_lines(tokenization.file_sha256))
    assert "reproduction check passed" in text
    assert "vocabulary (distinct)" in text
    assert "(c) against (b)" in text
    # No sentence from any transcript is printed, and no session is named.
    for sentence in [*DOCTOR, *PATIENT]:
        assert sentence not in text
    assert not any(f"session {sid}" in text for sid in SESSIONS)


@pytest.mark.usefixtures("roots")
def test_the_command_fails_with_its_own_exit_code(
    transcripts: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    nlp = nlp_features(transcripts, tmp_path / "work" / "nlp_features.csv", nudge=1.0)
    assert command(transcripts, nlp, tmp_path / "result") == EXIT_REPRODUCTION_FAILED
    assert "REPRODUCTION FAILED" in capsys.readouterr().err
    assert not (tmp_path / "result").exists()


def test_the_command_succeeds_and_prints_digests(
    transcripts: Path, tmp_path: Path, roots: DataRoots, capsys: pytest.CaptureFixture[str]
) -> None:
    nlp = nlp_features(transcripts, tmp_path / "work" / "nlp_features.csv")
    assert command(transcripts, nlp, tmp_path / "result") == 0
    assert capsys.readouterr().out.count("SHA-256") == 6
    assert (tmp_path / "result" / tokenization.TEXT_NEW).is_file()
    assert (tmp_path / "result" / tokenization.TEXT_NEW_C).is_file()
    assert roots.out.is_dir()
