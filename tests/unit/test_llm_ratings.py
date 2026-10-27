"""The LLM rater, on invented transcripts and a scripted backend. No call leaves the test."""

from __future__ import annotations

import email.message
import io
import json
import shutil
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from vc_multimodal.exploratory.llm_ratings import __main__ as cli
from vc_multimodal.exploratory.llm_ratings import plan as plan_module
from vc_multimodal.exploratory.llm_ratings import rater
from vc_multimodal.exploratory.llm_ratings.backend import (
    BackendUnavailableError,
    ChatCompletionsBackend,
    Completion,
    RequestRejectedError,
)
from vc_multimodal.exploratory.llm_ratings.dialogue import (
    TranscriptError,
    dialogue_lines,
    session_dialogue,
    transcript_files,
)
from vc_multimodal.exploratory.llm_ratings.plan import (
    FEATURES,
    ITEMS,
    BackendNotSetError,
    Endpoint,
    PlanError,
    Prompt,
    RequestSettings,
    load_plan,
    load_prompt,
    validate_response,
)

SESSIONS = [1, 3, 6, 7, 8]
INFO = {"backend": "local", "model": "fake", "prompt_sha256": "0" * 64, "commit": "c" * 40}


def ratings(value: int = 2) -> dict[str, int]:
    """A valid rating, every item at `value` clipped to its range."""
    return {item.key: min(max(value, item.low), item.high) for item in ITEMS}


class ScriptedBackend:
    """Answers from a script per session, recognised by a marker in the dialogue."""

    def __init__(self, script: dict[int, list[Any]] | None = None, default: Any = None) -> None:
        self.script = {session: list(answers) for session, answers in (script or {}).items()}
        self.default = default if default is not None else json.dumps(ratings())
        self.calls: list[int] = []

    @property
    def model(self) -> str:
        return "fake"

    def complete(self, system: str, user: str) -> Completion:
        session = int(user.split("セッション")[1].split("。", maxsplit=1)[0])
        self.calls.append(session)
        answers = self.script.get(session)
        answer = answers.pop(0) if answers else self.default
        if isinstance(answer, Exception):
            raise answer
        return Completion(answer, 1000, 40)


def write_transcripts(folder: Path, sessions: list[int] = SESSIONS) -> list[Path]:
    """Invented transcripts; each names its session so the scripted backend can tell."""
    folder.mkdir(parents=True, exist_ok=True)
    for session in sessions:
        (folder / f"{session}.txt").write_text(
            f"Speaker 0: セッション{session}。こんにちは\nSpeaker 1: はい\n", encoding="utf-8"
        )
    return transcript_files(folder)


@pytest.fixture
def prompt() -> Prompt:
    return Prompt(system="system", user_template="評価:\n{{DIALOGUE}}", sha256="0" * 64)


@pytest.fixture
def files(tmp_path: Path) -> list[Path]:
    return write_transcripts(tmp_path / "transcripts")


def run(
    files: list[Path], prompt: Prompt, backend: ScriptedBackend, tmp: Path, *, pilot: bool
) -> rater.RaterResult:
    cache = rater.Cache(tmp / "cache", "0" * 64, backend.model)
    return rater.run(
        files,
        prompt,
        backend,
        cache,
        tmp / "out",
        pilot=pilot,
        run_info=INFO,
        expected_sessions=len(SESSIONS),
    )


# ---------------------------------------------------------------------------
# The dialogue
# ---------------------------------------------------------------------------
def test_the_dialogue_marks_roles_and_strips_spaces_as_analysis_py_does() -> None:
    lines = ["Speaker 0: こんにちは 今日は\n", "雑音\n", "Speaker 1: はい　元気です\r\n"]
    assert dialogue_lines(lines, 3) == ["医師: こんにちは今日は", "参加者: はい元気です"]


def test_in_an_opposite_session_speaker_1_is_the_doctor() -> None:
    lines = ["Speaker 0: はい\n", "Speaker 1: どうですか\n"]
    assert dialogue_lines(lines, 6) == ["参加者: はい", "医師: どうですか"]


def test_the_dialogue_reads_a_bom_and_carries_nothing_but_the_lines(tmp_path: Path) -> None:
    path = tmp_path / "12.txt"
    path.write_text("Speaker 1: 始めます\nSpeaker 0: はい\n", encoding="utf-8-sig")
    assert session_dialogue(path) == "医師: 始めます\n参加者: はい"


def test_transcripts_sort_by_number_and_odd_names_are_refused(tmp_path: Path) -> None:
    assert [p.stem for p in write_transcripts(tmp_path / "a", [10, 2, 1])] == ["1", "2", "10"]
    (tmp_path / "a" / "notes.txt").write_text("x", encoding="utf-8")
    with pytest.raises(TranscriptError, match="not named by session number"):
        transcript_files(tmp_path / "a")


def test_a_line_without_a_speaker_number_is_refused_without_its_text() -> None:
    with pytest.raises(TranscriptError, match="line 1") as raised:
        dialogue_lines(["x Speaker 0: 秘密の内容\n"], 3)
    assert "秘密" not in str(raised.value)


# ---------------------------------------------------------------------------
# The plan and the prompt
# ---------------------------------------------------------------------------
def test_the_shipped_prompt_is_the_preregistered_one() -> None:
    plan = load_plan()
    prompt = load_prompt(plan.prompt_path, plan.prompt_sha256)
    prereg = (plan_module.REPO_ROOT / "docs" / "prereg" / "llm-ratings.md").read_text("utf-8")
    assert prompt.sha256 in prereg
    system, user = prompt.render("医師: こんにちは")
    assert "{{DIALOGUE}}" not in system + user
    assert user.count("医師: こんにちは") == 1
    for item in ITEMS:
        assert item.key in user


def test_a_changed_prompt_is_refused(tmp_path: Path) -> None:
    plan = load_plan()
    changed = tmp_path / "prompt.txt"
    shutil.copy(plan.prompt_path, changed)
    changed.write_text(changed.read_text("utf-8") + " ", encoding="utf-8")
    with pytest.raises(PlanError, match="pre-registered"):
        load_prompt(changed, plan.prompt_sha256)


def plan_file(tmp_path: Path, backend: str | None) -> Path:
    """The shipped plan with the backend set as given and an invented local endpoint."""
    raw = yaml.safe_load(plan_module.PLAN_PATH.read_text(encoding="utf-8"))
    raw["backend"] = backend
    raw["backends"]["local"].update(base_url="http://127.0.0.1:1/v1", model="fake")
    path = tmp_path / "llm_ratings.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    return path


def test_with_no_backend_set_no_endpoint_exists(tmp_path: Path) -> None:
    with pytest.raises(BackendNotSetError, match="no call is made"):
        load_plan(plan_file(tmp_path, None)).endpoint()
    assert load_plan(plan_file(tmp_path, "local")).endpoint().model == "fake"


def test_with_no_backend_set_the_command_stops_before_any_transcript(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = plan_file(tmp_path, None)
    monkeypatch.setattr(cli, "load_plan", lambda: load_plan(path))

    def no_transcripts(_: Path) -> list[Path]:
        raise AssertionError("transcripts were read")

    monkeypatch.setattr(cli, "transcript_files", no_transcripts)
    assert cli.main(["pilot", "--transcripts", str(tmp_path / "none")]) == cli.EXIT_SETUP_ERROR
    assert "no backend is set" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# A response
# ---------------------------------------------------------------------------
def test_a_valid_response_gives_every_item() -> None:
    checked = validate_response("  " + json.dumps(ratings(3)) + "\n")
    assert checked.ratings == ratings(3)


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        (None, plan_module.EMPTY),
        ("```json\n{}\n```", plan_module.NOT_JSON),
        ("[1, 2]", plan_module.NOT_OBJECT),
        (json.dumps({k: v for k, v in ratings().items() if k != "s_rrb"}), plan_module.WRONG_KEYS),
        (json.dumps({**ratings(), "rationale": 1}), plan_module.WRONG_KEYS),
        (json.dumps(ratings())[:-1] + ', "s_rrb": 2}', plan_module.WRONG_KEYS),
        (json.dumps({**ratings(), "k_effort": True}), plan_module.NOT_INTEGER),
        (json.dumps({**ratings(), "k_effort": 2.0}), plan_module.NOT_INTEGER),
        (json.dumps({**ratings(), "k_effort": None}), plan_module.NOT_INTEGER),
        (json.dumps({**ratings(), "k_effort": 5}), plan_module.OUT_OF_RANGE),
        (json.dumps({**ratings(), "s_rrb": 0}), plan_module.OUT_OF_RANGE),
    ],
)
def test_a_malformed_response_is_named_and_gives_nothing(text: str | None, problem: str) -> None:
    checked = validate_response(text)
    assert checked.ratings is None
    assert checked.problem == problem


# ---------------------------------------------------------------------------
# Calls, retries, sessions
# ---------------------------------------------------------------------------
def test_a_malformed_response_is_retried_until_valid() -> None:
    backend = ScriptedBackend({1: ["oops", "{}", json.dumps(ratings(1))]})
    record, responses = rater.rate_call(backend, "s", "セッション1。", 1, 1)
    assert record.status == rater.VALID
    assert record.attempts == 3
    assert record.problems == (plan_module.NOT_JSON, plan_module.WRONG_KEYS)
    assert len(responses) == 3


def test_a_call_fails_after_three_retries() -> None:
    backend = ScriptedBackend(default="oops")
    record, _ = rater.rate_call(backend, "s", "セッション1。", 1, 1)
    assert record.status == rater.MALFORMED
    assert record.attempts == 1 + plan_module.MAX_RETRIES
    assert record.ratings is None


def test_a_failed_call_fails_the_session_and_nothing_is_filled(
    files: list[Path], prompt: Prompt, tmp_path: Path
) -> None:
    backend = ScriptedBackend({6: ["oops"] * 4})
    result = run(files, prompt, backend, tmp_path, pilot=False)
    failed = next(s for s in result.sessions if s.session_id == 6)
    assert not failed.rated
    assert len(failed.calls) == 1, "the other calls of a failed session are not made"
    table = pd.read_csv(tmp_path / "out" / rater.FEATURES_FILE)
    assert table.loc[table["session_id"] == 6, list(FEATURES)].isna().all(axis=None)
    assert table.loc[table["session_id"] != 6, list(FEATURES)].notna().all(axis=None)
    assert result.counts()["sessions_failed"] == [6]


def test_features_are_the_mean_of_the_three_calls(
    files: list[Path], prompt: Prompt, tmp_path: Path
) -> None:
    answers = [json.dumps(ratings(v)) for v in (1, 2, 4)]
    run(files, prompt, ScriptedBackend({3: answers}), tmp_path, pilot=False)
    table = pd.read_csv(tmp_path / "out" / rater.FEATURES_FILE).set_index("session_id")
    assert list(table.index) == SESSIONS
    assert table.loc[3, "llm__k_nervous"] == pytest.approx((1 + 2 + 4) / 3)
    assert table.loc[3, "llm__s_rrb"] == pytest.approx((1 + 2 + 4) / 3)


# ---------------------------------------------------------------------------
# The cache
# ---------------------------------------------------------------------------
def test_the_full_run_reuses_the_pilot_and_a_rerun_makes_no_call(
    files: list[Path], prompt: Prompt, tmp_path: Path
) -> None:
    backend = ScriptedBackend()
    pilot = run(files, prompt, backend, tmp_path, pilot=True)
    assert [s.session_id for s in pilot.sessions] == SESSIONS[:3]
    assert pilot.written == (tmp_path / "out" / rater.PILOT_FILE,)
    assert len(backend.calls) == 9

    full = run(files, prompt, backend, tmp_path, pilot=False)
    assert full.counts()["calls_from_cache"] == 9
    assert full.counts()["calls_made_now"] == 6

    again = ScriptedBackend()
    run(files, prompt, again, tmp_path, pilot=False)
    assert again.calls == []


def test_a_malformed_call_is_cached_but_a_rejected_one_is_not(
    files: list[Path], prompt: Prompt, tmp_path: Path
) -> None:
    rejected = RequestRejectedError("too long")
    run(files, prompt, ScriptedBackend({6: ["oops"] * 4, 7: [rejected]}), tmp_path, pilot=False)
    rerun = ScriptedBackend()
    result = run(files, prompt, rerun, tmp_path, pilot=False)
    assert rerun.calls == [7, 7, 7], "only the rejected session is asked again"
    assert result.counts()["sessions_failed"] == [6]


def test_the_cache_is_keyed_by_model_and_prompt_and_refuses_a_misfiled_entry(
    tmp_path: Path,
) -> None:
    record = rater.CallRecord(3, 1, rater.VALID, ratings(), attempts=1)
    cache = rater.Cache(tmp_path, "a" * 64, "model-a")
    cache.put(record, ["{}"], "local")
    assert cache.get(3, 1) is not None
    assert rater.Cache(tmp_path, "a" * 64, "model-b").get(3, 1) is None
    assert rater.Cache(tmp_path, "b" * 64, "model-a").get(3, 1) is None
    shutil.copy(cache.path(3, 1), cache.path(7, 1))
    with pytest.raises(rater.RaterError, match="does not match its key"):
        cache.get(7, 1)


def test_the_pilot_shows_no_rating(files: list[Path], prompt: Prompt, tmp_path: Path) -> None:
    lines = run(files, prompt, ScriptedBackend(), tmp_path, pilot=True).report_lines()
    text = "\n".join(lines)
    assert "parse success 100.0%" in text
    for item in ITEMS:
        assert item.key not in text
    payload = json.loads((tmp_path / "out" / rater.PILOT_FILE).read_text("utf-8"))
    assert "reliability" not in payload


def test_the_wrong_number_of_transcripts_is_refused(prompt: Prompt, tmp_path: Path) -> None:
    files = write_transcripts(tmp_path / "t", [1, 2])
    with pytest.raises(rater.RaterError, match="2 transcripts"):
        run(files, prompt, ScriptedBackend(), tmp_path, pilot=True)


# ---------------------------------------------------------------------------
# Reliability
# ---------------------------------------------------------------------------
def test_icc_matches_shrout_and_fleiss() -> None:
    table = np.array(
        [[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8], [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]],
        dtype=float,
    )
    single, average = rater.icc2(table)
    assert single == pytest.approx(0.29, abs=0.005)
    assert average == pytest.approx(0.62, abs=0.005)


def test_icc_is_one_for_identical_calls_and_undefined_without_variance() -> None:
    identical = np.repeat(np.array([[1.0], [2.0], [4.0]]), 3, axis=1)
    assert rater.icc2(identical) == pytest.approx((1.0, 1.0))
    assert all(np.isnan(rater.icc2(np.ones((4, 3)))))


# ---------------------------------------------------------------------------
# The HTTP client
# ---------------------------------------------------------------------------
class Reply:
    """A chat completion as urlopen returns it."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.body = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> Reply:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self) -> bytes:
        return self.body


def http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("u", code, "x", email.message.Message(), io.BytesIO(b""))


def client(
    answers: list[Any], *, api_key: str | None = None
) -> tuple[ChatCompletionsBackend, list[urllib.request.Request]]:
    """A client whose requests are answered from `answers` and recorded."""
    seen: list[urllib.request.Request] = []
    replies: Iterator[Any] = iter(answers)

    def opener(request: urllib.request.Request, timeout: float) -> Reply:
        seen.append(request)
        answer = next(replies)
        if isinstance(answer, Exception):
            raise answer
        return Reply(answer)

    endpoint = Endpoint("local", "http://127.0.0.1:1/v1", "fake", None)
    settings = RequestSettings(timeout_s=1.0, max_output_tokens=300, transient_retries=2)
    backend = ChatCompletionsBackend(
        endpoint, settings, api_key=api_key, opener=opener, sleep=lambda _: None
    )
    return backend, seen


COMPLETION = {
    "choices": [{"message": {"content": "{}"}}],
    "usage": {"prompt_tokens": 12, "completion_tokens": 3},
}


def test_the_request_is_the_planned_one() -> None:
    backend, seen = client([COMPLETION], api_key="k")
    assert backend.complete("sys", "user") == Completion("{}", 12, 3)
    body = json.loads(seen[0].data)  # type: ignore[arg-type]
    assert seen[0].full_url == "http://127.0.0.1:1/v1/chat/completions"
    assert body["temperature"] == 0
    assert body["response_format"] == {"type": "json_object"}
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert seen[0].get_header("Authorization") == "Bearer k"

    keyless, seen = client([COMPLETION])
    keyless.complete("s", "u")
    assert seen[0].get_header("Authorization") is None


def test_transient_failures_are_retried_and_then_stop_the_run() -> None:
    backend, seen = client([http_error(429), urllib.error.URLError("down"), COMPLETION])
    assert backend.complete("s", "u").text == "{}"
    assert len(seen) == 3
    backend, _ = client([http_error(503)] * 3)
    with pytest.raises(BackendUnavailableError, match="after 3 tries"):
        backend.complete("s", "u")


def test_a_rejected_request_and_a_refusal_to_serve_are_told_apart() -> None:
    backend, seen = client([http_error(400)])
    with pytest.raises(RequestRejectedError):
        backend.complete("s", "u")
    assert len(seen) == 1
    backend, _ = client([http_error(401)])
    with pytest.raises(BackendUnavailableError, match="HTTP 401"):
        backend.complete("s", "u")
