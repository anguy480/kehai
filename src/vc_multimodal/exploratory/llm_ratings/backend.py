"""The two backends, both an OpenAI-compatible `/chat/completions` endpoint.

`openai` is the OpenAI API with `gpt-4.1-mini`; `local` is any server that
speaks the same protocol (vLLM, llama.cpp, Ollama, LM Studio). One client
serves both, through the standard library, so neither needs a dependency and
the request is the same apart from its URL, model and key.

Three kinds of failure are kept apart, because the plan treats them
differently:

* **A malformed response** came back but is not a valid rating. The rater
  retries it, and counts it (`rater.py`).
* **A rejected request** (HTTP 400, 413 or 422) will be rejected again, such as
  a dialogue longer than the model's context. It fails that call, and so the
  session, without retrying.
* **An unavailable backend** (network failure, timeout, rate limit, server
  error, after `transient_retries` tries with backoff; or a bad key, unknown
  model or wrong URL at once) stops the run. Nothing came back to rate, so it
  is not a rating attempt, and whatever was cached before it stays cached.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final, Protocol

from vc_multimodal.exploratory.llm_ratings.plan import (
    TEMPERATURE,
    Endpoint,
    PlanError,
    RequestSettings,
)

REJECTED_STATUSES: Final = frozenset({400, 413, 422})
TRANSIENT_STATUSES: Final = frozenset({408, 409, 429, 500, 502, 503, 504})
BACKOFF_BASE_S: Final = 2.0
BACKOFF_CAP_S: Final = 60.0


@dataclass(frozen=True, slots=True)
class Completion:
    """One response: its text and what it cost."""

    text: str | None
    prompt_tokens: int | None
    completion_tokens: int | None


class RequestRejectedError(RuntimeError):
    """The backend refused this request, and would again."""


class BackendUnavailableError(RuntimeError):
    """The backend could not be reached or would not serve; the run stops."""


class Backend(Protocol):
    """Anything that turns a system and a user message into one completion."""

    @property
    def model(self) -> str:
        """The model name, part of every cache key."""
        ...

    def complete(self, system: str, user: str) -> Completion:
        """One call.

        Raises:
            RequestRejectedError: if this request would always be refused.
            BackendUnavailableError: if no response could be had.
        """
        ...


Opener = Callable[[urllib.request.Request, float], Any]


def _open(request: urllib.request.Request, timeout: float) -> Any:
    return urllib.request.urlopen(request, timeout=timeout)


class ChatCompletionsBackend:
    """An OpenAI-compatible chat completions endpoint."""

    def __init__(
        self,
        endpoint: Endpoint,
        settings: RequestSettings,
        *,
        api_key: str | None = None,
        opener: Opener = _open,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """A client for `endpoint`; `opener` and `sleep` are replaced only in tests."""
        self._endpoint = endpoint
        self._settings = settings
        self._api_key = api_key
        self._opener = opener
        self._sleep = sleep

    @property
    def model(self) -> str:
        """The model name."""
        return self._endpoint.model

    @property
    def url(self) -> str:
        """Where requests go."""
        return f"{self._endpoint.base_url}/chat/completions"

    def body(self, system: str, user: str) -> dict[str, Any]:
        """The request, as the plan fixes it: temperature 0, JSON requested."""
        return {
            "model": self._endpoint.model,
            "temperature": TEMPERATURE,
            "max_tokens": self._settings.max_output_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }

    def complete(self, system: str, user: str) -> Completion:
        """One call, retrying only what may succeed if tried again."""
        payload = json.dumps(self.body(system, user), ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        tries = self._settings.transient_retries + 1
        last = ""
        for attempt in range(tries):
            if attempt:
                self._sleep(min(BACKOFF_BASE_S * 2 ** (attempt - 1), BACKOFF_CAP_S))
            request = urllib.request.Request(self.url, data=payload, headers=headers, method="POST")
            try:
                with self._opener(request, self._settings.timeout_s) as response:
                    return _completion(json.loads(response.read().decode("utf-8")))
            except urllib.error.HTTPError as exc:
                if exc.code in REJECTED_STATUSES:
                    msg = f"the backend rejected the request (HTTP {exc.code})"
                    raise RequestRejectedError(msg) from None
                if exc.code not in TRANSIENT_STATUSES:
                    msg = f"the backend refused to serve (HTTP {exc.code}) at {self.url}"
                    raise BackendUnavailableError(msg) from None
                last = f"HTTP {exc.code}"
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                last = type(exc).__name__
            except (json.JSONDecodeError, UnicodeDecodeError):
                last = "a reply that is not a chat completion"
        msg = f"no response from {self.url} after {tries} tries (last: {last})"
        raise BackendUnavailableError(msg)


def _completion(reply: Any) -> Completion:
    """The text and token counts of a chat completion reply."""
    try:
        text = reply["choices"][0]["message"].get("content")
    except (KeyError, IndexError, TypeError, AttributeError):
        text = None
    usage = reply.get("usage") if isinstance(reply, dict) else None
    usage = usage if isinstance(usage, dict) else {}
    prompt_tokens, completion_tokens = usage.get("prompt_tokens"), usage.get("completion_tokens")
    return Completion(
        text=text if isinstance(text, str) else None,
        prompt_tokens=prompt_tokens if isinstance(prompt_tokens, int) else None,
        completion_tokens=completion_tokens if isinstance(completion_tokens, int) else None,
    )


def make_backend(endpoint: Endpoint, settings: RequestSettings) -> ChatCompletionsBackend:
    """The configured backend, with its key read from the environment.

    Raises:
        PlanError: if the backend needs a key and its variable is unset.
    """
    api_key = None
    if endpoint.api_key_env:
        api_key = os.environ.get(endpoint.api_key_env)
        if not api_key:
            msg = f"set {endpoint.api_key_env} to the {endpoint.backend} API key"
            raise PlanError(msg)
    return ChatCompletionsBackend(endpoint, settings, api_key=api_key)
