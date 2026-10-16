"""Groq generation adapter; retrieval and answer auditing remain in rag.py."""

from __future__ import annotations

import math
import os
import time
from typing import Any, Sequence

import requests

ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"


class HostedLLMError(RuntimeError):
    """A public-safe error, deliberately excluding provider bodies and credentials."""

    def __init__(self, message: str, *, code: str, status_code: int = 503,
                 retryable: bool = False, retry_after: int | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.retryable = retryable
        self.retry_after = retry_after


class GroqLLM:
    provider = "groq"

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.load_seconds: float | None = None
        self.notes: list[str] = []

    def load(self) -> None:
        """Validate configuration without downloading weights or spending quota."""
        if not os.getenv("GROQ_API_KEY", "").strip():
            raise HostedLLMError(
                "Chat has not been configured by the site owner yet.",
                code="chat_not_configured",
            )
        if not self.model_name:
            raise HostedLLMError("The chat model has not been configured.",
                                 code="chat_not_configured")
        self.load_seconds = 0.0

    def chat(self, messages: Sequence[dict[str, str]], max_new_tokens: int = 600,
             temperature: float = 0.0) -> tuple[str, dict[str, float]]:
        self.load()
        body: dict[str, Any] = {
            "model": self.model_name,
            "messages": list(messages),
            "temperature": float(temperature),
            "max_completion_tokens": int(max_new_tokens),
            "stream": False,
        }
        if self.model_name in {"openai/gpt-oss-20b", "openai/gpt-oss-120b"}:
            # Groq counts internal reasoning in the completion budget as well.
            body.update(reasoning_effort="low", include_reasoning=False,
                        max_completion_tokens=int(max_new_tokens) + 1024)

        started = time.monotonic()
        try:
            response = requests.post(
                ENDPOINT, json=body,
                headers={"Authorization": "Bearer " + os.environ["GROQ_API_KEY"].strip()},
                timeout=(10, 60), allow_redirects=False,
            )
        except requests.Timeout:
            raise HostedLLMError(
                "The chat service took too long to respond. Please try again.",
                code="provider_timeout", status_code=504, retryable=True,
            ) from None
        except requests.RequestException:
            raise HostedLLMError(
                "The chat service is temporarily unreachable. Please try again later.",
                code="provider_unreachable", retryable=True, retry_after=15,
            ) from None

        try:
            if response.status_code == 429:
                try:
                    delay = float(response.headers.get("Retry-After", "60"))
                    retry_after = max(1, min(3600, math.ceil(delay)))
                except (ValueError, OverflowError):
                    retry_after = 60
                raise HostedLLMError(
                    "The shared chat allowance is temporarily exhausted. "
                    f"Please try again in {retry_after} seconds.",
                    code="provider_rate_limited", status_code=429,
                    retryable=True, retry_after=retry_after,
                )
            if response.status_code in {401, 403}:
                raise HostedLLMError(
                    "The chat service could not authenticate. The site owner needs to "
                    "check its configuration.", code="provider_auth_error",
                )
            if not 200 <= response.status_code < 300:
                transient = response.status_code >= 500
                raise HostedLLMError(
                    "The chat service is temporarily unavailable. Please try again later."
                    if transient else "The chat service rejected the request. "
                    "The site owner needs to check its model configuration.",
                    code="provider_unavailable" if transient else "provider_request_rejected",
                    status_code=503 if transient else 502,
                    retryable=transient, retry_after=15 if transient else None,
                )
            try:
                payload = response.json()
                choice = payload["choices"][0]
                text = choice["message"]["content"]
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("missing final answer")
                usage = payload.get("usage") or {}
                prompt_tokens = float(usage.get("prompt_tokens", 0))
                new_tokens = float(usage.get("completion_tokens", 0))
                if not all(math.isfinite(n) and n >= 0 for n in (prompt_tokens, new_tokens)):
                    raise ValueError("invalid usage")
            except (ValueError, KeyError, IndexError, TypeError, AttributeError, OverflowError):
                raise HostedLLMError(
                    "The chat service returned no usable answer. Please try again.",
                    code="provider_invalid_response", status_code=502, retryable=True,
                ) from None
        finally:
            response.close()

        seconds = time.monotonic() - started
        return text.strip(), {
            "prompt_tokens": prompt_tokens,
            "new_tokens": new_tokens,
            "seconds": seconds,
            "tokens_per_second": new_tokens / seconds if seconds else 0.0,
            "truncated": float(choice.get("finish_reason") == "length"),
        }
