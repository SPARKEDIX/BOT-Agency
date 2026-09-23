"""Single shared NIM client. All traffic goes through integrate.api.nvidia.com/v1."""
from __future__ import annotations

import random
import time
from typing import Callable

from openai import APITimeoutError, APIConnectionError, APIStatusError, OpenAI

import config
from agency.rate_limiter import limiter


_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=config.BASE_URL,
            api_key=config.require_key(),
            timeout=120.0,
        )
    return _client


def _is_retryable(err: Exception) -> bool:
    msg = str(err).lower()
    retry_words = (
        "429", "500", "502", "503", "529",
        "overload", "temporarily", "try again", "timeout",
        "rate", "limit", "busy", "unavailable", "connection",
        "reset", "closed",
    )
    if isinstance(err, (APITimeoutError, APIConnectionError)):
        return True
    if isinstance(err, APIStatusError):
        return err.status_code in (429, 500, 502, 503, 529)
    return any(w in msg for w in retry_words)


def chat(
    messages: list[dict],
    model: str,
    temperature: float = 1.0,
    top_p: float = 0.95,
    max_tokens: int = 16384,
    enable_thinking: bool = False,
    stream_output: bool = True,
    retries: int = 5,
    on_retry: Callable[[int, int, str], None] | None = None,
) -> dict:
    """Rate-limited chat with retries for NIM overloads.

    Streams like your base code (reasoning_content + content).
    Retries transient errors (overloaded / 5xx / timeout) with
    exponential backoff + jitter. Falls back to non-streaming
    on the last attempt if streaming keeps failing.
    """
    client = get_client()
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        limiter.wait()  # 40 RPM shared budget
        use_stream = not (attempt == retries) or retries == 1
        # last attempt: try non-streaming fallback if streaming failed before
        if attempt == retries and last_err is not None:
            use_stream = False
        try:
            if use_stream:
                completion = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
                    stream=True,
                )
                reasoning_parts: list[str] = []
                content_parts: list[str] = []
                for chunk in completion:  # <- your traceback was here: overloaded mid-stream
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    reasoning = getattr(delta, "reasoning_content", None)
                    if reasoning:
                        reasoning_parts.append(reasoning)
                        if stream_output:
                            print(reasoning, end="")
                    if delta.content is not None:
                        content_parts.append(delta.content)
                        if stream_output:
                            print(delta.content, end="")
                if stream_output:
                    print()
                content = "".join(content_parts)
                if not content and attempt < retries:
                    raise RuntimeError("Empty response from NIM (transient), retrying")
                return {"content": content, "reasoning": "".join(reasoning_parts)}
            # non-streaming fallback
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
                stream=False,
            )
            delta = resp.choices[0].message
            reasoning = getattr(delta, "reasoning_content", None) or ""
            content = delta.content or ""
            if stream_output and content:
                print(content)
            return {"content": content, "reasoning": reasoning}
        except Exception as e:  # noqa: BLE001 - must catch APIError mid-stream too
            last_err = e
            if not _is_retryable(e) or attempt == retries:
                break
            wait = (2 ** (attempt - 1)) * 2 + random.uniform(0, 1.5)
            if on_retry:
                on_retry(attempt, retries, f"{type(e).__name__}: {e} — retry in {wait:.1f}s")
            else:
                print(f"[nim] {type(e).__name__}: {e} — retry {attempt}/{retries} in {wait:.1f}s")
            time.sleep(wait)
    raise RuntimeError(f"NIM request failed after {retries} tries: {last_err}")
