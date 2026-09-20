from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

from .util import canonical_json, sha256_text, utc_now


@dataclass
class OpenRouterResult:
    ok: bool
    http_status: Optional[int]
    response_json: Dict[str, Any]
    assistant_text: str
    latency_seconds: float
    attempts: List[Dict[str, Any]]
    error: Optional[str]
    request_hash: str


def _assistant_text(response: Dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        return ""
    content = (choices[0].get("message") or {}).get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                pieces.append(item["text"])
        return "\n".join(pieces)
    return str(content or "")


def call_openrouter(config: Dict[str, Any], model_id: str, messages: List[Dict[str, str]]) -> OpenRouterResult:
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is missing. Copy .env.example to .env and set the key.")
    api = config["api"]
    generation = config["generation"]
    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": generation["temperature"],
        "top_p": generation["top_p"],
        "n": generation["n"],
        "max_tokens": generation["max_tokens"],
        "stream": generation["stream"],
        "provider": api["provider"],
    }
    request_hash = sha256_text(canonical_json(payload))
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    referer = os.environ.get("OPENROUTER_HTTP_REFERER", "").strip()
    title = os.environ.get("OPENROUTER_APP_TITLE", "").strip()
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-OpenRouter-Title"] = title

    attempts: List[Dict[str, Any]] = []
    started_all = time.perf_counter()
    max_retries = int(api["max_network_retries"])
    backoff = float(api["retry_backoff_seconds"])
    timeout = int(api["timeout_seconds"])
    for attempt in range(max_retries + 1):
        started = time.perf_counter()
        try:
            response = requests.post(api["base_url"], headers=headers, json=payload, timeout=timeout)
            latency = time.perf_counter() - started
            try:
                body = response.json()
            except ValueError:
                body = {"unparsed_body": response.text[:2000]}
            attempts.append({
                "attempt": attempt,
                "timestamp_utc": utc_now(),
                "http_status": response.status_code,
                "latency_seconds": latency,
                "retryable": response.status_code == 429 or response.status_code >= 500,
            })
            if response.status_code == 200:
                return OpenRouterResult(
                    ok=True,
                    http_status=response.status_code,
                    response_json=body,
                    assistant_text=_assistant_text(body),
                    latency_seconds=time.perf_counter() - started_all,
                    attempts=attempts,
                    error=None,
                    request_hash=request_hash,
                )
            error = f"HTTP {response.status_code}: {body}"
            if response.status_code != 429 and response.status_code < 500:
                return OpenRouterResult(False, response.status_code, body, "", time.perf_counter() - started_all, attempts, error, request_hash)
        except requests.RequestException as exc:
            latency = time.perf_counter() - started
            error = f"{type(exc).__name__}: {exc}"
            attempts.append({
                "attempt": attempt,
                "timestamp_utc": utc_now(),
                "http_status": None,
                "latency_seconds": latency,
                "retryable": True,
                "error": error,
            })
            body = {}
        if attempt < max_retries:
            time.sleep(backoff * (2 ** attempt))
    return OpenRouterResult(False, attempts[-1].get("http_status"), body, "", time.perf_counter() - started_all, attempts, error, request_hash)

