"""Pluggable LLM backends: Ollama (local), OpenAI (ChatGPT), Google AI (Gemini)."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from .config import get_llm_config

try:
    import ollama

    OLLAMA_AVAILABLE = True
except ImportError:
    ollama = None
    OLLAMA_AVAILABLE = False

try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    requests = None
    REQUESTS_AVAILABLE = False


def _strip_json_fences(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r"^```json\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"^```\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()


def is_llm_available() -> bool:
    cfg = get_llm_config()
    provider = cfg["provider"]
    if provider == "ollama":
        return OLLAMA_AVAILABLE
    if provider == "openai":
        return bool(os.environ.get("OPENAI_API_KEY"))
    if provider == "google":
        return bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))
    return False


def _chat_ollama(system_prompt: str, user_prompt: str, model: str) -> dict[str, Any]:
    if not OLLAMA_AVAILABLE:
        raise RuntimeError("ollama package not installed (pip install ollama)")
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": 0.0},
        format="json",
    )
    raw = _strip_json_fences(response["message"]["content"])
    return json.loads(raw)


def _chat_openai(system_prompt: str, user_prompt: str, model: str) -> dict[str, Any]:
    if not REQUESTS_AVAILABLE:
        raise RuntimeError("requests package not installed")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    raw = response.json()["choices"][0]["message"]["content"]
    return json.loads(_strip_json_fences(raw))


def _chat_google(system_prompt: str, user_prompt: str, model: str) -> dict[str, Any]:
    if not REQUESTS_AVAILABLE:
        raise RuntimeError("requests package not installed")
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY or GEMINI_API_KEY not set")

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": {
            "temperature": 0.0,
            "responseMimeType": "application/json",
        },
    }
    response = requests.post(url, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()
    parts = data["candidates"][0]["content"]["parts"]
    raw = parts[0].get("text", "")
    return json.loads(_strip_json_fences(raw))


def chat_json(system_prompt: str, user_prompt: str, *, model: str | None = None) -> dict[str, Any]:
    """Call the configured LLM provider and parse a JSON object response."""
    cfg = get_llm_config()
    model = model or cfg["model"]
    provider = cfg["provider"]

    if provider == "openai":
        return _chat_openai(system_prompt, user_prompt, model)
    if provider == "google":
        return _chat_google(system_prompt, user_prompt, model)
    return _chat_ollama(system_prompt, user_prompt, model)


_logged_provider = False


def log_llm_provider_once() -> None:
    global _logged_provider
    if _logged_provider:
        return
    _logged_provider = True
    cfg = get_llm_config()
    print(
        f"LLM provider: {cfg['provider']} | model: {cfg['model']} | "
        f"available: {is_llm_available()}"
    )
