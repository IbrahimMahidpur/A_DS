"""
Unified LLM client — supports both OpenCode Zen and Ollama backends.
Routes by model prefix: opencode/ → OpenCode Zen, ollama/ → Ollama.
"""
import os
import logging
import time
from typing import Optional

import httpx

from multimodal_ds.config import (
    OLLAMA_BASE_URL,
    LLM_TIMEOUT,
    LLM_RETRIES,
)

logger = logging.getLogger(__name__)


def _call_opencode_zen(
    messages: list[dict],
    model: str,
    max_tokens: int = 4000,
    temperature: float = 0.3,
) -> str:
    """Call OpenCode Zen API with retry logic."""
    import json

    api_key = os.getenv("OPENCODE_ZEN_API_KEY", "")
    if not api_key:
        logger.warning("[LLM] OPENCODE_ZEN_API_KEY not set, falling back to Ollama")
        return _call_ollama(messages, model, max_tokens, temperature)

    api_url = "https://opencode.zenacademy.ai/api/chat"

    # Strip prefix from model name
    model = model.replace("opencode/", "")

    for attempt in range(LLM_RETRIES + 1):
        try:
            response = httpx.post(
                api_url,
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(connect=10.0, read=LLM_TIMEOUT, write=LLM_TIMEOUT, pool=5.0),
            )
            if response.status_code == 200:
                return response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
            elif response.status_code == 429:
                wait = 2 ** attempt
                logger.warning(f"[LLM] Rate limited, retrying in {wait}s...")
                time.sleep(wait)
                continue
            else:
                logger.warning(f"[LLM] OpenCode request failed: {response.status_code}")
        except Exception as e:
            logger.warning(f"[LLM] OpenCode call failed (attempt {attempt + 1}): {e}")
            if attempt < LLM_RETRIES:
                time.sleep(2 ** attempt)
                continue
    return f"[Error: OpenCode Zen failed after {LLM_RETRIES + 1} attempts]"


def _call_ollama(
    messages: list[dict],
    model: str,
    max_tokens: int = 4000,
    temperature: float = 0.3,
) -> str:
    """Call local Ollama instance."""
    # Strip prefix from model name
    model = model.replace("ollama/", "")

    try:
        response = httpx.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                },
            },
            timeout=httpx.Timeout(connect=10.0, read=LLM_TIMEOUT, write=LLM_TIMEOUT, pool=5.0),
        )
        if response.status_code == 200:
            return response.json().get("message", {}).get("content", "")
    except httpx.TimeoutException:
        logger.warning(f"[LLM] Ollama request timed out after {LLM_TIMEOUT}s")
    except Exception as e:
        logger.warning(f"[LLM] Ollama call failed: {e}")
    return f"[Error: Ollama request failed]"


def chat(
    messages: list[dict],
    model: str = "ollama/qwen2.5:7b",
    max_tokens: int = 4000,
    temperature: float = 0.3,
) -> str:
    """
    Unified chat interface — routes to appropriate backend based on model prefix.
    - opencode/* → OpenCode Zen API
    - ollama/*  → local Ollama
    """
    if model.startswith("opencode/"):
        return _call_opencode_zen(messages, model, max_tokens, temperature)
    elif model.startswith("ollama/"):
        return _call_ollama(messages, model, max_tokens, temperature)
    else:
        # Default to Ollama
        return _call_ollama(messages, f"ollama/{model}", max_tokens, temperature)


def chat_with_fallback(
    messages: list[dict],
    primary_model: str = "opencode/minimax-m2.5-free",
    fallback_model: str = "ollama/qwen2.5:7b",
    max_tokens: int = 4000,
    temperature: float = 0.3,
) -> str:
    """
    Try primary model first, fall back to secondary if it fails.
    Useful for when primary API is rate-limited or unavailable.
    """
    result = chat(messages, primary_model, max_tokens, temperature)
    if result.startswith("[Error:"):
        logger.warning(f"[LLM] Primary model {primary_model} failed, trying fallback...")
        return chat(messages, fallback_model, max_tokens, temperature)
    return result