"""Thin wrapper around the local Ollama HTTP API."""

from __future__ import annotations

import json
import re
from typing import Any

import requests

DEFAULT_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3.1:8b"


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        url: str = DEFAULT_URL,
        timeout: int = 120,
        temperature: float = 0.1,
    ):
        self.model = model
        self.url = url
        self.timeout = timeout
        self.temperature = temperature

    def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float | None = None,
        json_mode: bool = False,
        num_predict: int = 1024,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": f"{system}\n\n{prompt}" if system else prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature if temperature is None else temperature,
                "num_predict": num_predict,
            },
        }
        if json_mode:
            payload["format"] = "json"
        try:
            r = requests.post(self.url, json=payload, timeout=self.timeout)
        except requests.ConnectionError as e:
            raise OllamaError(
                f"Cannot reach Ollama at {self.url}. Is `ollama serve` running?"
            ) from e
        except requests.exceptions.Timeout as e:
            raise OllamaError(
                f"Ollama timed out after {self.timeout}s for model '{self.model}'."
            ) from e
        if r.status_code == 404:
            raise OllamaError(
                f"Model '{self.model}' not found. Run: ollama pull {self.model}"
            )
        r.raise_for_status()
        return r.json().get("response", "").strip()

    def generate_json(
        self,
        prompt: str,
        system: str = "",
        temperature: float | None = None,
        num_predict: int = 2048,
    ) -> Any:
        """Generate and parse a JSON response. Uses Ollama's native JSON mode."""
        raw = self.generate(
            prompt,
            system=system,
            temperature=temperature,
            json_mode=True,
            num_predict=num_predict,
        )
        return _parse_json_loose(raw)

    def health_check(self) -> None:
        """Raise OllamaError if the server or model is unreachable."""
        self.generate("ok", temperature=0.0, num_predict=1)


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_json_loose(text: str) -> Any:
    text = text.strip()
    if not text:
        raise OllamaError("Empty response from Ollama.")
    # Strip code fences if present
    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fall back: find the first {...} block
    m = _JSON_OBJ_RE.search(text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError as e:
            raise OllamaError(f"Could not parse JSON from model output: {text[:200]}") from e
    raise OllamaError(f"No JSON object in model output: {text[:200]}")
