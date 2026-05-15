"""
llm_local.py  —  Local-first LLM router for dss+ ESG Chatbot
=============================================================
Priority (auto-detected):
  1. OLLAMA  — fully local (phi3 recommended)
  2. HUGGINGFACE — free cloud tier
  3. AZURE OPENAI — enterprise Copilot

secrets.toml:
    LLM_PROVIDER = "ollama"
    OLLAMA_MODEL = "phi3"

Speed improvements:
  - call_stream() yields tokens as they arrive (use in UI for instant feedback)
  - call() still returns full string (used for non-streaming paths)
  - num_predict capped at 500 for faster responses
  - num_ctx kept at 2048 (matches our compact context builder)
"""
from __future__ import annotations
import os
import json
import requests
from typing import Optional, Generator

OLLAMA_BASE_URL      = "http://localhost:11434"
OLLAMA_DEFAULT_MODEL = "phi3"
HF_DEFAULT_MODEL     = "mistralai/Mistral-7B-Instruct-v0.3"


def _get_secret(key: str, default: str = "") -> str:
    try:
        import streamlit as st
        val = st.secrets.get(key, "")
        if val:
            return str(val)
    except Exception:
        pass
    return os.environ.get(key, default)


class LocalLLMEngine:
    def __init__(self):
        self._detect_provider()

    def _detect_provider(self):
        explicit = _get_secret("LLM_PROVIDER", "").lower()
        if explicit == "azure":
            self.provider = "azure"
        elif explicit == "huggingface":
            self.provider = "huggingface"
        elif explicit == "ollama":
            self.provider = "ollama"
        else:
            if _get_secret("AZURE_OPENAI_KEY") and _get_secret("AZURE_OPENAI_ENDPOINT"):
                self.provider = "azure"
            elif _get_secret("HF_API_TOKEN"):
                self.provider = "huggingface"
            else:
                self.provider = "ollama"

        if self.provider == "azure":
            self.azure_key      = _get_secret("AZURE_OPENAI_KEY")
            self.azure_endpoint = _get_secret("AZURE_OPENAI_ENDPOINT")
        elif self.provider == "huggingface":
            self.hf_token = _get_secret("HF_API_TOKEN")
            self.hf_model = _get_secret("HF_MODEL", HF_DEFAULT_MODEL)
        elif self.provider == "ollama":
            self.ollama_url   = _get_secret("OLLAMA_URL", OLLAMA_BASE_URL)
            self.ollama_model = _get_secret("OLLAMA_MODEL", OLLAMA_DEFAULT_MODEL)

    def provider_label(self) -> str:
        if self.provider == "ollama":
            return f"Ollama · {self.ollama_model}"
        if self.provider == "huggingface":
            return f"HuggingFace · {self.hf_model.split('/')[-1]}"
        if self.provider == "azure":
            return "Azure OpenAI (Copilot)"
        return "Unknown"

    def _resolve_model_name(self) -> str:
        try:
            r = requests.get(f"{self.ollama_url}/api/tags", timeout=4)
            models = [m["name"] for m in r.json().get("models", [])]
            base = self.ollama_model.split(":")[0].lower()
            for m in models:
                if m.lower() == self.ollama_model.lower():
                    return m
            for m in models:
                if m.lower().startswith(base + ":"):
                    return m
            return self.ollama_model
        except Exception:
            return self.ollama_model

    def is_available(self) -> tuple:
        if self.provider == "ollama":
            try:
                r = requests.get(f"{self.ollama_url}/api/tags", timeout=4)
                models = [m["name"] for m in r.json().get("models", [])]
                base = self.ollama_model.split(":")[0].lower()
                found = any(
                    m.lower() == self.ollama_model.lower() or
                    m.lower().startswith(base + ":")
                    for m in models
                )
                if found:
                    return True, f"Ollama · {self._resolve_model_name()} ready"
                return False, (f"Model '{self.ollama_model}' not found. "
                               f"Run: ollama pull {self.ollama_model}")
            except requests.exceptions.ConnectionError:
                return False, "Ollama not running. Open from system tray."
            except Exception as e:
                return False, f"Ollama check error: {e}"
        elif self.provider == "huggingface":
            if not _get_secret("HF_API_TOKEN"):
                return False, "HF_API_TOKEN missing in secrets.toml"
            return True, f"HuggingFace · {self.hf_model}"
        elif self.provider == "azure":
            if _get_secret("AZURE_OPENAI_KEY") and _get_secret("AZURE_OPENAI_ENDPOINT"):
                return True, "Azure OpenAI credentials found"
            return False, "Azure credentials missing"
        return False, "No provider configured"

    # ── Streaming call — yields text chunks as they arrive ────────────────────
    def call_stream(self, user_message: str, data_context: str,
                    history=None, system_prompt: str = "") -> Generator[str, None, None]:
        """
        Yields text tokens as they stream from the model.
        Only implemented for Ollama (HF/Azure fall back to full call).
        Usage in UI:
            for chunk in bot.copilot.call_stream(...):
                accumulated += chunk
                placeholder.markdown(accumulated + "▌")
        """
        messages = []
        if history:
            messages.extend(history[-6:])
        messages.append({
            "role":    "user",
            "content": f"{data_context}\n\n---\nQuestion: {user_message}",
        })

        if self.provider == "ollama":
            yield from self._stream_ollama(messages, system_prompt)
        else:
            # Non-streaming providers: yield full response at once
            result = self.call(user_message, data_context, history, system_prompt)
            yield result

    def _stream_ollama(self, messages: list, system_prompt: str) -> Generator[str, None, None]:
        model = self._resolve_model_name()
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        payload = {
            "model":    model,
            "messages": full_messages,
            "stream":   True,   # ← streaming enabled
            "options":  {
                "temperature": 0.3,
                "num_predict": 500,
                "num_ctx":     2048,
            },
        }
        try:
            resp = requests.post(
                f"{self.ollama_url}/api/chat",
                json=payload,
                stream=True,
                timeout=300,
            )
            if resp.status_code != 200:
                yield f"❌ Ollama error {resp.status_code}: {resp.text[:200]}"
                return

            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line.decode("utf-8"))
                    chunk = data.get("message", {}).get("content", "")
                    if chunk:
                        yield chunk
                    if data.get("done"):
                        break
                except json.JSONDecodeError:
                    continue

        except requests.exceptions.ConnectionError:
            yield "❌ Ollama not running. Open from system tray."
        except requests.exceptions.Timeout:
            yield "\n\n⏱️ Response timed out. Try a shorter question."
        except Exception as e:
            yield f"❌ Error: {str(e)}"

    # ── Non-streaming call (used internally + for HF/Azure) ───────────────────
    def call(self, user_message: str, data_context: str,
             history=None, system_prompt: str = "") -> str:
        messages = []
        if history:
            messages.extend(history[-6:])
        messages.append({
            "role":    "user",
            "content": f"{data_context}\n\n---\nQuestion: {user_message}",
        })
        if self.provider == "ollama":
            return self._call_ollama(messages, system_prompt)
        elif self.provider == "huggingface":
            return self._call_huggingface(messages, system_prompt)
        elif self.provider == "azure":
            return self._call_azure(messages, system_prompt)
        return "⚠️ No LLM provider configured."

    def _call_ollama(self, messages: list, system_prompt: str) -> str:
        model = self._resolve_model_name()
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        payload = {
            "model":    model,
            "messages": full_messages,
            "stream":   False,
            "options":  {"temperature": 0.3, "num_predict": 500, "num_ctx": 2048},
        }
        try:
            resp = requests.post(
                f"{self.ollama_url}/api/chat",
                json=payload,
                timeout=300,
            )
            if resp.status_code == 500:
                return "❌ Ollama returned an error. Try a shorter question."
            resp.raise_for_status()
            data = resp.json()
            if "message" in data:
                return data["message"].get("content", "")
            return data.get("response", "")
        except requests.exceptions.ConnectionError:
            return "❌ Ollama not running. Open from system tray."
        except requests.exceptions.Timeout:
            return "⏱️ Timed out. Try a shorter/simpler question."
        except Exception as e:
            return f"❌ Ollama error: {str(e)}"

    def _call_huggingface(self, messages: list, system_prompt: str) -> str:
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)
        url = f"https://api-inference.huggingface.co/models/{self.hf_model}/v1/chat/completions"
        payload = {"model": self.hf_model, "messages": full_messages,
                   "max_tokens": 800, "temperature": 0.3, "stream": False}
        try:
            resp = requests.post(
                url,
                headers={"Authorization": f"Bearer {self.hf_token}",
                         "Content-Type": "application/json"},
                json=payload, timeout=90)
            if resp.status_code == 503:
                return "⏳ HuggingFace model loading. Try again in 20s."
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"❌ HuggingFace error: {str(e)}"

    def _call_azure(self, messages: list, system_prompt: str) -> str:
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)
        payload = {"messages": full_messages, "max_tokens": 1000, "temperature": 0.3}
        try:
            resp = requests.post(
                self.azure_endpoint,
                headers={"api-key": self.azure_key, "Content-Type": "application/json"},
                json=payload, timeout=60)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"❌ Azure error: {str(e)}"