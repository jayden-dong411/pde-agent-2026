"""Model choice and API keys saved from the web app.

Stored as ``settings.json`` in the agent's home (``~/.pde_agent``), readable
only by the owner. A saved key is used in place of the provider's
environment variable; with nothing saved, the environment variable still works.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from .llm import DEEPSEEK_MODEL, DEFAULT_MODEL, PROVIDERS

SETTINGS_FILE = "settings.json"
MODELS = {
    "claude": [DEFAULT_MODEL, "claude-sonnet-5-5", "claude-haiku-5-5"],
    "deepseek": [DEEPSEEK_MODEL, "deepseek-v4-pro"],
}
DEFAULT_MODELS = {"claude": DEFAULT_MODEL, "deepseek": DEEPSEEK_MODEL}
_MODEL_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,99}")
_KEY = re.compile(r"[\x21-\x7e]{8,400}")  # printable ASCII, no spaces


def key_hint(key: str) -> str:
    """``sk-…abcd``: enough to recognise a key without showing it."""
    return f"{key[:3]}…{key[-4:]}" if len(key) > 10 else "…"


class Settings:
    def __init__(self, home: Path | str):
        self.path = Path(home) / SETTINGS_FILE
        self.data = self._load()

    def _load(self) -> dict:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        if not isinstance(data, dict):
            return {}
        keys = data.get("keys")
        clean = {
            "provider": data.get("provider") if data.get("provider") in PROVIDERS else None,
            "models": {p: m for p, m in (data.get("models") or {}).items()
                       if p in PROVIDERS and isinstance(m, str) and _MODEL_NAME.fullmatch(m)}
            if isinstance(data.get("models"), dict) else {},
            "keys": {p: k for p, k in keys.items() if p in PROVIDERS and isinstance(k, str) and k}
            if isinstance(keys, dict) else {},
        }
        cloud = data.get("cloud")
        if isinstance(cloud, dict) and isinstance(cloud.get("url"), str) and isinstance(cloud.get("token"), str):
            clean["cloud"] = {"url": cloud["url"], "token": cloud["token"], "enabled": bool(cloud.get("enabled"))}
        return clean

    @property
    def provider(self) -> str | None:
        return self.data.get("provider")

    def model(self, provider: str) -> str | None:
        return self.data.get("models", {}).get(provider)

    def key(self, provider: str) -> str:
        return self.data.get("keys", {}).get(provider, "")

    @property
    def cloud(self) -> dict:
        """``{"url", "token", "enabled"}`` for the experience hub, or ``{}``."""
        return dict(self.data.get("cloud") or {})

    def update_cloud(self, url: str, token: str = "", enabled: bool = True, clear: bool = False) -> None:
        """Remember (or forget) the experience hub. Raises ValueError on bad input."""
        data = {k: v for k, v in self.data.items()}
        if clear:
            data.pop("cloud", None)
        else:
            from .cloud import check_url

            url = check_url(url)
            token = token.strip() or (self.cloud.get("token", "") if self.cloud.get("url") == url else "")
            if not _KEY.fullmatch(token):
                raise ValueError("请填写经验中心的访问令牌")
            data["cloud"] = {"url": url, "token": token, "enabled": bool(enabled)}
        self._write(data)
        self.data = data

    def update(self, provider: str, model: str = "", api_key: str = "", clear_key: bool = False) -> None:
        """Choose ``provider`` (and its model); store or forget its key. Raises ValueError on bad input."""
        if provider not in PROVIDERS:
            raise ValueError(f"不认识的大模型：{provider}")
        model = model.strip()
        if model and not _MODEL_NAME.fullmatch(model):
            raise ValueError("模型名只能包含字母、数字和 . _ - : /")
        api_key = api_key.strip()
        if api_key and not _KEY.fullmatch(api_key):
            raise ValueError("API key 格式不对：应该是一串没有空格的字符")
        data = {**self.data, "provider": provider, "models": dict(self.data.get("models", {})),
                "keys": dict(self.data.get("keys", {}))}
        if model and model != DEFAULT_MODELS[provider]:
            data["models"][provider] = model
        else:
            data["models"].pop(provider, None)
        if clear_key:
            data["keys"].pop(provider, None)
        elif api_key:
            data["keys"][provider] = api_key
        self._write(data)
        self.data = data

    def _write(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        # created owner-only (0600) before the key is written into it
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)
        try:
            os.chmod(self.path, 0o600)
        except OSError:  # e.g. filesystems without permissions
            pass
