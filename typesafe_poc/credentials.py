"""Credential loading from the environment, and secret redaction for logs."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

ALLOWED_CREDENTIAL_VARS = ("NVIDIA_API_KEY", "TYPESAFE_API_KEY", "HF_TOKEN")

_SECRETS: list[str] = []


def _register_secret(value: str | None) -> None:
    if value and len(value) >= 8:
        _SECRETS.append(value)


def redact(text: Any) -> str:
    """Scrub any known credential (and key-shaped substrings) from text."""
    out = str(text)
    for secret in _SECRETS:
        if secret and secret in out:
            out = out.replace(secret, "***REDACTED***")
    # Belt and braces: catch key-shaped tokens even if never registered.
    out = re.sub(r"nvapi-[A-Za-z0-9_\-]{8,}", "nvapi-***REDACTED***", out)
    out = re.sub(r"\bsk-[A-Za-z0-9_\-]{16,}", "sk-***REDACTED***", out)
    out = re.sub(r"\bhf_[A-Za-z0-9]{16,}", "hf_***REDACTED***", out)
    return out


def load_credentials(require_nvidia: bool, require_typesafe: bool) -> dict[str, str | None]:
    """Read credentials from the environment only. Never returns values to logs."""
    try:
        from dotenv import load_dotenv

        load_dotenv(_project_root() / ".env", override=False)
    except ImportError:
        pass

    creds = {name: os.environ.get(name) or None for name in ALLOWED_CREDENTIAL_VARS}
    for value in creds.values():
        _register_secret(value)

    missing: list[str] = []
    if require_nvidia and not creds["NVIDIA_API_KEY"]:
        missing.append("NVIDIA_API_KEY")
    if require_typesafe and not creds["TYPESAFE_API_KEY"]:
        missing.append("TYPESAFE_API_KEY")

    if missing:
        hint = ""
        if "NVIDIA_API_KEY" in missing:
            raw = _project_root() / ".env"
            if raw.exists() and "NVIDIA-KEY" in raw.read_text(errors="ignore"):
                hint = (
                    "\n  HINT: your .env contains `NVIDIA-KEY`. A hyphen is not a valid\n"
                    "        environment-variable name. Rename it to `NVIDIA_API_KEY`."
                )
        raise SystemExit(
            f"\nERROR: missing required credential(s): {', '.join(missing)}\n"
            f"  Set them in .env or the environment. See .env.example.{hint}\n"
            "  This POC never fabricates API results, so it cannot continue.\n"
        )
    return creds


def _project_root() -> Path:
    """The repository root (one level above this package)."""
    return Path(__file__).resolve().parent.parent
