from __future__ import annotations

import os
from pathlib import Path


def as_path(path_text: str | Path) -> Path:
    """Resolve pasted Windows or POSIX paths on the local machine."""
    raw = str(path_text).strip().strip('"').strip("'").replace("\\", "/")
    if len(raw) >= 2 and raw[1] == ":" and os.name != "nt":
        raw = f"/mnt/{raw[0].lower()}{raw[2:]}"
    return Path(raw).expanduser().resolve()


def normalize_patient_id(value: str | int | float) -> str:
    """Normalize numeric IDs and existing patient IDs to patientNNNN."""
    text = str(value).strip()
    if text.lower().startswith("patient"):
        digits = "".join(ch for ch in text if ch.isdigit())
    else:
        digits = str(int(float(text)))
    return f"patient{int(digits):04d}"


def normalize_slice_type(value: str) -> str:
    """Normalize clinician-entered AHA slice type text."""
    text = str(value).strip().lower()
    aliases = {"base": "basal", "middle": "mid", "apex": "apical"}
    text = aliases.get(text, text)
    if text not in {"basal", "mid", "apical"}:
        raise ValueError(f"Unsupported slice_type {value!r}; expected basal, mid, or apical.")
    return text
