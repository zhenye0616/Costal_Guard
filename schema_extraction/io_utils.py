from typing import Any, Optional, Tuple


def load_source(path: str, explicit_format: Optional[str] = None) -> Tuple[Any, str]:
    fmt = explicit_format or ("txt" if path.lower().endswith(".txt") else None)
    if fmt != "txt":
        raise ValueError("Only .txt inputs are supported")
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read(), "txt"


def load_mock(path: str) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()
