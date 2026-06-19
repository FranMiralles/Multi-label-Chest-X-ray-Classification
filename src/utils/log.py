"""Stdout tee — mirrors prints to a timestamped file in results/logs/."""

import sys
from contextlib import contextmanager
from datetime import datetime

from src.config.config import LOGS_DIR


class _Tee:
    """Writes to both the original stdout and a file simultaneously."""

    def __init__(self, stream, filepath):
        self._stream = stream
        self._file = open(filepath, "w", buffering=1, encoding="utf-8")

    def write(self, data):
        self._stream.write(data)
        self._file.write(data)

    def flush(self):
        self._stream.flush()
        self._file.flush()

    def close(self):
        self._file.close()


@contextmanager
def log_run(name: str):
    """Mirrors all prints to results/logs/<name>_<YYYYMMDD_HHMMSS>.txt."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"{name}_{timestamp}.txt"

    tee = _Tee(sys.stdout, log_path)
    old_stdout = sys.stdout
    sys.stdout = tee
    try:
        print(f"Log: {log_path}")
        yield log_path
    finally:
        sys.stdout = old_stdout
        tee.close()
