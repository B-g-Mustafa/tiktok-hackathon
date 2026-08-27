"""Shared logging setup for the CLI scripts.

`logging.basicConfig(level=logging.INFO)` sets the ROOT logger's level, which
every library's logger inherits unless it sets its own. `huggingface_hub`'s
newer xet storage backend logs every chunk request through `httpx` at INFO
level -- "HTTP Request: GET ... 206 Partial Content" -- and with the root at
INFO, that floods stdout with lines that look like errors (206/302 are
success codes) but are actually just normal, healthy download traffic. This
was happening in every script that called `logging.basicConfig` directly;
`configure_logging()` is the fix, applied once, everywhere.

It also optionally writes to a log FILE, for jobs submitted via `sbatch`.
SLURM buffers a job's stdout and only flushes it to the `.out` file
periodically (or at exit), so `tail -f` on that file during a multi-hour run
can show nothing for a long time even though the job is progressing fine --
easy to mistake for a hang. `logging.FileHandler` flushes after every single
line by default (verified directly, not assumed), so a log file written this
way stays current in real time regardless of how SLURM buffers stdout.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

__all__ = ["configure_logging"]

# Libraries whose own INFO-level chatter is noise for our purposes, even
# though we want OUR script's INFO messages (progress, "wrote N rows", etc.)
# to show. Add to this list if a new noisy dependency shows up.
_NOISY_LOGGERS = ("httpx", "httpcore", "urllib3", "hf_xet", "filelock")


def configure_logging(
    level: int = logging.INFO,
    format: str = "%(message)s",  # noqa: A002 - matches logging.basicConfig's own name
    log_file: Path | str | None = None,
    **kwargs,
) -> None:
    """Set up root logging for a CLI script while silencing known-noisy
    third-party HTTP/transport loggers, and optionally mirroring everything
    to a log file that stays live-updated even under SLURM's stdout buffering.

    Parameter names deliberately match `logging.basicConfig` (including
    shadowing the `format` builtin) so callers can pass e.g. `stream=` or a
    custom `format=` straight through without a name collision.

    Every existing `logger.info/warning/error(...)` call anywhere in this
    codebase is captured automatically once this runs -- shard-fetch
    retries, resume messages, materialize's progress fallback, everything --
    since they all go through the root logger. Nothing needs to be rewritten
    per-script to benefit from `log_file`.
    """
    logging.basicConfig(level=level, format=format, **kwargs)
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
        )
        logging.getLogger().addHandler(file_handler)

        logging.getLogger(__name__).info(
            "logging to file: %s (also printed to stdout above)",
            log_file.resolve(),
        )


def stdout_is_interactive() -> bool:
    """False under `sbatch` (stdout is redirected to a file, not a TTY) or
    any other non-interactive context. Used to decide whether a live-redraw
    progress bar (tqdm) makes sense, or whether periodic plain log lines --
    which actually show up correctly in a static log file -- are what's
    needed instead.
    """
    return sys.stdout.isatty()
