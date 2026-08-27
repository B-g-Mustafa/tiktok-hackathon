"""Shared logging setup for the CLI scripts.

`logging.basicConfig(level=logging.INFO)` sets the ROOT logger's level, which
every library's logger inherits unless it sets its own. `huggingface_hub`'s
newer xet storage backend logs every chunk request through `httpx` at INFO
level -- "HTTP Request: GET ... 206 Partial Content" -- and with the root at
INFO, that floods stdout with lines that look like errors (206/302 are
success codes) but are actually just normal, healthy download traffic. This
was happening in every script that called `logging.basicConfig` directly;
`configure_logging()` is the fix, applied once, everywhere.
"""

from __future__ import annotations

import logging

__all__ = ["configure_logging"]

# Libraries whose own INFO-level chatter is noise for our purposes, even
# though we want OUR script's INFO messages (progress, "wrote N rows", etc.)
# to show. Add to this list if a new noisy dependency shows up.
_NOISY_LOGGERS = ("httpx", "httpcore", "urllib3", "hf_xet", "filelock")


def configure_logging(
    level: int = logging.INFO, format: str = "%(message)s", **kwargs  # noqa: A002
) -> None:
    """Set up root logging for a CLI script while silencing known-noisy
    third-party HTTP/transport loggers.

    Parameter names deliberately match `logging.basicConfig` (including
    shadowing the `format` builtin) so callers can pass e.g. `stream=` or a
    custom `format=` straight through without a name collision.
    """
    logging.basicConfig(level=level, format=format, **kwargs)
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
