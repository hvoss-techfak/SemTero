"""Logging configuration helpers.

Goal: keep console output readable during long background embedding runs.

- Suppress noisy HTTP client per-request logs (httpx/httpcore/urllib3).
- Optionally suppress overly chatty third-party SDKs.

Call :func:`setup_logging` early in the entrypoint.
"""

from __future__ import annotations

import logging


_NOISY_LOGGERS: tuple[str, ...] = (
    # Common HTTP client libraries
    "httpx",
    "httpcore",
    "urllib3",
    "requests",
    # Ollama python client can be chatty depending on version
    "ollama",
)


def setup_logging(*, quiet_http: bool = True, quiet_http_level: int = logging.WARNING) -> None:
    """Apply opinionated logging defaults.

    This function only adjusts selected third-party loggers. It doesn't change
    the root logger configuration/handlers.

    Args:
        quiet_http: If True, raise log level for known HTTP loggers.
        quiet_http_level: The minimum level for noisy loggers.
    """

    if quiet_http:
        for name in _NOISY_LOGGERS:
            logging.getLogger(name).setLevel(quiet_http_level)

    # These can be very verbose at DEBUG in some environments.
    logging.getLogger("mcp").setLevel(logging.INFO)
    logging.getLogger("anyio").setLevel(logging.INFO)

