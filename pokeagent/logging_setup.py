"""Per-invocation process log: every run leaves a file under ``runs/logs/``.

All framework logging (``pokeagent.*``, poke-env player logs, provider
warnings, unhandled exceptions) goes to console as before, but is ALSO
appended to a timestamped file so a finished battle can be debugged
offline. ``setup_logging`` is idempotent — calling it again swaps the file
handler instead of stacking duplicates.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import AgentConfig

_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def setup_logging(config: "AgentConfig") -> Path:
    """Attach a file handler (plus a WARNING console handler) to the root logger.

    Returns the log file path: ``<runs_dir>/logs/agent-<UTC timestamp>.log``.
    """
    log_dir = Path(config.runs_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"agent-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.log"

    root = logging.getLogger()
    # drop file handlers we added earlier (idempotency across repeated calls)
    for handler in [h for h in root.handlers if getattr(h, "_pokeagent_log", False)]:
        root.removeHandler(handler)

    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(_FORMAT))
    file_handler.setLevel(getattr(logging, config.log_level.upper(), logging.INFO))
    file_handler._pokeagent_log = True  # marker so the next call can find ours
    root.addHandler(file_handler)

    if not any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in root.handlers
    ):
        console = logging.StreamHandler()  # stderr
        console.setLevel(logging.WARNING)
        console.setFormatter(logging.Formatter(_FORMAT))
        root.addHandler(console)

    root.setLevel(logging.INFO)
    # third-party chatter we never want in the file
    for noisy in ("websockets", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.getLogger("poke_env").setLevel(
        getattr(logging, config.poke_env_log_level.upper(), logging.WARNING)
    )
    return path
