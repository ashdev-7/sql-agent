import logging
import json
import os
import time

# Standard logging.LogRecord keys — used to filter out the app's extra fields
_STDLIB_ATTRS: frozenset = frozenset(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()
    | {"message", "asctime"}
)


class _JsonFormatter(logging.Formatter):
    """
    Format each log record as a single JSON line.

    Extra fields passed via log.info("msg", extra={"key": "val"})
    are merged straight into record.__dict__ by Python logging —
    the formatter picks them out by excluding _STDLIB_ATTRS.
    """

    def format(self, record: logging.LogRecord) -> str:
        # Call super to populate record.message, record.asctime, etc.
        record.message = record.getMessage()

        payload: dict = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.message,
        }

        # Extra fields: every key in __dict__ that is not a stdlib attr
        app_extra = {k: v for k, v in record.__dict__.items() if k not in _STDLIB_ATTRS}
        if app_extra:
            payload.update(app_extra)

        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False, default=str)


def get_logger(name: str) -> logging.Logger:
    """
    Return a logger with the JSON formatter.
    Idempotent: repeated calls with the same name → same instance, no duplicate handlers.

    Usage:
        log = get_logger("agent")
        log.info("tool_call", extra={"req_id": "abc", "tool": "query_sql"})
    """
    logger = logging.getLogger(name)

    # Idempotent: only set up the first time
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(_JsonFormatter())
        logger.addHandler(handler)
        logger.propagate = False

    level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)
    logger.setLevel(level)

    return logger
