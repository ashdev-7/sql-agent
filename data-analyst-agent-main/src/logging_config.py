import logging
import json
import os
import time

# Keys chuẩn của logging.LogRecord — dùng để lọc ra extra của app
_STDLIB_ATTRS: frozenset = frozenset(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()
    | {"message", "asctime"}
)


class _JsonFormatter(logging.Formatter):
    """
    Format mỗi log record thành một dòng JSON.

    Extra fields được truyền qua log.info("msg", extra={"key": "val"})
    sẽ được merge thẳng vào record.__dict__ bởi Python logging —
    formatter lọc ra bằng cách loại _STDLIB_ATTRS.
    """

    def format(self, record: logging.LogRecord) -> str:
        # Gọi super để populate record.message, record.asctime, v.v.
        record.message = record.getMessage()

        payload: dict = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.message,
        }

        # Lọc extra fields: tất cả key trong __dict__ mà không phải stdlib attr
        app_extra = {k: v for k, v in record.__dict__.items() if k not in _STDLIB_ATTRS}
        if app_extra:
            payload.update(app_extra)

        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False, default=str)


def get_logger(name: str) -> logging.Logger:
    """
    Trả về logger với JSON formatter.
    Idempotent: gọi nhiều lần cùng name → cùng instance, handler không bị thêm trùng.

    Usage:
        log = get_logger("agent")
        log.info("tool_call", extra={"req_id": "abc", "tool": "query_sql"})
    """
    logger = logging.getLogger(name)

    # Idempotent: chỉ setup lần đầu
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(_JsonFormatter())
        logger.addHandler(handler)
        logger.propagate = False

    level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)
    logger.setLevel(level)

    return logger
