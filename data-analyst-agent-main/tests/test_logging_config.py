import json
import logging
import os
import sys
import io
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.logging_config import get_logger, _JsonFormatter, _STDLIB_ATTRS


class TestStdlibAttrs(unittest.TestCase):
    """_STDLIB_ATTRS phải bao gồm các key chuẩn của LogRecord."""

    def test_contains_common_stdlib_keys(self):
        for key in (
            "name",
            "msg",
            "levelname",
            "levelno",
            "pathname",
            "filename",
            "module",
            "exc_info",
            "exc_text",
            "created",
            "thread",
            "process",
            "args",
        ):
            self.assertIn(key, _STDLIB_ATTRS, f"'{key}' phải trong _STDLIB_ATTRS")

    def test_is_frozenset(self):
        self.assertIsInstance(_STDLIB_ATTRS, frozenset)


class TestJsonFormatter(unittest.TestCase):
    """_JsonFormatter output phải là valid JSON với đúng fields."""

    def setUp(self):
        self.formatter = _JsonFormatter()

    def _make_record(self, msg="test", level=logging.INFO, extra=None):
        record = logging.LogRecord(
            name="test.logger",
            level=level,
            pathname="test.py",
            lineno=1,
            msg=msg,
            args=(),
            exc_info=None,
        )
        if extra:
            for k, v in extra.items():
                setattr(record, k, v)
        return record

    def test_output_is_valid_json(self):
        record = self._make_record("hello")
        output = self.formatter.format(record)
        data = json.loads(output)
        self.assertIsInstance(data, dict)

    def test_required_fields_present(self):
        record = self._make_record("hello")
        data = json.loads(self.formatter.format(record))
        for field in ("ts", "level", "logger", "msg"):
            self.assertIn(field, data, f"'{field}' phải có trong output")

    def test_ts_format(self):
        record = self._make_record()
        data = json.loads(self.formatter.format(record))
        ts = data["ts"]
        # Format: 2024-01-15T08:30:00Z
        self.assertRegex(ts, r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")

    def test_level_field(self):
        record = self._make_record(level=logging.WARNING)
        data = json.loads(self.formatter.format(record))
        self.assertEqual(data["level"], "WARNING")

    def test_logger_field(self):
        record = self._make_record()
        data = json.loads(self.formatter.format(record))
        self.assertEqual(data["logger"], "test.logger")

    def test_extra_fields_extracted(self):
        """Extra fields phải xuất hiện trong JSON output."""
        record = self._make_record(extra={"req_id": "abc123", "tool": "query_sql"})
        data = json.loads(self.formatter.format(record))
        self.assertEqual(data.get("req_id"), "abc123")
        self.assertEqual(data.get("tool"), "query_sql")

    def test_extra_fields_not_polluted_by_stdlib(self):
        """Stdlib attrs như 'name', 'levelno' không được xuất hiện như extra."""
        record = self._make_record()
        data = json.loads(self.formatter.format(record))
        # levelno, lineno, thread, process không nên lẫn vào output
        # (chúng nằm trong payload chỉ qua ts/level/logger/msg)
        self.assertNotIn("levelno", data)
        self.assertNotIn("lineno", data)
        self.assertNotIn("threadName", data)

    def test_exc_info_included(self):
        """Exception info phải được format và đưa vào field 'exc'."""
        try:
            raise ValueError("test error")
        except ValueError:
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg="error occurred",
            args=(),
            exc_info=exc_info,
        )
        data = json.loads(self.formatter.format(record))
        self.assertIn("exc", data)
        self.assertIn("ValueError", data["exc"])
        self.assertIn("test error", data["exc"])

    def test_non_serializable_object_uses_str_fallback(self):
        """Non-JSON-serializable extra phải dùng str() fallback."""
        record = self._make_record(extra={"obj": object()})
        # Không được raise exception
        output = self.formatter.format(record)
        data = json.loads(output)
        # obj phải xuất hiện dưới dạng string
        self.assertIn("obj", data)
        self.assertIsInstance(data["obj"], str)

    def test_unicode_ensure_ascii_false(self):
        """Tiếng Việt phải giữ nguyên, không escape thành \\uXXXX."""
        record = self._make_record(msg="Xin chào thế giới")
        output = self.formatter.format(record)
        self.assertIn("Xin chào thế giới", output)
        # ensure_ascii=False → không có \\u escape
        self.assertNotIn("\\u", output)

    def test_extra_int_and_float(self):
        record = self._make_record(extra={"latency_ms": 42, "score": 0.95})
        data = json.loads(self.formatter.format(record))
        self.assertEqual(data["latency_ms"], 42)
        self.assertAlmostEqual(data["score"], 0.95)

    def test_no_extra_produces_clean_output(self):
        record = self._make_record("clean message")
        data = json.loads(self.formatter.format(record))
        # Không có extra key nào ngoài 4 field chuẩn
        self.assertEqual(set(data.keys()), {"ts", "level", "logger", "msg"})


class TestGetLogger(unittest.TestCase):
    """get_logger phải idempotent và respect LOG_LEVEL."""

    def test_returns_logger_instance(self):
        logger = get_logger("test.module.a")
        self.assertIsInstance(logger, logging.Logger)

    def test_same_name_returns_same_instance(self):
        l1 = get_logger("test.module.b")
        l2 = get_logger("test.module.b")
        self.assertIs(l1, l2)

    def test_no_duplicate_handlers(self):
        """Gọi nhiều lần → vẫn chỉ có 1 handler."""
        name = "test.module.c"
        # Reset trước
        logging.getLogger(name).handlers.clear()
        get_logger(name)
        get_logger(name)
        get_logger(name)
        logger = logging.getLogger(name)
        self.assertEqual(len(logger.handlers), 1)

    def test_handler_is_stream_handler(self):
        name = "test.module.d"
        logging.getLogger(name).handlers.clear()
        get_logger(name)
        logger = logging.getLogger(name)
        self.assertIsInstance(logger.handlers[0], logging.StreamHandler)

    def test_formatter_is_json_formatter(self):
        name = "test.module.e"
        logging.getLogger(name).handlers.clear()
        get_logger(name)
        logger = logging.getLogger(name)
        self.assertIsInstance(logger.handlers[0].formatter, _JsonFormatter)

    def test_log_level_from_env_info(self):
        name = "test.module.f"
        logging.getLogger(name).handlers.clear()
        with patch.dict(os.environ, {"LOG_LEVEL": "INFO"}):
            logger = get_logger(name)
        self.assertEqual(logger.level, logging.INFO)

    def test_log_level_from_env_debug(self):
        name = "test.module.g"
        logging.getLogger(name).handlers.clear()
        with patch.dict(os.environ, {"LOG_LEVEL": "DEBUG"}):
            logger = get_logger(name)
        self.assertEqual(logger.level, logging.DEBUG)

    def test_log_level_invalid_defaults_to_info(self):
        name = "test.module.h"
        logging.getLogger(name).handlers.clear()
        with patch.dict(os.environ, {"LOG_LEVEL": "NOTAVALID"}):
            logger = get_logger(name)
        self.assertEqual(logger.level, logging.INFO)

    def test_propagate_false(self):
        name = "test.module.i"
        logging.getLogger(name).handlers.clear()
        logger = get_logger(name)
        self.assertFalse(logger.propagate)

    def test_actual_log_output_is_json(self):
        """Ghi log thật và capture output, kiểm tra là JSON hợp lệ."""
        name = "test.module.j"
        raw_logger = logging.getLogger(name)
        raw_logger.handlers.clear()

        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(_JsonFormatter())
        raw_logger.addHandler(handler)
        raw_logger.setLevel(logging.DEBUG)
        raw_logger.propagate = False

        raw_logger.info("test message", extra={"req_id": "xyz", "step": 3})

        output = stream.getvalue().strip()
        self.assertTrue(output, "Output phải có nội dung")
        data = json.loads(output)
        self.assertEqual(data["msg"], "test message")
        self.assertEqual(data["req_id"], "xyz")
        self.assertEqual(data["step"], 3)


class TestLoggerIntegration(unittest.TestCase):
    """Integration test: logger dùng được trong agent context."""

    def test_multiple_loggers_independent(self):
        l1 = get_logger("agent")
        l2 = get_logger("sql_tool")
        self.assertIsNot(l1, l2)
        self.assertNotEqual(l1.name, l2.name)

    def test_logger_captures_all_extra_types(self):
        name = "test.module.k"
        raw_logger = logging.getLogger(name)
        raw_logger.handlers.clear()
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(_JsonFormatter())
        raw_logger.addHandler(handler)
        raw_logger.setLevel(logging.DEBUG)
        raw_logger.propagate = False

        raw_logger.info(
            "complex extras",
            extra={
                "string_val": "hello",
                "int_val": 42,
                "float_val": 3.14,
                "bool_val": True,
                "list_val": [1, 2, 3],
            },
        )

        data = json.loads(stream.getvalue().strip())
        self.assertEqual(data["string_val"], "hello")
        self.assertEqual(data["int_val"], 42)
        self.assertAlmostEqual(data["float_val"], 3.14)
        self.assertTrue(data["bool_val"])
        self.assertEqual(data["list_val"], [1, 2, 3])


if __name__ == "__main__":
    unittest.main(verbosity=2)
