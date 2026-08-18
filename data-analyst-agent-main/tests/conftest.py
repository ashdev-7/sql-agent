import sys
import os
import types
from unittest.mock import MagicMock

# Make project root + jobs/* importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "jobs"))
sys.path.insert(0, os.path.join(ROOT, "jobs", "ingestion"))
sys.path.insert(0, os.path.join(ROOT, "jobs", "streaming"))


# 1. confluent_kafka
confluent_stub = types.ModuleType("confluent_kafka")
confluent_stub.SerializingProducer = MagicMock
sys.modules.setdefault("confluent_kafka", confluent_stub)


# 2. dotenv
dotenv_stub = types.ModuleType("dotenv")
dotenv_stub.load_dotenv = lambda *a, **kw: None
sys.modules.setdefault("dotenv", dotenv_stub)


# 3. config module (used by spark_city.py / transform_parquet.py)
config_stub = types.ModuleType("config")
config_stub.configuration = {
    "AWS_ACCESS_KEY": "test-key",
    "AWS_SECRET_KEY": "test-secret",
    "AWS_REGION": "ap-southeast-1",
    "AWS_BUCKET_NAME": "smartcity-data-pipeline",
    "KAFKA_BOOTSTRAP_SERVERS": "broker:29092",
    "VEHICLE_TOPIC": "vehicle_data",
    "GPS_TOPIC": "gps_data",
    "TRAFFIC_TOPIC": "traffic_data",
    "WEATHER_TOPIC": "weather_data",
    "EMERGENCY_TOPIC": "emergency_data",
}
sys.modules.setdefault("config", config_stub)


# 4. groq (SDK, not the langchain wrapper)
groq_stub = types.ModuleType("groq")


class _FakeGroqRateLimitError(Exception):
    pass


groq_stub.RateLimitError = _FakeGroqRateLimitError
sys.modules.setdefault("groq", groq_stub)


# 5. tenacity
tenacity_stub = types.ModuleType("tenacity")


class _FakeRetryError(Exception):
    """Stub RetryError — mirrors tenacity.RetryError's interface."""

    def __init__(self, last_attempt=None):
        self.last_attempt = last_attempt or MagicMock()
        super().__init__(str(last_attempt))


def _noop_retry(**kwargs):
    """Stub @retry decorator — runs the function directly, no retry."""

    def decorator(fn):
        return fn

    return decorator


tenacity_stub.RetryError = _FakeRetryError
tenacity_stub.retry = _noop_retry
tenacity_stub.stop_after_attempt = lambda n: None
tenacity_stub.wait_exponential = lambda **kw: None
tenacity_stub.retry_if_exception = lambda fn: None
tenacity_stub.retry_if_exception_type = lambda *a: None
sys.modules.setdefault("tenacity", tenacity_stub)


# 6. rich
rich_stub = types.ModuleType("rich")
rich_console_stub = types.ModuleType("rich.console")
rich_panel_stub = types.ModuleType("rich.panel")


class _FakeConsole:
    def print(self, *a, **kw):
        pass

    def input(self, prompt=""):
        return ""


class _FakePanel:
    @staticmethod
    def fit(*a, **kw):
        return MagicMock()


rich_console_stub.Console = _FakeConsole
rich_panel_stub.Panel = _FakePanel
rich_stub.console = rich_console_stub
rich_stub.panel = rich_panel_stub
sys.modules.setdefault("rich", rich_stub)
sys.modules.setdefault("rich.console", rich_console_stub)
sys.modules.setdefault("rich.panel", rich_panel_stub)


# 7. langchain_core.messages / tools
lc_core = types.ModuleType("langchain_core")
lc_messages = types.ModuleType("langchain_core.messages")
lc_tools_mod = types.ModuleType("langchain_core.tools")


class _BaseMessage:
    def __init__(self, content="", **kwargs):
        self.content = content
        for k, v in kwargs.items():
            setattr(self, k, v)


class HumanMessage(_BaseMessage):
    pass


class SystemMessage(_BaseMessage):
    pass


class ToolMessage(_BaseMessage):
    def __init__(self, content="", tool_call_id="", **kwargs):
        super().__init__(content=content, **kwargs)
        self.tool_call_id = tool_call_id


lc_messages.HumanMessage = HumanMessage
lc_messages.SystemMessage = SystemMessage
lc_messages.ToolMessage = ToolMessage


def _tool_decorator(fn):
    """Stub @tool decorator."""
    fn.name = fn.__name__
    fn.invoke = lambda args: fn(**args) if isinstance(args, dict) else fn(args)
    return fn


lc_tools_mod.tool = _tool_decorator

lc_core.messages = lc_messages
sys.modules.setdefault("langchain_core", lc_core)
sys.modules.setdefault("langchain_core.messages", lc_messages)
sys.modules.setdefault("langchain_core.tools", lc_tools_mod)


# 8. langchain_groq
lc_groq = types.ModuleType("langchain_groq")


class _FakeChatGroq:
    def __init__(self, model="", temperature=0, **kwargs):
        self.model = model
        self.temperature = temperature
        self._tools = []

    def bind_tools(self, tools, **kwargs):
        self._tools = tools
        return self

    def invoke(self, messages):
        raise NotImplementedError("Mock this in your test")


lc_groq.ChatGroq = _FakeChatGroq
sys.modules.setdefault("langchain_groq", lc_groq)


# 9. duckdb
try:
    import duckdb  # noqa: F401
except ImportError:
    duckdb_stub = types.ModuleType("duckdb")

    class _FakeDuckDBCon:
        def __init__(self, *a, **kw):
            self._closed = False

        def execute(self, sql, params=None):
            return self

        def fetchdf(self):
            import pandas as pd

            return pd.DataFrame()

        def close(self):
            self._closed = True

    def _fake_connect(*a, **kw):
        return _FakeDuckDBCon()

    duckdb_stub.connect = _fake_connect
    sys.modules.setdefault("duckdb", duckdb_stub)


# Expose stubs for tests that need to reconfigure them
STUBS = {
    "RetryError": _FakeRetryError,
    "HumanMessage": HumanMessage,
    "SystemMessage": SystemMessage,
    "ToolMessage": ToolMessage,
    "GroqRateLimitError": _FakeGroqRateLimitError,
}
