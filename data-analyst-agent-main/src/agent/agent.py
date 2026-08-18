import os
import re
import json
import time
import uuid

from dotenv import load_dotenv
from rich.console import Console
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryError,
)

from src.logging_config import get_logger

load_dotenv()

console = Console()
log = get_logger("agent")

MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "5"))

# Optional langchain imports (module level, so they can be patched)
# Moved to module scope — previously these were imported inside
# create_analyst_agent(), which meant `patch("src.agent.agent.ChatGroq", ...)`
# raised AttributeError because the attribute didn't exist on the module
# until the function actually ran.
try:
    from langchain_groq import ChatGroq
    from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

    _LANGCHAIN_AVAILABLE = True
except ImportError:
    ChatGroq = None  # type: ignore[assignment]
    HumanMessage = SystemMessage = ToolMessage = None  # type: ignore[assignment]
    _LANGCHAIN_AVAILABLE = False


# System prompts

_SYSTEM_PROMPT_SMARTCITY = """You are a Data Analyst assistant for the Smart City pipeline \
(HCM → Vũng Tàu expressway). You analyze real-time IoT data using SQL.

DATABASE: SmartCity — 5 tables on S3 (Parquet, partitioned by date)

TABLE SCHEMAS:
1. vehicle_data
   - id (VARCHAR), vehicle_id (VARCHAR)
   - timestamp (TIMESTAMP), date (VARCHAR) — partition key, format 'YYYY-MM-DD'
   - location (VARCHAR), speed (DOUBLE), speed_kmh (DOUBLE)
   - direction (VARCHAR), make (VARCHAR), model (VARCHAR)
   - year (INTEGER), fuelType (VARCHAR) — values: 'Gasoline', 'Electric', 'Hybrid'
   - hour (INTEGER) — 0–23
   - is_ev (BOOLEAN) — true if fuelType = 'Electric'

2. weather_data
   - id (VARCHAR), vehicle_id (VARCHAR)
   - timestamp (TIMESTAMP), date (VARCHAR)
   - location (VARCHAR), temperature (DOUBLE) — Celsius, tropical range 25–38
   - weatherCondition (VARCHAR) — 'Sunny', 'Cloudy', 'Rain', 'Stormy'
   - precipitation (DOUBLE), windSpeed (DOUBLE), humidity (INTEGER) — 60–95%
   - airQualityIndex (DOUBLE) — 0–200
   - heat_index (DOUBLE), aqi_category (VARCHAR) — 'Good','Moderate','Unhealthy for Sensitive','Unhealthy'

3. emergency_data
   - id (VARCHAR), vehicle_id (VARCHAR), incidentId (VARCHAR)
   - timestamp (TIMESTAMP), date (VARCHAR)
   - type (VARCHAR) — 'Accident', 'Fire', 'Medical', 'Police', 'None'
   - status (VARCHAR) — 'Active', 'Resolved'
   - location (VARCHAR), description (VARCHAR)
   - is_active (BOOLEAN), is_real_incident (BOOLEAN) — false when type = 'None'

4. gps_data
   - id (VARCHAR), vehicle_id (VARCHAR)
   - timestamp (TIMESTAMP), date (VARCHAR)
   - speed (DOUBLE), direction (VARCHAR)
   - vehicleType (VARCHAR) — e.g. 'private', 'bus'
   - hour (INTEGER)

5. traffic_data
   - id (VARCHAR), vehicle_id (VARCHAR), camera_id (VARCHAR)
   - timestamp (TIMESTAMP), date (VARCHAR)
   - location (VARCHAR)

QUERY RULES:
- ALWAYS call list_tables first to confirm available columns before querying
- The "date" column is a VARCHAR partition key in 'YYYY-MM-DD' format (e.g. '2026-06-17'),
  NOT a DATE type. NEVER compare it to CURRENT_DATE directly — that comparison can
  silently fail or behave unpredictably across DuckDB versions.
  ALWAYS write: WHERE date = CAST(CURRENT_DATE AS VARCHAR)
  or, for a specific day: WHERE date = '2026-06-17'
- If a query for "today" / "hôm nay" returns 0 rows, do NOT immediately conclude
  there is no data — re-run with WHERE date = (SELECT MAX(date) FROM <table>)
  to check the most recent available date, and tell the user which date you used.
- For EV queries: WHERE is_ev = true  OR  WHERE fuelType = 'Electric'
- For real incidents only: WHERE is_real_incident = true AND type != 'None'
- Aggregate speed with AVG(speed_kmh) not AVG(speed)
- JOIN vehicle_data and weather_data on vehicle_id AND date
- NEVER use columns that don't exist in the schema above
- PREFER aggregate queries (COUNT, AVG, SUM, GROUP BY) over SELECT * — raw row
  dumps are expensive and get truncated by query_sql's row limit
- Respond in the same language as the user (Vietnamese if user writes Vietnamese)
- Format numbers clearly: use ROUND() for decimals, format large numbers readably"""

_SYSTEM_PROMPT_LOCAL = """You are a Data Analyst assistant. \
You analyze sales data using SQL on a local DuckDB database.

DATABASE: Local warehouse — 1 table

TABLE SCHEMA:
1. sales
   - month (VARCHAR) — e.g. '2024-01'
   - city (VARCHAR) — 'Hanoi', 'HCMC', 'Danang'
   - product (VARCHAR) — 'Laptop', 'Phone', 'Tablet'
   - unit_price (INTEGER) — price per unit in VND
   - quantity (INTEGER) — number of units sold
   - Revenue = CAST(unit_price AS BIGINT) * quantity  ← NO revenue column, must calculate

QUERY RULES:
- ALWAYS call list_tables first before querying
- Revenue MUST be: CAST(unit_price AS BIGINT) * quantity  (avoids INT32 overflow)
- NEVER write: unit_price * quantity
- Respond in the same language as the user (Vietnamese if user writes Vietnamese)"""


def _get_system_prompt() -> str:
    """Chọn system prompt phù hợp với mode hiện tại."""
    has_aws = bool(
        os.getenv("AWS_ACCESS_KEY")
        and os.getenv("AWS_SECRET_KEY")
        and os.getenv("AWS_BUCKET_NAME")
    )
    return _SYSTEM_PROMPT_SMARTCITY if has_aws else _SYSTEM_PROMPT_LOCAL


# Rate limit handling

try:
    from groq import RateLimitError as _GroqRateLimitError

    _RETRY_ON = (_GroqRateLimitError,)
except ImportError:
    _RETRY_ON = (Exception,)  # type: ignore[assignment]


def _is_rate_limit(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        isinstance(exc, _RETRY_ON)
        or "429" in msg
        or "rate limit" in msg
        or "rate_limit" in msg
    )


# Aliases for testability
_is_rate_limit_error = _is_rate_limit


def _extract_fallback_tool_call(error_str: str) -> tuple[str | None, dict | None]:
    """Parse Groq XML-style tool call từ error string. Hỗ trợ có/không có quotes."""
    match = re.search(r"'?<function=(\w+)\s*(\{.*?\})\s*</function>'?", error_str)
    if match:
        tool_name = match.group(1)
        try:
            tool_args = json.loads(match.group(2))
            return tool_name, tool_args
        except json.JSONDecodeError:
            pass
    return None, None


# Agent factory


def create_analyst_agent(tools: list):
    """
    Factory trả về closure run_agent(user_input) -> str.

    Args:
        tools: List LangChain tool objects. Thường là [query_sql, list_tables].

    Raises:
        RuntimeError: nếu langchain-groq không khả dụng.
    """
    # Guard: kiểm tra langchain-groq có sẵn không
    if not _LANGCHAIN_AVAILABLE:
        raise RuntimeError(
            "langchain-groq is required but not available. "
            "Install: pip install langchain-groq"
        )

    system_prompt = _get_system_prompt()
    mode = "SmartCity/S3" if "vehicle_data" in system_prompt else "Local/DuckDB"
    log.info("Agent initialized in %s mode", mode)
    console.print(f"[dim]Agent mode: [cyan]{mode}[/cyan][/dim]")

    llm = ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        temperature=0,
    ).bind_tools(tools, tool_choice="auto")

    tool_map: dict = {t.name: t for t in tools}

    # Retry wrapper
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(_RETRY_ON),
        reraise=True,
    )
    def _invoke_llm(messages: list):
        return llm.invoke(messages)

    # Main loop
    def run_agent(user_input: str) -> str:
        req_id = str(uuid.uuid4())[:8]
        t_start = time.perf_counter()

        log.info(
            "request_start",
            extra={"req_id": req_id, "input_preview": user_input[:200]},
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_input),
        ]

        for step in range(MAX_STEPS):
            # LLM call
            try:
                response = _invoke_llm(messages)
            except RetryError as exc:
                log.error(
                    "llm_retry_exhausted", extra={"req_id": req_id, "error": str(exc)}
                )
                return "Lỗi: Groq API rate limit, đã thử lại 3 lần. Vui lòng thử sau."
            except Exception as exc:
                error_str = str(exc)
                log.warning(
                    "llm_error_fallback",
                    extra={"req_id": req_id, "error": error_str[:300]},
                )

                # Groq XML-style fallback
                tool_name, tool_args = _extract_fallback_tool_call(error_str)
                if tool_name and tool_name in tool_map:
                    console.print(
                        f"[dim]→ Fallback tool: {tool_name}({tool_args})[/dim]"
                    )
                    try:
                        result = tool_map[tool_name].invoke(tool_args)
                        messages.append(
                            HumanMessage(
                                content=(
                                    f"Tool {tool_name} returned: {result}\n"
                                    "Please summarize this result for the user."
                                )
                            )
                        )
                        continue
                    except Exception as tool_exc:
                        return f"Lỗi tool: {tool_exc}"

                return f"Lỗi gọi API: {error_str}"

            messages.append(response)

            # Final answer
            if not response.tool_calls:
                elapsed_ms = round((time.perf_counter() - t_start) * 1000)
                log.info(
                    "request_done",
                    extra={
                        "req_id": req_id,
                        "steps": step + 1,
                        "latency_ms": elapsed_ms,
                    },
                )
                # Trả fallback nếu content rỗng
                content = response.content
                if not content or not content.strip():
                    return "Xin lỗi, không có phản hồi từ model. Vui lòng thử lại."
                return content

            # Execute tool calls
            for tc in response.tool_calls:
                t_name: str = tc["name"]
                t_args: dict = tc["args"]

                console.print(
                    f"[dim]→ Tool: [magenta]{t_name}[/magenta]({t_args})[/dim]"
                )
                log.info(
                    "tool_call",
                    extra={
                        "req_id": req_id,
                        "step": step,
                        "tool": t_name,
                        "args_preview": str(t_args)[:200],
                    },
                )

                if t_name in tool_map:
                    try:
                        result = tool_map[t_name].invoke(t_args)
                        log.info(
                            "tool_result",
                            extra={
                                "req_id": req_id,
                                "tool": t_name,
                                "success": "Error" not in str(result),
                                "result_preview": str(result)[:200],
                            },
                        )
                    except Exception as tool_exc:
                        result = f"Tool error: {tool_exc}"
                        log.error(
                            "tool_exception",
                            extra={
                                "req_id": req_id,
                                "tool": t_name,
                                "error": str(tool_exc),
                            },
                        )
                else:
                    result = f"Tool '{t_name}' không tồn tại."
                    log.warning(
                        "unknown_tool", extra={"req_id": req_id, "tool": t_name}
                    )

                messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

        elapsed_ms = round((time.perf_counter() - t_start) * 1000)
        log.warning(
            "max_iterations_reached",
            extra={"req_id": req_id, "latency_ms": elapsed_ms},
        )
        return f"Không thể hoàn thành sau {MAX_STEPS} vòng."

    return run_agent
