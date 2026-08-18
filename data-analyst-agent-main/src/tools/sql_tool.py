import os
import re
import importlib
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import duckdb
from langchain_core.tools import tool

from src.logging_config import get_logger

log = get_logger("sql_tool")

# Default DB path when WAREHOUSE_DB env var is not set.
_DEFAULT_DB_PATH: Path = (
    Path(__file__).parent.parent / "data" / "sample" / "warehouse.db"
)

# Smart-city pipeline table names (S3 Parquet views)
_SMARTCITY_TABLES = [
    "vehicle_data",
    "gps_data",
    "traffic_data",
    "weather_data",
    "emergency_data",
]

# SQL safety guards
_ALLOWED_START = re.compile(r"^\s*(SELECT|WITH|EXPLAIN)\b", re.IGNORECASE)

_DANGEROUS_KEYWORDS = re.compile(
    r"\b(DROP|DELETE|INSERT|UPDATE|ALTER|TRUNCATE|REPLACE"
    r"|ATTACH|DETACH|COPY|EXPORT|IMPORT|PRAGMA|CALL|EXECUTE|LOAD|INSTALL)\b",
    re.IGNORECASE,
)

_OVERFLOW_PATTERN = re.compile(r"\bunit_price\s*\*\s*quantity\b", re.IGNORECASE)
_OVERFLOW_SAFE = "CAST(unit_price AS BIGINT) * quantity"

# Lowered from 100 → 20. Groq free tier caps at 12,000 tokens/minute;
# 100 rows of wide tables (14 columns) routinely pushed a single tool
# result past 2,000+ tokens, and with system prompt + message history
# this triggered HTTP 413 "tokens per minute" errors mid-conversation.
_ROW_LIMIT = 20

# Cap on characters returned by list_tables — DESCRIBE on 5 wide tables
# was another major contributor to the 413 errors, since the agent calls
# list_tables on nearly every request per the system prompt's QUERY RULES.
_LIST_TABLES_CHAR_LIMIT = 1200


def _sanitize(sql: str) -> tuple[bool, str, str]:
    """
    Validate and sanitize a SQL query.

    Returns:
        (ok, safe_sql, error_message)
        ok=True  → safe_sql ready to execute, error_message=""
        ok=False → safe_sql="", error_message describes the rejection reason

    Note: dangerous-keyword check runs first so that the keyword name
    always appears in the error message (e.g. "INSERT", "DROP").
    """
    stripped = sql.strip()
    if not stripped:
        return False, "", "SQL Error: query is empty."

    danger = _DANGEROUS_KEYWORDS.search(stripped)
    if danger:
        kw = danger.group(0).upper()
        return False, "", f"SQL Error: keyword '{kw}' is not permitted."

    if not _ALLOWED_START.match(stripped):
        return (
            False,
            "",
            "SQL Error: only SELECT / WITH / EXPLAIN queries are allowed.",
        )

    safe = _OVERFLOW_PATTERN.sub(_OVERFLOW_SAFE, stripped)
    return True, safe, ""


def _is_s3_mode() -> bool:
    """
    Return True when all required AWS env vars are present.
    Pure env check — no connection is established here.
    """
    return bool(
        os.getenv("AWS_ACCESS_KEY")
        and os.getenv("AWS_SECRET_KEY")
        and os.getenv("AWS_BUCKET_NAME")
    )


def _make_local_connection() -> "duckdb.DuckDBPyConnection":
    """
    Open a connection to the local DuckDB warehouse file.
    Path is resolved fresh from WAREHOUSE_DB env var on every call.

    Raises:
        FileNotFoundError: if the database file does not exist.
    """
    env = os.getenv("WAREHOUSE_DB")
    db_path = Path(env) if env else _DEFAULT_DB_PATH
    if not db_path.exists():
        raise FileNotFoundError(f"Local warehouse not found: {db_path}")
    # Import fresh to avoid being affected by test patches on global duckdb
    _duckdb = importlib.import_module("duckdb")
    return _duckdb.connect(str(db_path))


def _make_s3_connection() -> "duckdb.DuckDBPyConnection":
    """
    Open an in-memory DuckDB connection configured for S3 via httpfs,
    then create views for all five smart-city Parquet tables.

    Path pattern note: transform_parquet.py writes with
    `.partitionBy("date").parquet(out)`, which produces Hive-style
    partition subfolders, e.g.:
        refined/vehicle_data/date=2024-03-15/part-00000-xxx.snappy.parquet
    A glob of "refined/vehicle_data/*.parquet" only matches files directly
    inside that folder — it does NOT recurse into date=.../ subfolders, so
    it always reports zero files even when the partitioned data exists.
    Using "**/*.parquet" makes DuckDB's httpfs glob recurse through any
    number of partition subfolders.

    Validates that at least the vehicle_data table actually has Parquet
    files on S3 — CREATE VIEW + read_parquet() is lazy and succeeds even
    when the path is empty, so without this check a query_sql call would
    surface a raw "No files found" IO error to the LLM instead of
    falling back to the local warehouse.

    Raises:
        RuntimeError: if httpfs extension cannot be loaded, or if the
            refined S3 data is missing/empty (triggers fallback to local
            in _get_connection).
    """
    bucket = os.getenv("AWS_BUCKET_NAME", "")
    region = os.getenv("AWS_REGION", "ap-southeast-1")
    access_key = os.getenv("AWS_ACCESS_KEY", "")
    secret_key = os.getenv("AWS_SECRET_KEY", "")

    conn = duckdb.connect()

    try:
        conn.execute("INSTALL httpfs")
        conn.execute("LOAD httpfs")
    except Exception as exc:
        conn.close()
        raise RuntimeError(f"Cannot load httpfs: {exc}") from exc

    conn.execute(f"SET s3_region='{region}'")
    conn.execute(f"SET s3_access_key_id='{access_key}'")
    conn.execute(f"SET s3_secret_access_key='{secret_key}'")

    for table in _SMARTCITY_TABLES:
        # "**/*.parquet" recurses into Hive partition subfolders
        # (date=YYYY-MM-DD/) written by partitionBy("date") in Spark.
        s3_path = f"s3://{bucket}/refined/{table}/**/*.parquet"
        conn.execute(
            f"CREATE VIEW {table} AS "
            f"SELECT * FROM read_parquet('{s3_path}', hive_partitioning=true)"
        )

    # Validate refined data actually exists
    # read_parquet() is lazy: CREATE VIEW above succeeds even if the S3
    # prefix is empty. Probe one table now so an empty pipeline triggers
    # RuntimeError → fallback to local, instead of a confusing IO error
    # surfacing mid-conversation from query_sql.
    try:
        conn.execute("SELECT 1 FROM vehicle_data LIMIT 1").fetchall()
    except Exception as exc:
        conn.close()
        raise RuntimeError(
            f"No refined data found at s3://{bucket}/refined/ "
            f"(has the pipeline been run yet?): {exc}"
        ) from exc

    return conn


@contextmanager
def _get_connection() -> Generator["duckdb.DuckDBPyConnection", None, None]:
    """
    Context manager that yields the appropriate DuckDB connection.

    Priority:
      1. S3 via httpfs  (when AWS creds present AND refined data exists)
      2. Local DuckDB   (fallback if S3 fails, is empty, or no creds)

    Raises:
        ConnectionError: if both S3 and local connections fail.
    """
    conn = None
    try:
        if _is_s3_mode():
            try:
                conn = _make_s3_connection()
            except RuntimeError as exc:
                log.warning(
                    "s3_connection_fallback",
                    extra={"reason": str(exc)[:300]},
                )
                conn = _make_local_connection()
        else:
            conn = _make_local_connection()

        yield conn

    except (FileNotFoundError, OSError) as exc:
        raise ConnectionError(f"Both S3 and local connections failed: {exc}") from exc

    finally:
        if conn is not None:
            conn.close()


@tool
def list_tables() -> str:
    """
    List all available tables in the database
    along with their column names and types.
    """
    try:
        with _get_connection() as conn:
            tables_df = conn.execute("SHOW TABLES").fetchdf()
            if tables_df.empty:
                return "No tables found in database."

            result: list[str] = []
            for table in tables_df["name"].tolist():
                cols = conn.execute(f"DESCRIBE {table}").fetchdf()
                col_info = ", ".join(
                    f"{row['column_name']} ({row['column_type']})"
                    for _, row in cols.iterrows()
                )
                result.append(f"Table '{table}': {col_info}")

            output = "\n".join(result)

            # Token-budget guard: DESCRIBE on 5 wide tables can exceed
            # 1000+ tokens on its own. The agent calls list_tables on
            # almost every request (per system prompt), so this single
            # tool result was a major contributor to 413 TPM errors.
            if len(output) > _LIST_TABLES_CHAR_LIMIT:
                output = output[:_LIST_TABLES_CHAR_LIMIT] + "\n...(truncated)"

            return output

    except ConnectionError as exc:
        return f"Connection Error: {exc}"
    except Exception as exc:
        return f"Error: {exc}"


@tool
def query_sql(sql: str) -> str:
    """
    Execute a SQL query on the database and return results as a string.

    Only SELECT / WITH / EXPLAIN statements are accepted.
    Revenue must be calculated as: CAST(unit_price AS BIGINT) * quantity
    Results are capped at 20 rows — write aggregate queries (COUNT, AVG,
    GROUP BY) rather than relying on this tool to summarize raw rows.

    Args:
        sql: A valid SELECT query string.
    """
    ok, safe_sql, err = _sanitize(sql)
    if not ok:
        return err

    try:
        with _get_connection() as conn:
            result_df = conn.execute(safe_sql).fetchdf()

            if result_df.empty:
                return "Query returned no results."

            total = len(result_df)
            if total > _ROW_LIMIT:
                return (
                    f"Showing {_ROW_LIMIT} of {total} rows "
                    f"(narrow your query with WHERE/GROUP BY/LIMIT for full data):\n"
                    + result_df.head(_ROW_LIMIT).to_string(index=False)
                )

            return result_df.to_string(index=False)

    except ConnectionError as exc:
        return f"Connection Error: {exc}"
    except Exception as exc:
        return f"SQL Error: {exc}"
