import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import pandas as pd


# Helpers


def _make_mock_conn(
    tables: list[str] | None = None, describe_df: pd.DataFrame | None = None
):
    """Build a mock DuckDB connection whose fetchdf() returns data."""
    conn = MagicMock()

    if tables is not None:
        tables_df = pd.DataFrame({"table_name": tables, "name": tables})
        _default_df = MagicMock()
        _default_df.fetchdf.return_value = tables_df
        conn.execute.return_value = _default_df

    if describe_df is not None:
        _desc = MagicMock()
        _desc.fetchdf.return_value = describe_df
        conn.execute.return_value = _desc

    return conn


# _is_s3_mode


class TestIsS3Mode:
    def test_returns_true_when_all_aws_env_set(self, monkeypatch):
        monkeypatch.setenv("AWS_ACCESS_KEY", "key123")
        monkeypatch.setenv("AWS_SECRET_KEY", "secret123")
        monkeypatch.setenv("AWS_BUCKET_NAME", "my-bucket")

        import importlib
        import src.tools.sql_tool as m

        importlib.reload(m)

        assert m._is_s3_mode() is True

    def test_returns_false_when_bucket_missing(self, monkeypatch):
        monkeypatch.setenv("AWS_ACCESS_KEY", "key123")
        monkeypatch.setenv("AWS_SECRET_KEY", "secret123")
        monkeypatch.delenv("AWS_BUCKET_NAME", raising=False)

        import importlib
        import src.tools.sql_tool as m

        importlib.reload(m)

        assert m._is_s3_mode() is False

    def test_returns_false_when_no_aws_env(self, monkeypatch):
        for k in ["AWS_ACCESS_KEY", "AWS_SECRET_KEY", "AWS_BUCKET_NAME"]:
            monkeypatch.delenv(k, raising=False)

        import importlib
        import src.tools.sql_tool as m

        importlib.reload(m)

        assert m._is_s3_mode() is False


# _sanitize


class TestSanitize:
    @pytest.fixture(autouse=True)
    def _import(self):
        from src.tools.sql_tool import _sanitize

        self._sanitize = _sanitize

    def test_empty_sql_rejected(self):
        ok, _, err = self._sanitize("   ")
        assert ok is False
        assert "empty" in err.lower()

    def test_select_allowed(self):
        ok, safe, _ = self._sanitize("SELECT 1")
        assert ok is True
        assert "SELECT 1" in safe

    def test_with_allowed(self):
        ok, _, _ = self._sanitize("WITH cte AS (SELECT 1) SELECT * FROM cte")
        assert ok is True

    def test_explain_allowed(self):
        ok, _, _ = self._sanitize("EXPLAIN SELECT * FROM vehicle_data")
        assert ok is True

    def test_insert_rejected(self):
        ok, _, err = self._sanitize("INSERT INTO t VALUES (1)")
        assert ok is False
        assert "INSERT" in err

    def test_drop_rejected(self):
        ok, _, err = self._sanitize("DROP TABLE vehicle_data")
        assert ok is False
        assert "DROP" in err

    def test_delete_rejected(self):
        ok, _, err = self._sanitize("DELETE FROM vehicle_data")
        assert ok is False
        assert "DELETE" in err

    def test_update_rejected(self):
        ok, _, err = self._sanitize("UPDATE vehicle_data SET speed = 0")
        assert ok is False

    def test_install_rejected(self):
        ok, _, err = self._sanitize("INSTALL httpfs")
        assert ok is False
        assert "INSTALL" in err

    def test_load_rejected(self):
        ok, _, err = self._sanitize("LOAD httpfs")
        assert ok is False
        assert "LOAD" in err

    def test_overflow_rewritten(self):
        sql = "SELECT unit_price * quantity FROM sales"
        ok, safe, _ = self._sanitize(sql)
        assert ok is True
        assert "CAST(unit_price AS BIGINT) * quantity" in safe
        assert "unit_price * quantity" not in safe.replace(
            "CAST(unit_price AS BIGINT) * quantity", ""
        )

    def test_overflow_case_insensitive(self):
        sql = "SELECT UNIT_PRICE * QUANTITY FROM sales"
        ok, safe, _ = self._sanitize(sql)
        assert ok is True
        assert "CAST" in safe

    def test_safe_select_unchanged(self):
        sql = "SELECT make, AVG(speed_kmh) FROM vehicle_data GROUP BY make"
        ok, safe, _ = self._sanitize(sql)
        assert ok is True
        assert safe.strip() == sql.strip()

    def test_dangerous_keyword_mid_query_rejected(self):
        sql = "SELECT * FROM sales WHERE name = 'DROP'"
        # 'DROP' inside a string value — the current regex still catches it
        # Test documents current behaviour
        ok, _, _ = self._sanitize(sql)
        assert ok is False  # conservative: block anyway


# _make_local_connection


class TestMakeLocalConnection:
    def test_raises_file_not_found_when_db_missing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("WAREHOUSE_DB", str(tmp_path / "nonexistent.db"))

        import importlib
        import src.tools.sql_tool as m

        importlib.reload(m)

        with pytest.raises(FileNotFoundError, match="Local warehouse not found"):
            m._make_local_connection()

    def test_connects_when_db_exists(self, tmp_path, monkeypatch):
        db_path = tmp_path / "test.db"

        # 1. Create the DB file first (in a subprocess so the mocks don't affect it)
        import subprocess, sys

        subprocess.run(
            [
                sys.executable,
                "-c",
                f"import duckdb; c = duckdb.connect({str(db_path)!r}); "
                f"c.execute('CREATE TABLE test (id INTEGER)'); c.close()",
            ],
            check=True,
        )

        # 2. Verify the file exists
        assert db_path.exists(), f"DB file not created: {db_path}"

        # 3. Set env
        monkeypatch.setenv("WAREHOUSE_DB", str(db_path))

        # 4. Call it directly
        from src.tools import sql_tool

        conn = sql_tool._make_local_connection()
        assert conn is not None
        conn.close()


# _make_s3_connection


class TestMakeS3Connection:
    @pytest.fixture(autouse=True)
    def _set_aws_env(self, monkeypatch):
        monkeypatch.setenv("AWS_ACCESS_KEY", "test-key")
        monkeypatch.setenv("AWS_SECRET_KEY", "test-secret")
        monkeypatch.setenv("AWS_REGION", "ap-southeast-1")
        monkeypatch.setenv("AWS_BUCKET_NAME", "smartcity-bucket")

    def test_raises_runtime_error_when_httpfs_fails(self, monkeypatch):
        import importlib
        import src.tools.sql_tool as m

        importlib.reload(m)

        with patch("duckdb.connect") as mock_connect:
            mock_conn = MagicMock()
            mock_connect.return_value = mock_conn
            mock_conn.execute.side_effect = Exception("httpfs not available")

            with pytest.raises(RuntimeError, match="Cannot load httpfs"):
                m._make_s3_connection()

    def test_sets_s3_credentials(self, monkeypatch):
        import importlib
        import src.tools.sql_tool as m

        importlib.reload(m)

        execute_calls = []

        with patch("duckdb.connect") as mock_connect:
            mock_conn = MagicMock()
            mock_connect.return_value = mock_conn

            def _execute(sql):
                execute_calls.append(sql)
                return MagicMock()

            mock_conn.execute.side_effect = _execute

            try:
                m._make_s3_connection()
            except Exception:
                pass  # view creation might fail — that's ok for this test

            all_sql = " ".join(execute_calls)
            assert "test-key" in all_sql
            assert "test-secret" in all_sql
            assert "ap-southeast-1" in all_sql

    def test_creates_views_for_all_3_tables(self, monkeypatch):
        import importlib
        import src.tools.sql_tool as m

        importlib.reload(m)

        with patch("duckdb.connect") as mock_connect:
            mock_conn = MagicMock()
            mock_connect.return_value = mock_conn
            mock_conn.execute.return_value = MagicMock()

            m._make_s3_connection()

            view_calls = [
                str(c)
                for c in mock_conn.execute.call_args_list
                if "CREATE VIEW" in str(c)
            ]
            assert len(view_calls) == 3
            for table in m._SMARTCITY_TABLES:
                assert any(table in c for c in view_calls)


# _get_connection context manager


class TestGetConnection:
    def test_yields_s3_connection_when_aws_configured(self, monkeypatch):
        monkeypatch.setenv("AWS_ACCESS_KEY", "key")
        monkeypatch.setenv("AWS_SECRET_KEY", "secret")
        monkeypatch.setenv("AWS_BUCKET_NAME", "bucket")

        import importlib
        import src.tools.sql_tool as m

        importlib.reload(m)

        mock_conn = MagicMock()
        with patch.object(m, "_make_s3_connection", return_value=mock_conn) as mock_s3:
            with m._get_connection() as conn:
                assert conn is mock_conn
            mock_s3.assert_called_once()

    def test_yields_local_connection_when_no_aws(self, monkeypatch):
        for k in ["AWS_ACCESS_KEY", "AWS_SECRET_KEY", "AWS_BUCKET_NAME"]:
            monkeypatch.delenv(k, raising=False)

        import importlib
        import src.tools.sql_tool as m

        importlib.reload(m)

        mock_conn = MagicMock()
        with patch.object(
            m, "_make_local_connection", return_value=mock_conn
        ) as mock_local:
            with m._get_connection() as conn:
                assert conn is mock_conn
            mock_local.assert_called_once()

    def test_connection_closed_after_context_exits(self, monkeypatch):
        for k in ["AWS_ACCESS_KEY", "AWS_SECRET_KEY", "AWS_BUCKET_NAME"]:
            monkeypatch.delenv(k, raising=False)

        import importlib
        import src.tools.sql_tool as m

        importlib.reload(m)

        mock_conn = MagicMock()
        with patch.object(m, "_make_local_connection", return_value=mock_conn):
            with m._get_connection():
                pass

        mock_conn.close.assert_called_once()

    def test_s3_failure_falls_back_to_local(self, monkeypatch):
        monkeypatch.setenv("AWS_ACCESS_KEY", "key")
        monkeypatch.setenv("AWS_SECRET_KEY", "secret")
        monkeypatch.setenv("AWS_BUCKET_NAME", "bucket")

        import importlib
        import src.tools.sql_tool as m

        importlib.reload(m)

        mock_local_conn = MagicMock()
        with patch.object(
            m, "_make_s3_connection", side_effect=RuntimeError("httpfs fail")
        ):
            with patch.object(
                m, "_make_local_connection", return_value=mock_local_conn
            ) as mock_local:
                with m._get_connection() as conn:
                    assert conn is mock_local_conn
                mock_local.assert_called_once()

    def test_both_s3_and_local_fail_raises_connection_error(self, monkeypatch):
        monkeypatch.setenv("AWS_ACCESS_KEY", "key")
        monkeypatch.setenv("AWS_SECRET_KEY", "secret")
        monkeypatch.setenv("AWS_BUCKET_NAME", "bucket")

        import importlib
        import src.tools.sql_tool as m

        importlib.reload(m)

        with patch.object(
            m, "_make_s3_connection", side_effect=RuntimeError("s3 fail")
        ):
            with patch.object(
                m, "_make_local_connection", side_effect=FileNotFoundError("no db")
            ):
                with pytest.raises(ConnectionError, match="Both S3 and local"):
                    with m._get_connection():
                        pass


# list_tables tool


class TestListTables:
    def test_returns_table_list_in_local_mode(self, monkeypatch):
        for k in ["AWS_ACCESS_KEY", "AWS_SECRET_KEY", "AWS_BUCKET_NAME"]:
            monkeypatch.delenv(k, raising=False)

        import importlib
        import src.tools.sql_tool as m

        importlib.reload(m)

        mock_conn = MagicMock()
        tables_df = pd.DataFrame({"name": ["sales"], "table_name": ["sales"]})
        describe_df = pd.DataFrame(
            {
                "column_name": ["month", "city", "unit_price"],
                "column_type": ["VARCHAR", "VARCHAR", "INTEGER"],
            }
        )

        def _execute(sql):
            r = MagicMock()
            if "SHOW TABLES" in sql:
                r.fetchdf.return_value = tables_df
            elif "DESCRIBE" in sql:
                r.fetchdf.return_value = describe_df
            return r

        mock_conn.execute.side_effect = _execute

        with patch.object(m, "_get_connection") as mock_ctx:
            mock_ctx.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)

            result = m.list_tables.invoke({})

        assert "sales" in result
        assert "month" in result

    def test_returns_no_tables_message_when_empty(self, monkeypatch):
        for k in ["AWS_ACCESS_KEY", "AWS_SECRET_KEY", "AWS_BUCKET_NAME"]:
            monkeypatch.delenv(k, raising=False)

        import importlib
        import src.tools.sql_tool as m

        importlib.reload(m)

        mock_conn = MagicMock()
        empty_df = pd.DataFrame({"name": [], "table_name": []})
        r = MagicMock()
        r.fetchdf.return_value = empty_df
        mock_conn.execute.return_value = r

        with patch.object(m, "_get_connection") as mock_ctx:
            mock_ctx.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)

            result = m.list_tables.invoke({})

        assert "No tables" in result

    def test_returns_error_string_on_connection_error(self, monkeypatch):
        for k in ["AWS_ACCESS_KEY", "AWS_SECRET_KEY", "AWS_BUCKET_NAME"]:
            monkeypatch.delenv(k, raising=False)

        import importlib
        import src.tools.sql_tool as m

        importlib.reload(m)

        with patch.object(m, "_get_connection") as mock_ctx:
            mock_ctx.return_value.__enter__ = MagicMock(
                side_effect=ConnectionError("both failed")
            )
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)

            result = m.list_tables.invoke({})

        assert "Connection Error" in result or "Error" in result


# query_sql tool


class TestQuerySql:
    @pytest.fixture(autouse=True)
    def _no_aws(self, monkeypatch):
        for k in ["AWS_ACCESS_KEY", "AWS_SECRET_KEY", "AWS_BUCKET_NAME"]:
            monkeypatch.delenv(k, raising=False)

        import importlib
        import src.tools.sql_tool as m

        importlib.reload(m)
        self.m = m

    def _patch_conn(self, df: pd.DataFrame):
        mock_conn = MagicMock()
        r = MagicMock()
        r.fetchdf.return_value = df
        mock_conn.execute.return_value = r
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=mock_conn)
        ctx.__exit__ = MagicMock(return_value=False)
        return ctx

    def test_returns_formatted_dataframe(self):
        df = pd.DataFrame({"city": ["Boston", "Seattle"], "total": [100, 200]})
        with patch.object(self.m, "_get_connection", return_value=self._patch_conn(df)):
            result = self.m.query_sql.invoke({"sql": "SELECT city, total FROM sales"})
        assert "Boston" in result
        assert "Seattle" in result

    def test_empty_result_message(self):
        df = pd.DataFrame()
        with patch.object(self.m, "_get_connection", return_value=self._patch_conn(df)):
            result = self.m.query_sql.invoke({"sql": "SELECT * FROM sales WHERE 1=0"})
        assert "no results" in result.lower()

    def test_rejects_insert(self):
        result = self.m.query_sql.invoke({"sql": "INSERT INTO sales VALUES (1)"})
        assert "SQL Error" in result
        assert "INSERT" in result

    def test_rejects_drop(self):
        result = self.m.query_sql.invoke({"sql": "DROP TABLE sales"})
        assert "SQL Error" in result

    def test_rewrites_overflow(self):
        df = pd.DataFrame({"revenue": [1000000]})
        executed_sql = []

        mock_conn = MagicMock()
        r = MagicMock()
        r.fetchdf.return_value = df

        def _exec(sql):
            executed_sql.append(sql)
            return r

        mock_conn.execute.side_effect = _exec

        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=mock_conn)
        ctx.__exit__ = MagicMock(return_value=False)

        with patch.object(self.m, "_get_connection", return_value=ctx):
            self.m.query_sql.invoke({"sql": "SELECT unit_price * quantity FROM sales"})

        assert any("CAST(unit_price AS BIGINT)" in s for s in executed_sql)

    def test_row_limit_100(self):
        # 150 rows → only show _ROW_LIMIT rows (was hard-coded to 100;
        # now reads the live constant so this test tracks any future
        # change to _ROW_LIMIT instead of silently going stale)
        limit = self.m._ROW_LIMIT
        df = pd.DataFrame({"n": range(150)})
        with patch.object(self.m, "_get_connection", return_value=self._patch_conn(df)):
            result = self.m.query_sql.invoke({"sql": "SELECT n FROM big_table"})
        assert "150" in result  # mentions total count
        assert str(limit) in result  # mentions the actual configured limit

    def test_sql_exception_returns_error_string(self):
        mock_conn = MagicMock()
        mock_conn.execute.side_effect = Exception("column not found: xyz")
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=mock_conn)
        ctx.__exit__ = MagicMock(return_value=False)

        with patch.object(self.m, "_get_connection", return_value=ctx):
            result = self.m.query_sql.invoke({"sql": "SELECT xyz FROM sales"})

        assert "SQL Error" in result or "Error" in result
        assert "xyz" in result or "column" in result.lower()

    def test_connection_error_returns_friendly_message(self):
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(side_effect=ConnectionError("both failed"))
        ctx.__exit__ = MagicMock(return_value=False)

        with patch.object(self.m, "_get_connection", return_value=ctx):
            result = self.m.query_sql.invoke({"sql": "SELECT 1"})

        assert "Connection Error" in result


#  _SMARTCITY_TABLES constant


class TestSmartCityTablesConstant:
    def test_has_three_tables(self):
        from src.tools.sql_tool import _SMARTCITY_TABLES

        assert len(_SMARTCITY_TABLES) == 3

    def test_contains_all_expected_tables(self):
        from src.tools.sql_tool import _SMARTCITY_TABLES

        expected = {
            "vehicle_data",
            "weather_data",
            "emergency_data",
        }
        assert expected == set(_SMARTCITY_TABLES)
