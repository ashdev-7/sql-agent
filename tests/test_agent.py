import os
import sys
import unittest
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# conftest must be imported first so the stubs are registered
import tests.conftest  # noqa: F401
from tests.conftest import STUBS

RetryError = STUBS["RetryError"]
ToolMessage = STUBS["ToolMessage"]


# ── Helper factories ───────────────────────────────────────────────────────


def _make_response(content: str = "", tool_calls: list | None = None) -> MagicMock:
    """Build a mock that looks like a LangChain AIMessage."""
    r = MagicMock()
    r.content = content
    r.tool_calls = tool_calls or []
    return r


def _make_tool_call(name: str, args: dict, call_id: str = "tc_001") -> dict:
    return {"name": name, "args": args, "id": call_id}


def _make_tool(name: str, return_value: str = "ok") -> MagicMock:
    t = MagicMock()
    t.name = name
    t.invoke.return_value = return_value
    return t


def _build_llm(responses: list) -> MagicMock:
    """
    Build a mock ChatGroq instance.
    bind_tools() returns itself.
    invoke() returns the items in responses one by one.
    If an item is an Exception instance → raise it.
    """
    llm = MagicMock()
    llm.bind_tools.return_value = llm

    def _invoke_side_effect(messages):
        if not responses:
            raise StopIteration("mock exhausted")
        item = responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    llm.invoke.side_effect = _invoke_side_effect
    return llm


# ── Fixtures ───────────────────────────────────────────────────────────────


def _patch_groq(llm_mock):
    """Context manager: patch ChatGroq and return llm_mock."""
    return patch("src.agent.agent.ChatGroq", return_value=llm_mock)


# ══════════════════════════════════════════════════════════════════════════════
# Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestDirectAnswer(unittest.TestCase):
    """LLM answers immediately without calling a tool."""

    def test_returns_llm_content(self):
        llm = _build_llm([_make_response(content="Q1 revenue is 5 billion.")])
        with _patch_groq(llm):
            from src.agent.agent import create_analyst_agent

            agent = create_analyst_agent([])
        result = agent("total Q1 revenue?")
        self.assertIn("5 billion", result)

    def test_exact_content_returned(self):
        llm = _build_llm([_make_response(content="Result: 42")])
        with _patch_groq(llm):
            from src.agent.agent import create_analyst_agent

            agent = create_analyst_agent([])
        result = agent("any question")
        self.assertEqual(result, "Result: 42")

    def test_empty_tool_list_works(self):
        llm = _build_llm([_make_response(content="No data.")])
        with _patch_groq(llm):
            from src.agent.agent import create_analyst_agent

            agent = create_analyst_agent([])
        result = agent("question that needs no tool")
        self.assertEqual(result, "No data.")

    def test_english_response_preserved(self):
        llm = _build_llm([_make_response(content="The answer is 100.")])
        with _patch_groq(llm):
            from src.agent.agent import create_analyst_agent

            agent = create_analyst_agent([])
        result = agent("what is the total?")
        self.assertEqual(result, "The answer is 100.")


class TestEmptyContentResponse(unittest.TestCase):
    """Groq quirk: returns empty content with no tool call."""

    def test_empty_content_returns_fallback_message(self):
        llm = _build_llm([_make_response(content="", tool_calls=[])])
        with _patch_groq(llm):
            from src.agent.agent import create_analyst_agent

            agent = create_analyst_agent([])
        result = agent("some question")
        self.assertIsInstance(result, str)
        self.assertTrue(result)  # must not return an empty string

    def test_whitespace_content_treated_as_empty(self):
        llm = _build_llm([_make_response(content="   \n  ", tool_calls=[])])
        with _patch_groq(llm):
            from src.agent.agent import create_analyst_agent

            agent = create_analyst_agent([])
        result = agent("question")
        self.assertIsInstance(result, str)
        self.assertTrue(result.strip())


class TestToolCallFlow(unittest.TestCase):
    """LLM calls a tool one or more times."""

    def test_tool_invoked_with_correct_args(self):
        tool = _make_tool("query_sql", return_value="city | revenue\nBoston | 5000000")
        tc = _make_tool_call("query_sql", {"sql": "SELECT city FROM sales"})
        llm = _build_llm(
            [
                _make_response(tool_calls=[tc]),
                _make_response(content="Boston revenue is 5 million."),
            ]
        )
        with _patch_groq(llm):
            from src.agent.agent import create_analyst_agent

            agent = create_analyst_agent([tool])
        result = agent("Boston revenue?")
        tool.invoke.assert_called_once_with({"sql": "SELECT city FROM sales"})
        self.assertIn("5 million", result)

    def test_tool_result_included_in_next_llm_call(self):
        """The ToolMessage with the result must be sent to the LLM on the next step."""
        tool = _make_tool("list_tables", return_value="Table 'sales': month, city")
        tc = _make_tool_call("list_tables", {}, call_id="tc_lt")
        llm = _build_llm(
            [
                _make_response(tool_calls=[tc]),
                _make_response(content="There is a sales table with columns month, city."),
            ]
        )
        with _patch_groq(llm):
            from src.agent.agent import create_analyst_agent

            agent = create_analyst_agent([tool])
        result = agent("which tables are there?")

        # The 2nd invoke must receive the ToolMessage in messages
        second_call_messages = llm.invoke.call_args_list[1][0][0]
        tool_msgs = [m for m in second_call_messages if isinstance(m, ToolMessage)]
        self.assertTrue(len(tool_msgs) > 0)
        self.assertIn("Table 'sales'", tool_msgs[0].content)

    def test_multiple_tool_calls_in_one_step(self):
        tool_a = _make_tool("list_tables", return_value="Table 'sales'")
        tool_b = _make_tool("query_sql", return_value="rows: 6")
        tc_a = _make_tool_call("list_tables", {}, call_id="tc_a")
        tc_b = _make_tool_call(
            "query_sql", {"sql": "SELECT COUNT(*) FROM sales"}, call_id="tc_b"
        )
        llm = _build_llm(
            [
                _make_response(tool_calls=[tc_a, tc_b]),
                _make_response(content="The sales table has 6 rows."),
            ]
        )
        with _patch_groq(llm):
            from src.agent.agent import create_analyst_agent

            agent = create_analyst_agent([tool_a, tool_b])
        result = agent("how many rows?")
        tool_a.invoke.assert_called_once()
        tool_b.invoke.assert_called_once()
        self.assertIn("6", result)

    def test_tool_call_id_matched_in_tool_message(self):
        tool = _make_tool("query_sql", return_value="data")
        tc = _make_tool_call("query_sql", {"sql": "SELECT 1"}, call_id="my_call_id")
        llm = _build_llm(
            [
                _make_response(tool_calls=[tc]),
                _make_response(content="Done."),
            ]
        )
        with _patch_groq(llm):
            from src.agent.agent import create_analyst_agent

            agent = create_analyst_agent([tool])
        agent("test")
        second_call_messages = llm.invoke.call_args_list[1][0][0]
        tool_msgs = [m for m in second_call_messages if isinstance(m, ToolMessage)]
        self.assertEqual(tool_msgs[0].tool_call_id, "my_call_id")

    def test_two_step_tool_calls(self):
        """LLM calls a tool in step 1, another tool in step 2, then answers."""
        tool_a = _make_tool("list_tables", return_value="sales")
        tool_b = _make_tool("query_sql", return_value="total: 100")
        tc_a = _make_tool_call("list_tables", {})
        tc_b = _make_tool_call("query_sql", {"sql": "SELECT SUM(quantity) FROM sales"})
        llm = _build_llm(
            [
                _make_response(tool_calls=[tc_a]),
                _make_response(tool_calls=[tc_b]),
                _make_response(content="The total is 100."),
            ]
        )
        with _patch_groq(llm):
            from src.agent.agent import create_analyst_agent

            agent = create_analyst_agent([tool_a, tool_b])
        result = agent("total quantity?")
        self.assertIn("100", result)
        self.assertEqual(tool_a.invoke.call_count, 1)
        self.assertEqual(tool_b.invoke.call_count, 1)


class TestUnknownTool(unittest.TestCase):
    """LLM calls a tool that is not in tool_map."""

    def test_unknown_tool_does_not_raise(self):
        tc = _make_tool_call("ghost_tool", {})
        llm = _build_llm(
            [
                _make_response(tool_calls=[tc]),
                _make_response(content="Tool does not exist."),
            ]
        )
        with _patch_groq(llm):
            from src.agent.agent import create_analyst_agent

            agent = create_analyst_agent([])
        result = agent("call a ghost tool?")
        self.assertIsInstance(result, str)

    def test_unknown_tool_error_in_tool_message(self):
        """The ToolMessage for an unknown tool must contain an error message."""
        tc = _make_tool_call("ghost_tool", {}, call_id="tc_ghost")
        llm = _build_llm(
            [
                _make_response(tool_calls=[tc]),
                _make_response(content="Done."),
            ]
        )
        with _patch_groq(llm):
            from src.agent.agent import create_analyst_agent

            agent = create_analyst_agent([])
        agent("?")
        second_messages = llm.invoke.call_args_list[1][0][0]
        tool_msgs = [m for m in second_messages if isinstance(m, ToolMessage)]
        self.assertTrue(any("ghost_tool" in m.content for m in tool_msgs))


class TestMaxIterations(unittest.TestCase):
    """Agent must stop after MAX_STEPS steps and return a fallback message."""

    def test_returns_fallback_after_max_steps(self):
        from src.agent.agent import MAX_STEPS

        tool = _make_tool("query_sql", return_value="ok")
        tc = _make_tool_call("query_sql", {"sql": "SELECT 1"})
        # Enough responses for MAX_STEPS steps (one invoke per step)
        llm = _build_llm([_make_response(tool_calls=[tc])] * MAX_STEPS)
        with _patch_groq(llm):
            from src.agent.agent import create_analyst_agent

            agent = create_analyst_agent([tool])
        result = agent("infinite loop")
        self.assertIn(str(MAX_STEPS), result)
        self.assertIn("steps", result)

    def test_tool_called_at_most_max_steps_times(self):
        from src.agent.agent import MAX_STEPS

        tool = _make_tool("query_sql", return_value="ok")
        tc = _make_tool_call("query_sql", {"sql": "SELECT 1"})
        llm = _build_llm([_make_response(tool_calls=[tc])] * MAX_STEPS)
        with _patch_groq(llm):
            from src.agent.agent import create_analyst_agent

            agent = create_analyst_agent([tool])
        agent("loop")
        self.assertLessEqual(tool.invoke.call_count, MAX_STEPS)

    def test_fallback_message_contains_max_steps_constant(self):
        """Fallback must use the MAX_STEPS constant, not a hardcoded number."""
        from src.agent import agent as agent_module

        original_max = agent_module.MAX_STEPS
        agent_module.MAX_STEPS = 2  # Temporary override

        tool = _make_tool("query_sql", return_value="ok")
        tc = _make_tool_call("query_sql", {"sql": "SELECT 1"})
        llm = _build_llm([_make_response(tool_calls=[tc])] * 3)
        with _patch_groq(llm):
            from src.agent.agent import create_analyst_agent

            agent = create_analyst_agent([tool])
        result = agent("test")
        agent_module.MAX_STEPS = original_max  # Restore

        self.assertIn("2", result)

    def test_does_not_raise_stop_iteration(self):
        """Mock responses run out → StopIteration must not escape."""
        from src.agent.agent import MAX_STEPS

        tool = _make_tool("query_sql", return_value="ok")
        tc = _make_tool_call("query_sql", {"sql": "SELECT 1"})
        # Only MAX_STEPS - 1 responses, so the last call raises StopIteration
        llm = _build_llm([_make_response(tool_calls=[tc])] * (MAX_STEPS - 1))
        with _patch_groq(llm):
            from src.agent.agent import create_analyst_agent

            agent = create_analyst_agent([tool])
        # Must not raise an exception
        try:
            result = agent("test stop iteration")
            self.assertIsInstance(result, str)
        except StopIteration:
            self.fail("StopIteration must not propagate out of run_agent()")


class TestRateLimitRetry(unittest.TestCase):
    """Groq 429 → RetryError → friendly message."""

    def test_retry_error_returns_friendly_message(self):
        llm = _build_llm([RetryError()])
        with _patch_groq(llm):
            from src.agent.agent import create_analyst_agent

            agent = create_analyst_agent([])
        result = agent("any question at all")
        lower = result.lower()
        self.assertTrue("rate limit" in lower or "groq" in lower or "try again" in lower)

    def test_rate_limit_message_is_not_empty(self):
        llm = _build_llm([RetryError()])
        with _patch_groq(llm):
            from src.agent.agent import create_analyst_agent

            agent = create_analyst_agent([])
        result = agent("test")
        self.assertIsInstance(result, str)
        self.assertTrue(result.strip())


class TestIsRateLimitError(unittest.TestCase):
    """_is_rate_limit_error must correctly recognise each exception type."""

    def setUp(self):
        from src.agent.agent import _is_rate_limit_error

        self.fn = _is_rate_limit_error

    def test_exception_with_429_in_message(self):
        self.assertTrue(self.fn(Exception("HTTP 429 Too Many Requests")))

    def test_exception_with_rate_limit_in_message(self):
        self.assertTrue(self.fn(Exception("rate limit exceeded")))

    def test_exception_with_rate_limit_mixed_case(self):
        self.assertTrue(self.fn(Exception("Rate_Limit_Error occurred")))

    def test_generic_exception_returns_false(self):
        self.assertFalse(self.fn(Exception("connection timeout")))

    def test_value_error_returns_false(self):
        self.assertFalse(self.fn(ValueError("invalid input")))

    def test_runtime_error_returns_false(self):
        self.assertFalse(self.fn(RuntimeError("db error")))

    def test_keyboard_interrupt_returns_false(self):
        """KeyboardInterrupt must not be retried."""
        self.assertFalse(self.fn(KeyboardInterrupt()))

    def test_system_exit_returns_false(self):
        """SystemExit must not be retried."""
        self.assertFalse(self.fn(SystemExit(0)))


class TestFallbackXmlParsing(unittest.TestCase):
    """Groq quirk: tool call trong exception string."""

    def setUp(self):
        from src.agent.agent import _extract_fallback_tool_call

        self.fn = _extract_fallback_tool_call

    def test_standard_xml_format(self):
        err = 'Failed: \'<function=query_sql {"sql": "SELECT 1"} </function>\''
        name, args = self.fn(err)
        self.assertEqual(name, "query_sql")
        self.assertEqual(args, {"sql": "SELECT 1"})

    def test_xml_without_quotes(self):
        err = "<function=list_tables {} </function>"
        name, args = self.fn(err)
        self.assertEqual(name, "list_tables")
        self.assertEqual(args, {})

    def test_no_space_before_brace(self):
        """Groq sometimes omits the space between the function name and '{'."""
        err = '<function=query_sql{"sql": "SELECT 1"}</function>'
        name, args = self.fn(err)
        self.assertEqual(name, "query_sql")
        self.assertEqual(args, {"sql": "SELECT 1"})

    def test_extra_whitespace_handled(self):
        err = '<function=query_sql   {"sql": "SELECT 1"}  </function>'
        name, args = self.fn(err)
        self.assertIsNotNone(name)

    def test_invalid_json_returns_none(self):
        err = "<function=query_sql {not valid json} </function>"
        name, args = self.fn(err)
        self.assertIsNone(name)
        self.assertIsNone(args)

    def test_no_xml_returns_none(self):
        name, args = self.fn("some random error message")
        self.assertIsNone(name)
        self.assertIsNone(args)

    def test_empty_string_returns_none(self):
        name, args = self.fn("")
        self.assertIsNone(name)
        self.assertIsNone(args)

    def test_tool_executed_from_exception(self):
        """Full flow: exception with XML → tool is run → LLM summarises."""
        tool = _make_tool("query_sql", return_value="rows: 3")
        xml_err = Exception(
            '<function=query_sql {"sql": "SELECT COUNT(*) FROM sales"} </function>'
        )
        llm = _build_llm([xml_err, _make_response(content="There are 3 rows.")])
        with _patch_groq(llm):
            from src.agent.agent import create_analyst_agent

            agent = create_analyst_agent([tool])
        result = agent("count rows in sales")
        tool.invoke.assert_called_once()
        self.assertIn("3", result)

    def test_unknown_tool_in_exception_returns_error(self):
        xml_err = Exception('<function=nonexistent {"x": 1} </function>')
        llm = _build_llm([xml_err])
        with _patch_groq(llm):
            from src.agent.agent import create_analyst_agent

            agent = create_analyst_agent([])
        result = agent("trigger unknown fallback")
        self.assertIsInstance(result, str)
        self.assertTrue(result)


class TestToolException(unittest.TestCase):
    """tool.invoke() raises → agent does not crash, returns an error message."""

    def test_tool_exception_handled_gracefully(self):
        broken_tool = MagicMock()
        broken_tool.name = "query_sql"
        broken_tool.invoke.side_effect = RuntimeError("DuckDB connection failed")

        tc = _make_tool_call("query_sql", {"sql": "SELECT 1"})
        llm = _build_llm(
            [
                _make_response(tool_calls=[tc]),
                _make_response(content="An error occurred."),
            ]
        )
        with _patch_groq(llm):
            from src.agent.agent import create_analyst_agent

            agent = create_analyst_agent([broken_tool])
        result = agent("failing query")
        self.assertIsInstance(result, str)
        self.assertTrue(result)

    def test_tool_exception_error_passed_to_llm(self):
        """A tool error must be sent to the LLM as a ToolMessage."""
        broken_tool = MagicMock()
        broken_tool.name = "query_sql"
        broken_tool.invoke.side_effect = ValueError("invalid query")

        tc = _make_tool_call("query_sql", {"sql": "SELECT ?"}, call_id="err_tc")
        llm = _build_llm(
            [
                _make_response(tool_calls=[tc]),
                _make_response(content="Error."),
            ]
        )
        with _patch_groq(llm):
            from src.agent.agent import create_analyst_agent

            agent = create_analyst_agent([broken_tool])
        agent("test")

        second_messages = llm.invoke.call_args_list[1][0][0]
        tool_msgs = [m for m in second_messages if isinstance(m, ToolMessage)]
        self.assertTrue(
            any(
                "Tool error" in m.content or "invalid query" in m.content
                for m in tool_msgs
            )
        )


class TestApiError(unittest.TestCase):
    """LLM raises a generic Exception → agent returns an error message without crashing."""

    def test_api_error_returns_error_string(self):
        llm = _build_llm([Exception("connection timeout")])
        with _patch_groq(llm):
            from src.agent.agent import create_analyst_agent

            agent = create_analyst_agent([])
        result = agent("test api error")
        self.assertIsInstance(result, str)
        self.assertTrue(result)

    def test_api_error_message_contains_info(self):
        llm = _build_llm([Exception("HTTP 500 Internal Server Error")])
        with _patch_groq(llm):
            from src.agent.agent import create_analyst_agent

            agent = create_analyst_agent([])
        result = agent("test")
        # There must be some error message
        self.assertIsInstance(result, str)
        self.assertTrue(result.strip())


class TestCreateAnalystAgent(unittest.TestCase):
    """create_analyst_agent factory behavior."""

    def test_returns_callable(self):
        llm = _build_llm([])
        with _patch_groq(llm):
            from src.agent.agent import create_analyst_agent

            agent = create_analyst_agent([])
        self.assertTrue(callable(agent))

    def test_bind_tools_called_with_tool_list(self):
        tool_a = _make_tool("query_sql")
        tool_b = _make_tool("list_tables")
        llm = _build_llm([])
        with _patch_groq(llm):
            from src.agent.agent import create_analyst_agent

            create_analyst_agent([tool_a, tool_b])
        llm.bind_tools.assert_called_once()
        bound_tools = llm.bind_tools.call_args[0][0]
        self.assertEqual(len(bound_tools), 2)

    def test_tool_choice_auto(self):
        llm = _build_llm([])
        with _patch_groq(llm):
            from src.agent.agent import create_analyst_agent

            create_analyst_agent([_make_tool("t")])
        _, kwargs = llm.bind_tools.call_args
        self.assertEqual(kwargs.get("tool_choice"), "auto")

    def test_factory_not_available_raises(self):
        with patch("src.agent.agent._LANGCHAIN_AVAILABLE", False):
            with patch("src.agent.agent.ChatGroq", None):
                from src.agent.agent import create_analyst_agent

                with self.assertRaises(RuntimeError) as ctx:
                    create_analyst_agent([])
                self.assertIn("langchain-groq", str(ctx.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
