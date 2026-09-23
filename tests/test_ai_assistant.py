"""Spend, privacy, and execution boundaries for the optional semantic planner."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import http.client
import io
import json
import os
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import urllib.error

import pandas as pd

from aml.ai_assistant import AIAnalystAssistant, ENDPOINT, UsageLedger, QuotaExceeded


GID_A = "100000000000000001"
GID_B = "100000000000000002"
GID_C = "100000000000000003"


class LocalStub:
    def __init__(self):
        self.nodes = pd.DataFrame(index=[GID_A, GID_B, GID_C])
        self.executed = []

    def get_node(self, gid):
        self.executed.append(("get_node", gid))
        return {"gid": gid, "role": "consolidator", "evidence": "5 плательщиков; 100000 KZT."}

    def paths(self, source, target):
        self.executed.append(("paths", source, target))
        return {"path": [source, GID_C, target], "hops": 2}

    def common_receivers(self, gids):
        self.executed.append(("common_receivers", gids))
        return {"source_gids": gids, "common_receiver_gids": [GID_C]}

    def cluster_summary(self, cluster_id):
        self.executed.append(("cluster_summary", cluster_id))
        return {"cluster_id": cluster_id, "n_nodes": 3, "n_seed": 1,
                "sum_kzt_internal": 100000, "top_gids": GID_A + ";" + GID_B,
                "hypothesis": "Признаки сбора средств."}

    def answer(self, question):
        self.executed.append(("offline", question))
        return {"mode": "offline_deterministic", "tool": "help", "result": {"message": "local only"}}


def completed(action="get_node", refs=None, cluster_id=None):
    return {
        "status": "completed",
        "output": [{"type": "function_call", "name": "plan_query", "status": "completed",
                    "arguments": json.dumps({"action": action,
                                             "node_refs": ["node_1"] if refs is None else refs,
                                             "cluster_id": cluster_id})}],
        "usage": {"input_tokens": 121, "output_tokens": 37, "total_tokens": 158,
                  "input_tokens_details": {"cached_tokens": 0},
                  "output_tokens_details": {"reasoning_tokens": 12}},
    }


class FakeResponse(io.BytesIO):
    pass


class PlannerTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.local = LocalStub()
        self.db = Path(self.temp.name) / "usage.sqlite"
        self.planner = AIAnalystAssistant(assistant=self.local, usage_db=self.db)
        self.env = patch.dict(os.environ, {"OPENAI_API_KEY": "fake-test-only-secret", "OPENAI_MODEL": "test-model"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def answer(self, response=None, question=None, **kwargs):
        with patch("aml.ai_assistant._http_request", return_value=completed() if response is None else response) as http:
            result = self.planner.answer(question or "Объясни " + GID_A, online=True, **kwargs)
        return result, http

    def test_default_is_offline_even_with_key_and_model(self):
        with patch("aml.ai_assistant._http_request") as http:
            result = self.planner.answer("Объясни " + GID_A)
        self.assertEqual(result["fallback_reason"], "online_not_enabled")
        http.assert_not_called()
        self.assertFalse(self.db.exists())

    def test_missing_key_or_model_never_uses_network(self):
        for missing in ("OPENAI_API_KEY", "OPENAI_MODEL"):
            with self.subTest(missing=missing), patch.dict(os.environ, {missing: ""}):
                result, http = self.answer()
                self.assertTrue(result["fallback_reason"].startswith("missing_"))
                http.assert_not_called()

    def test_aliases_replace_repeated_ids_and_only_local_values_are_executed(self):
        question = f"Как связаны {GID_A} и {GID_B}? Начни с {GID_A}."
        result, http = self.answer(completed("paths", ["node_1", "node_2"]), question)
        payload = http.call_args.args[0]
        sent = json.dumps(payload)
        self.assertNotIn(GID_A, sent)
        self.assertNotIn(GID_B, sent)
        self.assertNotIn(GID_C, sent)
        self.assertNotIn("100000 KZT", sent)
        parsed = json.loads(payload["input"])
        self.assertEqual(parsed["question"], "Как связаны node_1 и node_2? Начни с node_1.")
        self.assertEqual(parsed["available_node_refs"], ["node_1", "node_2"])
        self.assertEqual(self.local.executed, [("paths", GID_A, GID_B)])
        self.assertEqual(result["result"]["path"], [GID_A, GID_C, GID_B])
        self.assertIn(GID_C, result["answer"])
        self.assertIn({"gid": GID_C, "source": "nodes_roles.csv", "key": "gid"}, result["citations"])
        self.assertEqual(result["plan"]["node_refs"], ["node_1", "node_2"])
        http.assert_called_once()

    def test_wire_request_is_fixed_bounded_and_strict(self):
        body = completed()
        with patch("aml.ai_assistant.urllib.request.build_opener") as build:
            build.return_value.open.return_value = FakeResponse(json.dumps(body).encode())
            result = self.planner.answer("Объясни " + GID_A, online=True)
        request = build.return_value.open.call_args.args[0]
        self.assertEqual(request.full_url, ENDPOINT)
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(build.return_value.open.call_args.kwargs["timeout"], 30)
        self.assertIsNone(build.call_args.args[0].redirect_request(None, None, 302, "", {}, "https://example.org"))
        payload = json.loads(request.data)
        self.assertFalse(payload["store"])
        self.assertEqual(payload["max_output_tokens"], 1024)
        self.assertFalse(payload["parallel_tool_calls"])
        schema = payload["tools"][0]["parameters"]
        self.assertTrue(payload["tools"][0]["strict"])
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(set(schema["required"]), set(schema["properties"]))
        self.assertEqual(schema["properties"]["node_refs"]["items"]["enum"], ["node_1"])
        self.assertEqual(result["evidence"][0]["gid"], GID_A)
        self.assertNotIn("fake-test-only-secret", json.dumps(result))

    def test_unknown_gid_and_reserved_alias_never_go_online(self):
        for question, reason in (("Объясни 999999999999999999", "unknown_local_gid"),
                                 (f"Объясни {GID_A} и node_9", "reserved_alias_in_question")):
            with self.subTest(question=question):
                result, http = self.answer(question=question)
                self.assertEqual(result["fallback_reason"], reason)
                http.assert_not_called()

    def test_input_caps_block_network(self):
        for question, reason in (("x" * 2001, "question_limit"),
                                 (" ".join(str(100000000000000000 + i) for i in range(9)), "node_reference_limit")):
            with self.subTest(reason=reason):
                result, http = self.answer(question=question)
                self.assertEqual(result["fallback_reason"], reason)
                http.assert_not_called()

    def test_all_supported_operations_execute_once(self):
        cases = (("get_node", ["node_1", "node_2"], None),
                 ("paths", ["node_1", "node_2"], None),
                 ("common_receivers", ["node_1", "node_2"], None),
                 ("cluster_summary", [], 1), ("help", [], None))
        for action, refs, cluster in cases:
            with self.subTest(action=action):
                result, http = self.answer(completed(action, refs, cluster), f"{GID_A} {GID_B} кластер 1")
                self.assertEqual(result["mode"], "online_semantic_plan_local_execution")
                self.assertEqual(result["tool"], action)
                http.assert_called_once()

    def test_injected_or_malformed_plan_does_not_execute_model_tool(self):
        mutations = [
            {"action": "paths", "node_refs": ["node_1", "node_99"], "cluster_id": None},
            {"action": "get_node", "node_refs": [GID_A], "cluster_id": None},
            {"action": "shell", "node_refs": [], "cluster_id": None},
            {"action": "get_node", "node_refs": ["node_1"], "cluster_id": None, "command": "bad"},
            {"action": "get_node", "node_refs": ["node_1", "node_1"], "cluster_id": None},
            {"action": "paths", "node_refs": ["node_1"], "cluster_id": None},
            {"action": "cluster_summary", "node_refs": [], "cluster_id": True},
            {"action": "cluster_summary", "node_refs": ["node_1"], "cluster_id": 1},
        ]
        for plan in mutations:
            with self.subTest(plan=plan):
                self.local.executed.clear()
                response = completed()
                response["output"][0]["arguments"] = json.dumps(plan)
                result, _ = self.answer(response)
                self.assertEqual(result["mode"], "offline_deterministic")
                self.assertEqual(self.local.executed, [("offline", "Объясни " + GID_A)])
                self.assertIsNone(result["plan"])

    def test_multiple_refused_incomplete_and_invalid_responses_fall_back(self):
        responses = [
            {"status": "completed", "output": completed()["output"] * 2},
            {"status": "incomplete", "output": completed()["output"]},
            {"status": "completed", "output": [{"type": "message", "content": [{"type": "refusal", "refusal": "no"}]}]},
            {"status": "completed", "output": [{"type": "message", "content": [{"type": "output_text", "text": "Made up"}]}]},
            {"status": "completed", "output": [{"type": "function_call", "name": "plan_query", "arguments": "{"}]},
            {"status": "completed", "output": None}, [],
        ]
        for response in responses:
            with self.subTest(response=response):
                result, _ = self.answer(response)
                self.assertEqual(result["mode"], "offline_deterministic")
                self.assertIn("fallback_reason", result)

    def test_network_errors_are_sanitized_and_do_not_retry(self):
        errors = [urllib.error.HTTPError(ENDPOINT, 401, "secret in reason", {"private": "secret"}, io.BytesIO(b"secret")),
                  urllib.error.URLError("secret connection details"), TimeoutError("secret")]
        for error in errors:
            with self.subTest(error=type(error).__name__), patch("aml.ai_assistant._http_request", side_effect=error) as http:
                result = self.planner.answer("Объясни " + GID_A, online=True)
                self.assertNotIn("secret", json.dumps(result))
                self.assertEqual(result["mode"], "offline_deterministic")
                http.assert_called_once()

    def test_http_protocol_errors_are_sanitized_and_consume_quota(self):
        errors = (http.client.IncompleteRead(b"secret response", 20),
                  http.client.BadStatusLine("secret status"))
        for index, error in enumerate(errors):
            planner = AIAnalystAssistant(assistant=self.local,
                                        usage_db=Path(self.temp.name) / f"protocol-{index}.sqlite", max_calls=1)
            with self.subTest(error=type(error).__name__), patch("aml.ai_assistant._http_request", side_effect=error) as request:
                first = planner.answer("Объясни " + GID_A, online=True)
                second = planner.answer("Объясни " + GID_A, online=True)
                self.assertEqual(first["fallback_reason"], "api_network_error")
                self.assertEqual(second["fallback_reason"], "request_limit_reached")
                self.assertNotIn("secret", json.dumps(first))
                request.assert_called_once()

    def test_common_receivers_cite_each_supporting_directed_edge(self):
        result, _ = self.answer(completed("common_receivers", ["node_1", "node_2"]),
                                f"Общие получатели {GID_A} и {GID_B}")
        edge_citations = [citation for citation in result["citations"] if citation["source"] == "edges.parquet"]
        self.assertEqual(edge_citations, [{"source": "edges.parquet", "src": GID_A, "dst": GID_C},
                                          {"source": "edges.parquet", "src": GID_B, "dst": GID_C}])

    def test_explicit_cluster_number_cannot_be_changed_by_model(self):
        questions = ("Кластер 1", "Сводка кластера №1", "Сообщество номер 1",
                     "Сводка сообщества 1", "Cluster #1", "community number 1")
        for question in questions:
            with self.subTest(question=question):
                self.local.executed.clear()
                result, _ = self.answer(completed("cluster_summary", [], 2), question)
                self.assertEqual(result["fallback_reason"], "cluster_id_mismatch")
                self.assertEqual(self.local.executed, [("offline", question)])
                accepted, _ = self.answer(completed("cluster_summary", [], 1), question)
                self.assertEqual(accepted["mode"], "online_semantic_plan_local_execution")

    def test_unrelated_numbers_and_gids_do_not_constrain_semantic_cluster(self):
        questions = ("Покажи сообщество номер два", "Сводка сообщества за 2 дня",
                     f"Сообщество {GID_A}", "Подскажи группу, у меня 3 вопроса")
        for question in questions:
            with self.subTest(question=question):
                result, _ = self.answer(completed("cluster_summary", [], 1), question)
                self.assertEqual(result["mode"], "online_semantic_plan_local_execution")

    def test_persistent_quota_counts_failed_requests_and_stops_before_network(self):
        self.planner = AIAnalystAssistant(assistant=self.local, usage_db=self.db, max_calls=1)
        with patch("aml.ai_assistant._http_request", side_effect=TimeoutError) as http:
            first = self.planner.answer("Объясни " + GID_A, online=True)
            fresh_process = AIAnalystAssistant(assistant=self.local, usage_db=self.db, max_calls=1)
            second = fresh_process.answer("Объясни " + GID_A, online=True)
        self.assertEqual(first["fallback_reason"], "api_timeout")
        self.assertEqual(second["fallback_reason"], "request_limit_reached")
        http.assert_called_once()
        with closing(sqlite3.connect(self.db)) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM calls").fetchone()[0], 1)

    def test_only_reported_usage_is_saved_without_questions_keys_or_model_output(self):
        response = completed()
        response["usage"]["invented_cost_usd"] = 123
        result, _ = self.answer(response)
        self.assertEqual(result["api_usage"]["output_tokens_details"]["reasoning_tokens"], 12)
        self.assertNotIn("invented_cost_usd", result["api_usage"])
        with closing(sqlite3.connect(self.db)) as db:
            row = db.execute("SELECT status, usage_json FROM calls").fetchone()
        self.assertEqual(row[0], "completed")
        self.assertEqual(json.loads(row[1]), result["api_usage"])
        raw = self.db.read_bytes()
        self.assertNotIn(GID_A.encode(), raw)
        self.assertNotIn(b"fake-test-only-secret", raw)

    def test_reservation_failure_is_offline(self):
        with patch.object(self.planner.ledger, "reserve", side_effect=sqlite3.OperationalError("private path")):
            result, http = self.answer()
        self.assertEqual(result["fallback_reason"], "usage_ledger_unavailable")
        http.assert_not_called()

    def test_no_usage_does_not_invent_zero_tokens(self):
        response = completed()
        response.pop("usage")
        result, _ = self.answer(response)
        self.assertIsNone(result["api_usage"])


class LedgerTests(unittest.TestCase):
    def test_concurrent_requests_cannot_exceed_quota(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "usage.sqlite"

            def reserve(_):
                try:
                    UsageLedger(path, max_calls=3).reserve()
                    return 1
                except QuotaExceeded:
                    return 0

            with ThreadPoolExecutor(max_workers=8) as pool:
                reserved = list(pool.map(reserve, range(12)))
            self.assertEqual(sum(reserved), 3)

    def test_hard_limit_and_invalid_caps(self):
        for cap in (0, -1, 201, 1000, True, 1.5):
            with self.subTest(cap=cap), self.assertRaises(ValueError):
                UsageLedger("unused.sqlite", max_calls=cap)


if __name__ == "__main__":
    unittest.main()
