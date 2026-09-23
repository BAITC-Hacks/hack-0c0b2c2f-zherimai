"""Offline query contracts, without loading saved results or running the pipeline."""

import unittest
from unittest.mock import Mock

import networkx as nx
import pandas as pd

from aml.assistant import AnalystAssistant


class AssistantRoutingTests(unittest.TestCase):
    # Synthetic opaque identifiers exceed JavaScript's exact-integer range.
    SOURCE = "90071992547409931"
    SECOND = "90071992547409933"
    RECEIVER = "90071992547409935"
    ISOLATE = "90071992547409937"
    UNKNOWN = "90071992547409939"

    def setUp(self):
        self.assistant = AnalystAssistant.__new__(AnalystAssistant)
        gids = [self.SOURCE, self.SECOND, self.RECEIVER, self.ISOLATE]
        self.assistant.nodes = pd.DataFrame(
            {"gid": gids, "role": ["peripheral"] * len(gids)}
        ).set_index("gid")
        self.assistant.graph = nx.DiGraph()
        self.assistant.graph.add_nodes_from(int(gid) for gid in gids)
        self.assistant.graph.add_edges_from(
            [(int(self.SOURCE), int(self.RECEIVER)),
             (int(self.SECOND), int(self.RECEIVER))]
        )
        self.assistant.clusters = pd.DataFrame(
            [{"cluster_id": 7, "n_nodes": 3, "hypothesis": "Synthetic test cluster"}]
        )
        self.tools = {}
        for name in ("get_node", "paths", "common_receivers", "cluster_summary"):
            wrapped = Mock(wraps=getattr(self.assistant, name))
            setattr(self.assistant, name, wrapped)
            self.tools[name] = wrapped

    def answer(self, question):
        response = self.assistant.answer(question)
        self.assertEqual(response["mode"], "offline_deterministic")
        return response

    def reset_calls(self):
        for tool in self.tools.values():
            tool.reset_mock()

    def assert_no_tool_calls(self):
        for name, tool in self.tools.items():
            with self.subTest(unexpected_tool=name):
                tool.assert_not_called()

    def assert_explanation(self, response):
        self.assertNotEqual(response["tool"], "get_node")
        self.assertIsInstance(response["result"], dict)
        self.assertTrue(
            any(response["result"].get(key) for key in ("error", "message", "explanation")),
            "An incomplete or unsupported request needs an explanation",
        )

    def test_documented_russian_node_query(self):
        response = self.answer(f"Объясни {self.SOURCE}")
        self.assertEqual(response["tool"], "get_node")
        self.assertEqual(response["result"][0]["gid"], self.SOURCE)
        self.assertEqual(response["result"][0]["outgoing_gids"], [self.RECEIVER])
        self.tools["get_node"].assert_called_once_with(self.SOURCE)

    def test_bare_gid_preserves_exact_identifier(self):
        response = self.answer(self.SECOND)
        self.assertEqual(response["tool"], "get_node")
        self.assertEqual(response["result"][0]["gid"], self.SECOND)
        self.assertIsInstance(response["result"][0]["gid"], str)
        self.tools["get_node"].assert_called_once_with(self.SECOND)

    def test_node_query_accepts_trailing_period_and_whitespace(self):
        for gid in (self.SOURCE, self.UNKNOWN):
            for question in (f"Объясни {gid}.", f" \tОбъясни {gid}. \n"):
                with self.subTest(question=question):
                    self.reset_calls()
                    response = self.answer(question)
                    self.assertEqual(response["tool"], "get_node")
                    self.assertEqual(response["result"][0]["gid"], gid)
                    self.assertEqual(bool(response["result"][0].get("error")), gid == self.UNKNOWN)
                    self.tools["get_node"].assert_called_once_with(gid)

    def test_unicode_digits_cannot_alias_an_ascii_gid(self):
        variants = (
            self.SOURCE.translate(str.maketrans("0123456789", "０１２３４５６７８９")),
            self.SOURCE.translate(str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")),
            "１" + self.SOURCE,
            self.SOURCE + "１",
            "١" + self.SOURCE,
            self.SOURCE + "١",
        )
        for gid in variants:
            with self.subTest(gid=gid):
                self.reset_calls()
                response = self.answer(f"Объясни {gid}")
                self.assert_explanation(response)
                self.assert_no_tool_calls()

    def test_english_node_aliases(self):
        for command in ("explain", "node"):
            with self.subTest(command=command):
                self.reset_calls()
                response = self.answer(f"{command} {self.SOURCE}")
                self.assertEqual(response["tool"], "get_node")
                self.assertEqual(response["result"][0]["gid"], self.SOURCE)
                self.tools["get_node"].assert_called_once_with(self.SOURCE)

    def test_documented_russian_path_preserves_direction(self):
        response = self.answer(f"Путь от {self.SOURCE} до {self.RECEIVER}")
        self.assertEqual(response["tool"], "paths")
        self.assertEqual(response["result"]["path"], [self.SOURCE, self.RECEIVER])
        self.tools["paths"].assert_called_once_with(self.SOURCE, self.RECEIVER)

    def test_english_path_alias(self):
        response = self.answer(f"path from {self.SECOND} to {self.RECEIVER}")
        self.assertEqual(response["tool"], "paths")
        self.assertEqual(response["result"]["path"], [self.SECOND, self.RECEIVER])
        self.tools["paths"].assert_called_once_with(self.SECOND, self.RECEIVER)

    def test_no_path_is_explained(self):
        response = self.answer(f"Путь от {self.RECEIVER} до {self.SOURCE}")
        self.assertEqual(response["tool"], "paths")
        self.assertEqual(response["result"]["path"], [])
        self.assertTrue(response["result"].get("explanation"))

    def test_documented_common_receivers_query(self):
        response = self.answer(f"Общие получатели {self.SOURCE} и {self.SECOND}")
        self.assertEqual(response["tool"], "common_receivers")
        self.assertEqual(response["result"]["common_receiver_gids"], [self.RECEIVER])
        self.assertEqual(response["result"]["source_gids"], [self.SOURCE, self.SECOND])
        self.tools["common_receivers"].assert_called_once_with([self.SOURCE, self.SECOND])

    def test_english_common_receivers_alias(self):
        response = self.answer(f"common receivers {self.SOURCE} and {self.SECOND}")
        self.assertEqual(response["tool"], "common_receivers")
        self.assertEqual(response["result"]["common_receiver_gids"], [self.RECEIVER])

    def test_documented_and_english_cluster_queries(self):
        for question in ("Кластер 7", "Кластер #7", "cluster 7"):
            with self.subTest(question=question):
                self.reset_calls()
                response = self.answer(question)
                self.assertEqual(response["tool"], "cluster_summary")
                self.assertEqual(response["result"]["cluster_id"], 7)
                self.tools["cluster_summary"].assert_called_once_with(7)

    def test_unknown_gid_returns_error(self):
        response = self.answer(f"Объясни {self.UNKNOWN}")
        self.assertEqual(response["tool"], "get_node")
        self.assertTrue(response["result"][0].get("error"))
        self.assertEqual(response["result"][0]["gid"], self.UNKNOWN)

    def test_unknown_path_endpoint_returns_error(self):
        response = self.answer(f"Путь от {self.SOURCE} до {self.UNKNOWN}")
        self.assertEqual(response["tool"], "paths")
        self.assertTrue(response["result"].get("error"))
        self.assertNotIn("path", response["result"])

    def test_unknown_common_source_returns_error(self):
        response = self.answer(f"Общие получатели {self.SOURCE} и {self.UNKNOWN}")
        self.assertEqual(response["tool"], "common_receivers")
        self.assertTrue(response["result"].get("error"))

    def test_unknown_cluster_returns_error(self):
        response = self.answer("Кластер 8")
        self.assertEqual(response["tool"], "cluster_summary")
        self.assertTrue(response["result"].get("error"))

    def test_income_question_with_gid_is_unsupported(self):
        response = self.answer(f"Какой доход у клиента {self.SOURCE}?")
        self.assertEqual(response["tool"], "help")
        self.assert_explanation(response)
        self.assert_no_tool_calls()

    def test_arbitrary_text_with_gid_does_not_open_a_card(self):
        for question in (
            f"Какое финансовое состояние у {self.SOURCE}?",
            f"Случайный текст {self.SOURCE}",
            f"What is the income of {self.SOURCE}?",
            f"Объясни {self.SOURCE} и предскажи оборот",
        ):
            with self.subTest(question=question):
                self.reset_calls()
                response = self.answer(question)
                self.assertEqual(response["tool"], "help")
                self.assert_explanation(response)
                self.assert_no_tool_calls()

    def test_path_with_one_gid_explains_missing_endpoint(self):
        for question in (f"Путь от {self.SOURCE}", f"path from {self.SOURCE}"):
            with self.subTest(question=question):
                self.reset_calls()
                response = self.answer(question)
                self.assert_explanation(response)
                self.tools["get_node"].assert_not_called()
                self.tools["paths"].assert_not_called()

    def test_path_with_three_gids_does_not_choose_first_two(self):
        for question in (
            f"Путь от {self.SOURCE} до {self.RECEIVER} через {self.SECOND}",
            f"path {self.SOURCE} {self.RECEIVER} {self.SECOND}",
        ):
            with self.subTest(question=question):
                self.reset_calls()
                response = self.answer(question)
                self.assert_explanation(response)
                self.tools["get_node"].assert_not_called()
                self.tools["paths"].assert_not_called()

    def test_common_receivers_require_two_distinct_gids(self):
        for question in (
            f"Общие получатели {self.SOURCE}",
            f"Общие получатели {self.SOURCE} и {self.SOURCE}",
            f"common receivers {self.SOURCE}",
        ):
            with self.subTest(question=question):
                self.reset_calls()
                response = self.answer(question)
                self.assert_explanation(response)
                self.tools["get_node"].assert_not_called()
                self.assertNotIn("common_receiver_gids", response["result"])

    def test_negative_cluster_is_not_reinterpreted_as_positive(self):
        for question in ("Кластер -7", "Кластер #-7", "cluster -7"):
            with self.subTest(question=question):
                self.reset_calls()
                response = self.answer(question)
                self.assert_explanation(response)
                self.assertNotIn("cluster_id", response["result"])
                for call in self.tools["cluster_summary"].call_args_list:
                    self.assertEqual(str(call.args[0]), "-7")

    def test_malformed_cluster_number_is_not_truncated(self):
        for question in ("Кластер 7.5", "Кластер 7,5", "Кластер 7x", "Кластер abc"):
            with self.subTest(question=question):
                self.reset_calls()
                response = self.answer(question)
                self.assert_explanation(response)
                self.tools["cluster_summary"].assert_not_called()

    def test_empty_and_unsupported_questions_show_help(self):
        for question in ("", "   ", "Какая сегодня погода?"):
            with self.subTest(question=question):
                self.reset_calls()
                response = self.answer(question)
                self.assertEqual(response["tool"], "help")
                self.assert_explanation(response)
                self.assert_no_tool_calls()


if __name__ == "__main__":
    unittest.main()
