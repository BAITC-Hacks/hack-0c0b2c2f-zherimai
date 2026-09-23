"""Viewer payload must preserve identifiers, calculated requests, and safe text."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from aml.viewer import build_viewer


class ViewerPayloadTests(unittest.TestCase):
    def setUp(self):
        # Adjacent integers beyond JavaScript's exact range must stay distinct.
        self.gids = [100000000000000001 + i for i in range(4)]
        self.nodes = pd.DataFrame([
            {"gid": gid, "cluster_id": 1, "priority_score": priority,
             "role": role, "is_seed": seed, "truncated_by_depth": boundary,
             "unobserved_funding": False, "evidence": "Проверяемое основание 1"}
            for gid, priority, role, seed, boundary in zip(
                self.gids, [.9, .8, .7, .6],
                ["peripheral", "peripheral", "terminal", "transit"],
                [True, False, False, False], [False, True, False, False],
            )
        ])
        self.nodes["next_request"] = [
            "Проверить остаток на начало периода и невидимые входящие.",
            "Запросить исходящие следующего колена.",
            "Проверить наличные, межбанк и подпороговые операции.",
            "Проверить назначения и экономический смысл переводов.",
        ]
        self.edges = pd.DataFrame([{
            "src": self.gids[0], "dst": self.gids[1], "sum_kzt": 5000, "n_tx": 1,
        }])
        self.clusters = pd.DataFrame([{
            "cluster_id": 1, "n_nodes": 4, "n_seed": 1,
            "hypothesis": "Гипотеза для проверки",
        }])

    def build_payload(self):
        with TemporaryDirectory() as folder:
            output = Path(folder) / "network.html"
            self.assertEqual(build_viewer(self.nodes, self.edges, self.clusters, output), output)
            html = output.read_text(encoding="utf-8")
        opening = '<script id="graphData" type="application/json">'
        encoded = html.split(opening, 1)[1].split("</script>", 1)[0]
        return json.loads(encoded), encoded, html

    def test_large_identifiers_are_lossless_strings_everywhere(self):
        payload, _, _ = self.build_payload()
        expected = [str(gid) for gid in self.gids]
        self.assertEqual([node["gid"] for node in payload["nodes"]], expected)
        self.assertEqual(payload["edges"][0]["src"], expected[0])
        self.assertEqual(payload["edges"][0]["dst"], expected[1])
        self.assertTrue(all(node["cluster_id"] == "1" for node in payload["nodes"]))
        self.assertEqual(payload["clusters"][0]["cluster_id"], "1")

    def test_calculated_next_requests_are_preserved_verbatim(self):
        payload, _, _ = self.build_payload()
        actual = {node["gid"]: node["next_request"] for node in payload["nodes"]}
        expected = dict(zip(self.nodes.gid.map(str), self.nodes.next_request))
        self.assertEqual(actual, expected)

    def test_untrusted_text_cannot_close_embedded_json(self):
        hostile = '</script><script>alert("текст & данные")</script>'
        self.nodes.loc[0, "evidence"] = hostile
        self.nodes.loc[0, "next_request"] = hostile
        self.clusters.loc[0, "hypothesis"] = hostile
        payload, encoded, html = self.build_payload()
        self.assertEqual(payload["nodes"][0]["evidence"], hostile)
        self.assertEqual(payload["nodes"][0]["next_request"], hostile)
        self.assertEqual(payload["clusters"][0]["hypothesis"], hostile)
        self.assertNotIn(hostile, html)
        for character in "<>&":
            self.assertNotIn(character, encoded)


if __name__ == "__main__":
    unittest.main()
