"""Explanations retain material evidence and actionable observation limits."""
from types import SimpleNamespace
import unittest
import pandas as pd
from aml.explain import add_explanations, next_request, top_nodes


class ExplanationTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame([dict(gid=123, role="coordinator", base_role="consolidator",
            matched_roles="coordinator;consolidator;distributor", role_score=.8,
            priority_score=.7, structural_neighbors=4, seed_reach=7, betweenness=.02,
            temporal_seed_reach_strict_days=0, temporal_seed_reach_upper=1,
            in_deg=6, out_deg=99, in_kzt=90000., out_kzt=23000000., in_tx=10, out_tx=130,
            depth=2, is_seed=False, truncated_by_depth=False, unobserved_funding=True,
            priority_role=.3, priority_reach=.15, priority_volume=.12, priority_pagerank=.08,
            priority_patterns=.05, seed_discount=1.)])

    def test_primary_role_does_not_hide_large_distribution(self):
        result = top_nodes(add_explanations(self.frame())).iloc[0]
        self.assertIn("99 получателей", result.why)
        self.assertIn("23,000,000 KZT", result.why)
        self.assertIn("consolidator;distributor", result.why)
        self.assertIn("остаток на начало", result.why)
        self.assertIn("Слагаемые приоритета", result.why)
        self.assertIn("до 1 при допустимом порядке", result.why)
        self.assertIn("не происхождение денег", result.why)

    def test_boundary_request_precedes_balance_interpretation(self):
        row = SimpleNamespace(truncated_by_depth=True, is_seed=False, unobserved_funding=True, role="peripheral")
        self.assertIn("следующего колена", next_request(row))

    def test_evidence_contract_stays_short(self):
        result = add_explanations(self.frame())
        self.assertLessEqual(len(result.iloc[0].evidence), 200)
        self.assertTrue(result.iloc[0].next_request)


if __name__ == "__main__":
    unittest.main()
