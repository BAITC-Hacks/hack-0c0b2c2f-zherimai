"""Behavioral checks for the dataset's declared traps, not accuracy claims."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import networkx as nx
import pandas as pd
from aml.features import temporal_match
from aml.load import load_data
from aml.roles import assign_roles


class TemporalTests(unittest.TestCase):
    def test_does_not_reuse_one_outgoing_amount(self):
        d = pd.Timestamp("2026-07-01")
        share, lag = temporal_match([(d, 100), (d, 100)], [(d + pd.Timedelta(days=1), 100)])
        self.assertEqual(share, .5)
        self.assertEqual(lag, 1)

    def test_excludes_prior_same_day_and_late_outflows(self):
        d = pd.Timestamp("2026-07-04")
        outgoing = [(d + pd.Timedelta(days=n), 100) for n in (-1, 0, 3)]
        self.assertEqual(temporal_match([(d, 100)], outgoing), (0., 0.))

    def test_ignores_later_inflow_for_earlier_outgoing(self):
        d = pd.Timestamp("2026-07-01")
        self.assertEqual(temporal_match([(d + pd.Timedelta(days=2), 100)], [(d, 100)]), (0., 0.))


class RoleTrapTests(unittest.TestCase):
    def row(self, **overrides):
        values = dict(gid=1, depth=2, is_seed=False, in_deg=1, out_deg=0, in_kzt=500000., out_kzt=0.,
                      pass_through=0., seed_reach=0, betweenness=0., fast_out_share=0.,
                      truncated_by_depth=False, unobserved_funding=False, max_payers_same_day=1, in_cycle=False)
        values.update(overrides)
        return values

    def role(self, **overrides):
        graph = nx.DiGraph(); graph.add_node(1)
        result, _ = assign_roles(pd.DataFrame([self.row(**overrides)]), graph)
        return result.iloc[0]

    def test_depth4_is_never_terminal_even_with_large_inflow(self):
        self.assertEqual(self.role(depth=4, truncated_by_depth=True).role, "peripheral")

    def test_seed_cannot_be_transit(self):
        r = self.role(is_seed=True, depth=0, out_deg=1, out_kzt=500000., pass_through=1.)
        self.assertEqual(r.role, "peripheral")

    def test_boundary_consolidator_keeps_role_with_lower_score(self):
        normal = self.role(in_deg=5)
        cutoff = self.role(in_deg=5, depth=4, truncated_by_depth=True)
        self.assertEqual(cutoff.role, "consolidator")
        self.assertLess(cutoff.role_score, normal.role_score)

    def test_seed_cannot_use_retention_ratio(self):
        self.assertEqual(self.role(is_seed=True, depth=0, out_deg=1, pass_through=.1).role, "peripheral")

    def test_zero_betweenness_never_coordinator(self):
        self.assertNotEqual(self.role(seed_reach=10, out_deg=10).role, "coordinator")


class IntegrityTests(unittest.TestCase):
    def test_corrupted_aggregate_is_rejected(self):
        data = Path(__file__).resolve().parents[1] / "data"
        with TemporaryDirectory() as name:
            folder = Path(name)
            for table in ("nodes", "edges", "transactions"):
                frame = pd.read_parquet(data / f"{table}.parquet")
                if table == "edges":
                    frame.loc[0, "sum_kzt"] += 1
                frame.to_parquet(folder / f"{table}.parquet", index=False)
            with self.assertRaisesRegex(ValueError, "amounts differ"):
                load_data(folder)

    def test_isolates_and_large_integer_ids_survive_loading(self):
        data = Path(__file__).resolve().parents[1] / "data"
        nodes, _, _, graph = load_data(data)
        self.assertEqual(set(nodes.gid), set(graph))
        self.assertTrue(any(int(g) > 2 ** 53 for g in graph))
        self.assertEqual(len(list(nx.isolates(graph))), 19)


if __name__ == "__main__":
    unittest.main()
