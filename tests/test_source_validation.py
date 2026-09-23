"""Source-derived guards reject QA mutations even when CSV metadata masks them."""
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from aml.validate import validate_outputs


class SourceValidationTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.data, self.out = self.folder / "data", self.folder / "out"
        shutil.copytree(root / "data", self.data)
        shutil.copytree(root / "out", self.out)
        self.nodes = pd.read_csv(self.out / "nodes_roles.csv", dtype={"gid": "int64"})
        source = pd.read_parquet(self.data / "nodes.parquet").set_index("gid")
        edges = pd.read_parquet(self.data / "edges.parquet")
        top = set(pd.read_csv(self.out / "top_nodes.csv").gid)
        source_depth = self.nodes.gid.map(source.depth)
        source_seed = self.nodes.gid.map(source.is_seed)
        incident = set(edges.src) | set(edges.dst)
        self.depth4 = self.nodes.index[(source_depth == 4) & ~self.nodes.gid.isin(top)][0]
        self.isolate = self.nodes.index[~self.nodes.gid.isin(incident)][0]
        self.seed = self.nonseed = None
        for _, group in self.nodes.loc[~self.nodes.gid.isin(top)].groupby("cluster_id", sort=True):
            seed_indices = group.index[source_seed.loc[group.index]]
            nonseed_indices = group.index[~source_seed.loc[group.index]]
            if len(seed_indices) and len(nonseed_indices):
                self.seed, self.nonseed = seed_indices[0], nonseed_indices[0]
                break
        self.assertIsNotNone(self.seed, "Need a seed and non-seed in the same observed cluster")

    def check(self, nodes=None):
        if nodes is not None:
            nodes.to_csv(self.out / "nodes_roles.csv", index=False, float_format="%.12g")
        return validate_outputs(self.data, self.out)

    def swap_seeds(self, nodes):
        nodes.loc[self.seed, "is_seed"] = False
        nodes.loc[self.nonseed, "is_seed"] = True
        return nodes

    def test_baseline_and_reordered_rows_pass(self):
        self.assertEqual(self.check()["status"], "PASS")
        self.assertEqual(self.check(self.nodes.iloc[::-1])["status"], "PASS")

    def test_depth_only_is_rejected(self):
        bad = self.nodes.copy(); bad.loc[self.depth4, "depth"] = 3
        with self.assertRaisesRegex(ValueError, "Source depth mismatch"):
            self.check(bad)

    def test_depth4_terminal_control_is_rejected(self):
        bad = self.nodes.copy(); bad.loc[self.depth4, "role"] = "terminal"
        with self.assertRaisesRegex(ValueError, "Depth-4 false terminal"):
            self.check(bad)

    def test_depth4_terminal_with_masked_depth_is_rejected(self):
        bad = self.nodes.copy(); bad.loc[self.depth4, ["depth", "role"]] = [3, "terminal"]
        with self.assertRaisesRegex(ValueError, "Source depth mismatch"):
            self.check(bad)

    def test_isolate_degree_only_is_rejected(self):
        bad = self.nodes.copy(); bad.loc[self.isolate, "in_deg"] = 1
        with self.assertRaisesRegex(ValueError, "Source in_deg mismatch"):
            self.check(bad)

    def test_isolate_role_control_is_rejected(self):
        bad = self.nodes.copy(); bad.loc[self.isolate, "role"] = "consolidator"
        with self.assertRaisesRegex(ValueError, "Isolated node role"):
            self.check(bad)

    def test_isolate_role_with_masked_degree_is_rejected(self):
        bad = self.nodes.copy(); bad.loc[self.isolate, ["in_deg", "role"]] = [1, "consolidator"]
        with self.assertRaisesRegex(ValueError, "Source in_deg mismatch"):
            self.check(bad)

    def test_seed_swap_preserving_cluster_count_is_rejected(self):
        bad = self.swap_seeds(self.nodes.copy())
        self.assertEqual(bad.is_seed.sum(), self.nodes.is_seed.sum())
        self.assertEqual(bad.loc[self.seed, "cluster_id"], bad.loc[self.nonseed, "cluster_id"])
        with self.assertRaisesRegex(ValueError, "Source is_seed mismatch"):
            self.check(bad)

    def test_seed_transit_control_is_rejected(self):
        bad = self.nodes.copy(); bad.loc[self.seed, "role"] = "transit"
        with self.assertRaisesRegex(ValueError, "Seed balance used for transit"):
            self.check(bad)

    def test_seed_transit_with_masked_seed_is_rejected(self):
        bad = self.swap_seeds(self.nodes.copy()); bad.loc[self.seed, "role"] = "transit"
        with self.assertRaisesRegex(ValueError, "Source is_seed mismatch"):
            self.check(bad)

    def test_edge_derived_sums_counts_and_out_degree_are_checked(self):
        for column in ("in_kzt", "out_kzt", "in_tx", "out_tx", "out_deg"):
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, f"Source {column} mismatch"):
                bad = self.nodes.copy(); bad.loc[0, column] += 1
                self.check(bad)

    def test_source_derived_cutoff_and_isolate_flags_are_checked(self):
        bad = self.nodes.copy(); bad.loc[self.depth4, "truncated_by_depth"] = False
        with self.assertRaisesRegex(ValueError, "Source depth-cutoff flag mismatch"):
            self.check(bad)
        for index, flag in ((self.depth4, "truncated_depth4"), (self.isolate, "no_observed_edges")):
            with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, f"Source {flag} flag mismatch"):
                bad = self.nodes.copy()
                bad.loc[index, "flags"] = ";".join(part for part in str(bad.loc[index, "flags"]).split(";") if part != flag)
                self.check(bad)


if __name__ == "__main__":
    unittest.main()
