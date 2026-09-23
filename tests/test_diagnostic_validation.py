"""Reject corrupted diagnostic artifacts while preserving mixed-row CSV NaNs."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from aml.validate_diagnostics import validate_diagnostics, STABILITY_NUMERIC, SCOPE


def fixture(folder):
    nodes = pd.DataFrame({
        "gid": [100000000000000001, 100000000000000002, 100000000000000003],
        "seed_reach": [0, 1, 0], "temporal_seed_reach_upper": [0, 1, 0],
        "temporal_seed_reach_strict_days": [0, 1, 0], "in_deg": [0, 1, 0], "out_deg": [1, 0, 0],
        "cluster_stability": [.8123456789012345, .8, 1.], "cluster_stability_min": [.6, .6, 1.],
        SCOPE: ["observed_community", "observed_community", "isolate_not_assessed"],
        "priority_baseline_rank": [1, 2, 3], "priority_top20_frequency": [1., 1., 1.],
        "priority_min_rank": [1, 1, 3], "priority_max_rank": [2, 2, 3],
    })
    nodes[["gid", *STABILITY_NUMERIC, SCOPE]].iloc[::-1].to_csv(
        folder / "node_stability.csv", index=False, float_format="%.12g")
    scenarios = [{"family": "louvain", "scenario": f"louvain_seed_{seed}", "seed": seed,
                  "n_communities": 1, "evaluated_nodes": 2, "modularity": .2,
                  "ari": 1. if seed == 42 else .8, "pair_agreement": 1. if seed == 42 else .9}
                 for seed in (42, 43, 44, 45, 46)]
    weights = {"role": .3, "reach": .25, "volume": .2, "pagerank": .15, "patterns": .1}
    variants = [("priority_baseline", "none", 1., weights)]
    for component in weights:
        for factor, direction in ((.8, "minus20"), (1.2, "plus20")):
            changed = {**weights, component: weights[component] * factor}
            total = sum(changed.values())
            variants.append((f"priority_{component}_{direction}", component, factor,
                             {name: value / total for name, value in changed.items()}))
    for name, component, factor, values in variants:
        row = {"family": "priority", "scenario": name, "changed_component": component,
               "multiplier": factor, "top20_size": 3, "top20_overlap": 3,
               "top20_overlap_fraction": 1., "top20_jaccard": 1.}
        row.update({f"weight_{key}": value for key, value in values.items()})
        scenarios.append(row)
    pd.DataFrame(scenarios).to_csv(folder / "sensitivity.csv", index=False, float_format="%.12g")
    return nodes


class DiagnosticValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.nodes = fixture(self.folder)

    def test_valid_rounding_exact_large_gids_and_inapplicable_nans(self):
        scenarios = pd.read_csv(self.folder / "sensitivity.csv")
        self.assertTrue(scenarios.loc[scenarios.family == "priority", "ari"].isna().all())
        self.assertTrue(scenarios.loc[scenarios.family == "louvain", "weight_role"].isna().all())
        self.assertIsNone(validate_diagnostics(self.nodes, self.folder))

    def test_missing_fractional_nonfinite_and_impossible_temporal_counts(self):
        mutations = [self.nodes.drop(columns="temporal_seed_reach_upper")]
        for column, value in (("temporal_seed_reach_upper", -.5),
                              ("temporal_seed_reach_upper", np.inf),
                              ("temporal_seed_reach_upper", .5),
                              ("temporal_seed_reach_upper", 2),
                              ("temporal_seed_reach_strict_days", 2)):
            bad = self.nodes.copy()
            bad[column] = bad[column].astype(float)
            bad.loc[1, column] = value
            mutations.append(bad)
        for i, bad in enumerate(mutations):
            with self.subTest(mutation=i), self.assertRaises(ValueError):
                validate_diagnostics(bad, self.folder)

    def test_stability_bounds_isolate_scope_and_rank_intervals(self):
        changes = [(0, "cluster_stability", 2.), (0, "cluster_stability_min", .99),
                   (2, SCOPE, "observed_community"), (0, SCOPE, "isolate_not_assessed"),
                   (2, "cluster_stability", .9), (0, "priority_top20_frequency", -.1),
                   (0, "priority_min_rank", 2), (0, "priority_max_rank", 0),
                   (0, "priority_baseline_rank", 2)]
        for index, column, value in changes:
            with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                bad = self.nodes.copy(); bad.loc[index, column] = value
                validate_diagnostics(bad, self.folder)

    def test_missing_artifacts_and_exported_gid_or_value_mismatch(self):
        for filename in ("node_stability.csv", "sensitivity.csv"):
            with self.subTest(missing=filename):
                (self.folder / filename).unlink()
                with self.assertRaisesRegex(ValueError, "Missing diagnostic artifact"):
                    validate_diagnostics(self.nodes, self.folder)
                fixture(self.folder)
        original = pd.read_csv(self.folder / "node_stability.csv", dtype={"gid": str})
        corrupted = [original.iloc[:-1].copy(), pd.concat([original, original.iloc[:1]])]
        wrong_gid = original.copy(); wrong_gid.loc[0, "gid"] = "100000000000000099"; corrupted.append(wrong_gid)
        wrong_value = original.copy(); wrong_value.loc[0, "cluster_stability"] = .4; corrupted.append(wrong_value)
        fractional_rank = original.copy()
        fractional_rank["priority_baseline_rank"] = fractional_rank.priority_baseline_rank.astype(float)
        fractional_rank.loc[0, "priority_baseline_rank"] += 1e-10
        corrupted.append(fractional_rank)
        for i, bad in enumerate(corrupted):
            with self.subTest(corruption=i), self.assertRaises(ValueError):
                bad.to_csv(self.folder / "node_stability.csv", index=False)
                validate_diagnostics(self.nodes, self.folder)

    def test_sensitivity_missing_duplicate_or_unexpected_scenarios(self):
        original = pd.read_csv(self.folder / "sensitivity.csv")
        bad_identity = original.copy(); bad_identity.loc[0, "scenario"] = "louvain_seed_999"
        for bad in (original.iloc[:-1], pd.concat([original.iloc[:-1], original.iloc[:1]]), bad_identity):
            with self.subTest(rows=len(bad)), self.assertRaises(ValueError):
                bad.to_csv(self.folder / "sensitivity.csv", index=False)
                validate_diagnostics(self.nodes, self.folder)

    def test_sensitivity_applicable_values_and_metadata(self):
        original = pd.read_csv(self.folder / "sensitivity.csv")
        changes = [(0, "ari", np.nan), (0, "pair_agreement", 1.1), (0, "seed", 43),
                   (0, "evaluated_nodes", 3), (0, "n_communities", 3),
                   (5, "weight_role", .9), (5, "top20_overlap", 4),
                   (5, "top20_jaccard", .4), (6, "multiplier", .9)]
        for index, column, value in changes:
            with self.subTest(column=column), self.assertRaises(ValueError):
                bad = original.copy(); bad.loc[index, column] = value
                bad.to_csv(self.folder / "sensitivity.csv", index=False)
                validate_diagnostics(self.nodes, self.folder)


if __name__ == "__main__":
    unittest.main()
