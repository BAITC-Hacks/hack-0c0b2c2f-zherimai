"""Behavioral checks for partition comparison, ranking and analysis boundaries."""
import copy
import json
import unittest

import networkx as nx
import numpy as np
import pandas as pd

from aml.clusters import assign_clusters
from aml.sensitivity import (PRIORITY_WEIGHTS, _membership_jaccard,
                             adjusted_rand_index, analyze_sensitivity,
                             partition_agreement)


def fixture(count=27, connected=26):
    # Large generated IDs catch accidental float conversion without dataset IDs.
    gids = [10 ** 17 + i for i in range(count)]
    graph = nx.DiGraph()
    graph.add_nodes_from(gids)
    if connected > 1:
        for i in range(connected):
            graph.add_edge(gids[i], gids[(i + 1) % connected], sum_kzt=5000. + i)
    frame = pd.DataFrame({"gid": gids, "is_seed": [i % 7 == 0 for i in range(count)]})
    frame, projection = assign_clusters(frame, graph)
    random = np.random.default_rng(731)
    for component, weight in PRIORITY_WEIGHTS.items():
        frame[f"priority_{component}"] = random.uniform(.05, 1., size=count) * weight
    frame["seed_discount"] = np.where(frame.is_seed, .6, 1.)
    frame["priority_score"] = (frame[[f"priority_{c}" for c in PRIORITY_WEIGHTS]].sum(axis=1)
                               * frame.seed_discount).round(8)
    return frame, graph, projection


class PartitionTests(unittest.TestCase):
    def test_identical_partition_ignores_label_names(self):
        self.assertEqual(partition_agreement([2, 2, 9, 9, 5], ["a", "a", "z", "z", "b"]), (1., 1.))

    def test_moved_node_changes_ari_and_its_membership(self):
        self.assertAlmostEqual(adjusted_rand_index([0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 1, 1]), 12 / 37)
        baseline = dict(zip("abcdef", [0, 0, 0, 1, 1, 1]))
        alternative = dict(zip("abcdef", [0, 0, 1, 1, 1, 1]))
        np.testing.assert_allclose(_membership_jaccard(list("abcdef"), baseline, alternative),
                                   [2 / 3, 2 / 3, 1 / 6, 3 / 4, 3 / 4, 3 / 4])

    def test_below_chance_and_degenerate_partitions(self):
        self.assertAlmostEqual(adjusted_rand_index([0, 0, 1, 1], [0, 1, 0, 1]), -.5)
        for left, right in (([], []), ([0], [9]), ([0, 0], [9, 9]), ([0, 1], [9, 8])):
            self.assertEqual(adjusted_rand_index(left, right), 1.)
        with self.assertRaises(ValueError):
            adjusted_rand_index([0], [])


class SensitivityTests(unittest.TestCase):
    def test_shuffle_does_not_change_gid_aligned_results_or_inputs(self):
        frame, graph, projection = fixture()
        original = frame.copy(deep=True)
        graph_edges = copy.deepcopy(list(graph.edges(data=True)))
        projection_edges = copy.deepcopy(list(projection.edges(data=True)))
        summary, nodes, scenarios = analyze_sensitivity(frame, graph, projection)
        shuffled_summary, shuffled_nodes, shuffled_scenarios = analyze_sensitivity(
            frame.sample(frac=1., random_state=9), graph, projection)
        self.assertEqual(summary, shuffled_summary)
        pd.testing.assert_frame_equal(nodes, shuffled_nodes)
        pd.testing.assert_frame_equal(scenarios, shuffled_scenarios)
        pd.testing.assert_frame_equal(frame, original)
        self.assertEqual(list(graph.edges(data=True)), graph_edges)
        self.assertEqual(list(projection.edges(data=True)), projection_edges)
        self.assertEqual(set(nodes.gid), set(frame.gid))
        self.assertTrue(pd.api.types.is_integer_dtype(nodes.gid))
        json.dumps(summary, allow_nan=False)

    def test_excludes_isolates_and_keeps_scope_explicit(self):
        frame, graph, projection = fixture()
        summary, nodes, scenarios = analyze_sensitivity(frame, graph, projection)
        isolated = nodes[nodes.gid.isin(nx.isolates(graph))]
        self.assertEqual(len(isolated), 1)
        self.assertTrue(isolated.cluster_stability_scope.eq("isolate_not_assessed").all())
        self.assertTrue(isolated.cluster_stability.eq(1).all())
        self.assertEqual(summary["louvain"]["evaluated_nodes"], 26)
        self.assertEqual(summary["louvain"]["excluded_isolates"], 1)
        self.assertTrue(summary["louvain"]["seed42_matches_baseline"])
        community_rows = scenarios[scenarios.family == "louvain"]
        self.assertEqual(list(community_rows.seed.astype(int)), [42, 43, 44, 45, 46])
        self.assertTrue(community_rows.evaluated_nodes.eq(26).all())

    def test_weights_normalized_and_rank_frequency_counts_all_scenarios(self):
        frame, graph, projection = fixture()
        summary, nodes, scenarios = analyze_sensitivity(frame, graph, projection)
        priority = scenarios[scenarios.family == "priority"]
        self.assertEqual(len(priority), 11)
        weight_columns = [f"weight_{c}" for c in PRIORITY_WEIGHTS]
        np.testing.assert_allclose(priority[weight_columns].sum(axis=1), 1.)
        self.assertEqual(summary["priority"]["scenario_count_including_baseline"], 11)
        self.assertAlmostEqual(nodes.priority_top20_frequency.sum(), 20.)
        self.assertTrue(nodes.priority_min_rank.le(nodes.priority_baseline_rank).all())
        self.assertTrue(nodes.priority_max_rank.ge(nodes.priority_baseline_rank).all())
        self.assertTrue(nodes.priority_top20_frequency.between(0, 1).all())
        self.assertTrue(nodes.cluster_stability.between(0, 1).all())
        baseline = priority[priority.scenario == "priority_baseline"].iloc[0]
        self.assertEqual(baseline.top20_jaccard, 1.)
        self.assertEqual(baseline.top20_overlap, 20)

    def test_ties_use_exact_large_gid_and_seed_discount_is_preserved(self):
        frame, graph, projection = fixture()
        frame["is_seed"] = False
        frame.loc[0, "is_seed"] = True
        frame["seed_discount"] = np.where(frame.is_seed, .6, 1.)
        for component, weight in PRIORITY_WEIGHTS.items():
            frame[f"priority_{component}"] = weight
        frame["priority_score"] = frame.seed_discount
        _, nodes, _ = analyze_sensitivity(frame.sample(frac=1., random_state=19), graph, projection)
        expected = frame.gid.iloc[1:].tolist() + [frame.gid.iloc[0]]
        actual = nodes.sort_values("priority_baseline_rank").gid.tolist()
        self.assertEqual(actual, expected)
        self.assertTrue(nodes.priority_min_rank.eq(nodes.priority_max_rank).all())
        self.assertTrue(nodes.loc[nodes.gid == frame.gid.iloc[0], "priority_top20_frequency"].eq(0).all())

    def test_only_isolates_and_small_top_have_defined_output(self):
        frame, graph, projection = fixture(count=3, connected=0)
        summary, nodes, _ = analyze_sensitivity(frame, graph, projection)
        self.assertEqual(summary["louvain"]["evaluated_nodes"], 0)
        self.assertIsNone(summary["louvain"]["membership_jaccard_mean"])
        self.assertEqual(summary["priority"]["top_k"], 3)
        self.assertTrue(nodes.priority_top20_frequency.eq(1).all())
        json.dumps(summary, allow_nan=False)
        empty_frame, empty_graph, empty_projection = fixture(count=0, connected=0)
        empty_summary, empty_nodes, _ = analyze_sensitivity(empty_frame, empty_graph, empty_projection)
        self.assertEqual(empty_summary["priority"]["top_k"], 0)
        self.assertEqual(len(empty_nodes), 0)
        json.dumps(empty_summary, allow_nan=False)

    def test_rejects_invalid_component_and_zero_projection_weight(self):
        frame, graph, projection = fixture()
        invalid = frame.copy()
        invalid.loc[0, "priority_role"] = .31
        with self.assertRaisesRegex(ValueError, "documented baseline weight"):
            analyze_sensitivity(invalid, graph, projection)
        invalid = frame.copy()
        invalid.loc[0, "priority_volume"] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            analyze_sensitivity(invalid, graph, projection)
        broken_projection = projection.copy()
        src, dst = next(iter(broken_projection.edges))
        broken_projection[src][dst]["weight"] = 0.
        with self.assertRaisesRegex(ValueError, "positive projection weights"):
            analyze_sensitivity(frame, graph, broken_projection)

    def test_rejects_stale_scores_discount_and_incomplete_projection(self):
        frame, graph, projection = fixture()
        bad = frame.copy()
        bad.loc[0, "priority_score"] += .02
        with self.assertRaisesRegex(ValueError, "baseline scores"):
            analyze_sensitivity(bad, graph, projection)
        bad = frame.copy()
        bad.loc[0, "seed_discount"] = 1.
        with self.assertRaisesRegex(ValueError, "seed discount"):
            analyze_sensitivity(bad, graph, projection)
        smaller = projection.copy()
        smaller.remove_node(next(iter(smaller)))
        with self.assertRaisesRegex(ValueError, "non-isolated"):
            analyze_sensitivity(frame, graph, smaller)


if __name__ == "__main__":
    unittest.main()
