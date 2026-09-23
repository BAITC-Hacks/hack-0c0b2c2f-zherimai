"""Bounded sensitivity of observed communities and review priority, not accuracy.

The existing roles and component values stay fixed. Louvain runs use exactly the
supplied undirected projection; only the random seed changes. Priority scenarios
change one weight at a time by +/-20%, then normalize all five weights to sum to
one. These checks do not measure sensitivity to role thresholds or missing data.
"""
from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd


LOUVAIN_SEEDS = (42, 43, 44, 45, 46)
PRIORITY_WEIGHTS = {
    "role": .30, "reach": .25, "volume": .20, "pagerank": .15, "patterns": .10,
}


def _choose_two(n):
    return n * (n - 1) // 2


def partition_agreement(labels_a, labels_b):
    """Return adjusted Rand and pair-agreement indices, independent of labels.

    Counts are exact integers until the final division. Degenerate identical
    partitions (zero/one item, all singleton groups, or one group) return 1.
    """
    labels_a, labels_b = list(labels_a), list(labels_b)
    if len(labels_a) != len(labels_b):
        raise ValueError("Partitions must contain the same number of nodes")
    pairs = _choose_two(len(labels_a))
    if not pairs:
        return 1., 1.
    together_a = sum(_choose_two(n) for n in Counter(labels_a).values())
    together_b = sum(_choose_two(n) for n in Counter(labels_b).values())
    together_both = sum(_choose_two(n) for n in Counter(zip(labels_a, labels_b)).values())
    denominator = pairs * (together_a + together_b) - 2 * together_a * together_b
    ari = (2 * (together_both * pairs - together_a * together_b) / denominator
           if denominator else 1.)
    separate_both = pairs - together_a - together_b + together_both
    return float(ari), float((together_both + separate_both) / pairs)


def adjusted_rand_index(labels_a, labels_b):
    """Adjusted Rand index; 1 is identical, 0 is chance expectation, negative is possible."""
    return partition_agreement(labels_a, labels_b)[0]


def _membership_jaccard(gids, baseline, alternative):
    """Compare the two complete communities containing each node, not their IDs."""
    baseline_sizes = Counter(baseline[g] for g in gids)
    alternative_sizes = Counter(alternative[g] for g in gids)
    intersections = Counter((baseline[g], alternative[g]) for g in gids)
    return np.asarray([
        intersections[baseline[g], alternative[g]]
        / (baseline_sizes[baseline[g]] + alternative_sizes[alternative[g]]
           - intersections[baseline[g], alternative[g]])
        for g in gids
    ], dtype=float)


def _rank(gids, scores):
    # The same tie break as aml.explain.top_nodes; gids never pass through floats.
    ranking = pd.DataFrame({"gid": gids, "score": scores}).sort_values(
        ["score", "gid"], ascending=[False, True], kind="stable")
    return dict(zip(ranking.gid, range(1, len(ranking) + 1)))


def _validate_inputs(df, graph, projection):
    required = {"gid", "cluster_id", "priority_score", "seed_discount", "is_seed"}
    required.update(f"priority_{name}" for name in PRIORITY_WEIGHTS)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Sensitivity input lacks columns: {sorted(missing)}")
    if df.gid.isna().any() or not df.gid.is_unique:
        raise ValueError("Sensitivity requires unique, nonempty gids")
    if set(df.gid) != set(graph):
        raise ValueError("Sensitivity features and graph must contain the same gids")
    if df.cluster_id.isna().any():
        raise ValueError("Sensitivity requires a baseline cluster for every gid")
    if projection.is_directed() or projection.is_multigraph():
        raise ValueError("Sensitivity needs the original undirected simple projection")
    observed = {g for g in graph if graph.degree(g) > 0}
    if set(projection) != observed:
        raise ValueError("Sensitivity projection must contain all and only non-isolated nodes")
    if {frozenset((u, v)) for u, v in graph.edges} != {
        frozenset((u, v)) for u, v in projection.edges
    }:
        raise ValueError("Sensitivity projection must preserve the observed graph edges")
    if any(not np.isfinite(data.get("weight", np.nan)) or data["weight"] <= 0
           for _, _, data in projection.edges(data=True)):
        raise ValueError("Sensitivity requires finite, positive projection weights")
    numeric = ["priority_score", "seed_discount"] + [f"priority_{n}" for n in PRIORITY_WEIGHTS]
    if not np.isfinite(df[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Sensitivity priority inputs must be finite")
    if not df.is_seed.isin([True, False]).all():
        raise ValueError("Sensitivity is_seed values must be boolean")
    expected_discount = np.where(df.is_seed, .6, 1.)
    if not np.allclose(df.seed_discount, expected_discount, rtol=0, atol=1e-12):
        raise ValueError("Sensitivity must retain the existing 0.6 seed discount")
    for name, weight in PRIORITY_WEIGHTS.items():
        values = df[f"priority_{name}"]
        if ((values < -1e-10) | (values > weight + 1e-10)).any():
            raise ValueError(f"priority_{name} is incompatible with the documented baseline weight")
    recomputed = (df[[f"priority_{n}" for n in PRIORITY_WEIGHTS]].sum(axis=1)
                  * df.seed_discount).clip(0, 1).round(8)
    if not np.allclose(df.priority_score, recomputed, rtol=0, atol=1e-8):
        raise ValueError("Sensitivity baseline scores disagree with their disclosed components")


def analyze_sensitivity(features_df, graph, projection):
    """Return ``(JSON-safe summary, per-node DataFrame, scenario DataFrame)``.

    Inputs are not mutated. ``cluster_stability`` is mean membership Jaccard over
    seeds 43--46 against input ``cluster_id``. Isolates carry the technical value
    1 with ``cluster_stability_scope=isolate_not_assessed`` and are excluded from
    community summaries. ``priority_top20_frequency`` includes baseline plus ten
    weight variants (11 scenarios), with k=min(20, number of nodes).

    ``scenarios.family`` distinguishes Louvain rows from priority rows; columns
    inapplicable to a row's family are deliberately empty, not zero measurements.
    """
    _validate_inputs(features_df, graph, projection)
    df = features_df.sort_values("gid", kind="stable").reset_index(drop=True)
    gids = list(df.gid)
    observed = [g for g in gids if g in projection]
    baseline = dict(zip(df.gid, df.cluster_id))
    base_labels = [baseline[g] for g in observed]
    baseline_groups = {}
    for gid in observed:
        baseline_groups.setdefault(baseline[gid], set()).add(gid)
    baseline_modularity = (nx.community.modularity(
        projection, list(baseline_groups.values()), weight="weight", resolution=1)
        if projection.number_of_edges() else 0.)
    scenarios, community_rows, memberships = [], [], []
    for seed in LOUVAIN_SEEDS:
        communities = (nx.community.louvain_communities(
            projection, weight="weight", resolution=1, seed=seed)
            if projection.number_of_edges() else [])
        membership = {g: i for i, community in enumerate(communities) for g in community}
        ari, pairs = partition_agreement(base_labels, [membership[g] for g in observed])
        modularity = (nx.community.modularity(projection, communities, weight="weight", resolution=1)
                      if projection.number_of_edges() else 0.)
        row = {"family": "louvain", "scenario": f"louvain_seed_{seed}", "seed": seed,
               "n_communities": len(communities), "modularity": float(modularity),
               "ari": ari, "pair_agreement": pairs, "evaluated_nodes": len(observed)}
        scenarios.append(row)
        community_rows.append(row)
        if seed != LOUVAIN_SEEDS[0]:
            memberships.append(_membership_jaccard(observed, baseline, membership))
    membership_mean = np.mean(memberships, axis=0) if observed else np.array([])
    membership_min = np.min(memberships, axis=0) if observed else np.array([])
    stability = dict(zip(observed, membership_mean))
    minimum_stability = dict(zip(observed, membership_min))

    baseline_ranks = _rank(gids, df.priority_score)
    k = min(20, len(df))
    baseline_top = {g for g, rank in baseline_ranks.items() if rank <= k}
    components = np.column_stack([
        df[f"priority_{name}"].to_numpy() / weight
        for name, weight in PRIORITY_WEIGHTS.items()
    ])
    discounts = df.seed_discount.to_numpy()
    weight_sets = [("priority_baseline", "none", 1., dict(PRIORITY_WEIGHTS))]
    for name in PRIORITY_WEIGHTS:
        for factor in (.8, 1.2):
            changed = dict(PRIORITY_WEIGHTS)
            changed[name] *= factor
            total = sum(changed.values())
            normalized = {key: weight / total for key, weight in changed.items()}
            weight_sets.append((f"priority_{name}_{'minus20' if factor < 1 else 'plus20'}",
                                name, factor, normalized))
    rank_columns, priority_rows = [], []
    for scenario, changed_component, factor, weights in weight_sets:
        if scenario == "priority_baseline":
            ranks = baseline_ranks
        else:
            scores = np.round(np.clip(
                components @ np.asarray(list(weights.values())) * discounts, 0., 1.), 8)
            ranks = _rank(gids, scores)
        rank_columns.append([ranks[g] for g in gids])
        top = {g for g, rank in ranks.items() if rank <= k}
        intersection, union = len(top & baseline_top), len(top | baseline_top)
        row = {"family": "priority", "scenario": scenario,
               "changed_component": changed_component, "multiplier": factor,
               "top20_size": k, "top20_overlap": intersection,
               "top20_overlap_fraction": intersection / k if k else 1.,
               "top20_jaccard": intersection / union if union else 1.}
        row.update({f"weight_{name}": weight for name, weight in weights.items()})
        scenarios.append(row)
        priority_rows.append(row)
    ranks_array = np.asarray(rank_columns, dtype=int)
    per_node = pd.DataFrame({
        "gid": gids,
        "cluster_stability": [float(stability.get(g, 1.)) for g in gids],
        "cluster_stability_min": [float(minimum_stability.get(g, 1.)) for g in gids],
        "cluster_stability_scope": ["observed_community" if g in projection
                                    else "isolate_not_assessed" for g in gids],
        "priority_baseline_rank": [baseline_ranks[g] for g in gids],
        "priority_top20_frequency": np.mean(ranks_array <= k, axis=0),
        "priority_min_rank": np.min(ranks_array, axis=0),
        "priority_max_rank": np.max(ranks_array, axis=0),
    })
    alternative_communities = community_rows[1:]
    alternative_priorities = priority_rows[1:]
    summary = {
        "louvain": {
            "seeds": list(LOUVAIN_SEEDS), "resolution": 1., "weight": "weight",
            "evaluated_nodes": len(observed), "excluded_isolates": len(df) - len(observed),
            "baseline_communities": len(baseline_groups),
            "baseline_modularity": float(baseline_modularity),
            "seed42_matches_baseline": community_rows[0]["ari"] == 1.,
            "communities_min": min(r["n_communities"] for r in community_rows),
            "communities_max": max(r["n_communities"] for r in community_rows),
            "alternative_ari_min": min(r["ari"] for r in alternative_communities),
            "alternative_ari_mean": float(np.mean([r["ari"] for r in alternative_communities])),
            "alternative_pair_agreement_min": min(r["pair_agreement"] for r in alternative_communities),
            "membership_jaccard_mean": float(membership_mean.mean()) if observed else None,
            "membership_jaccard_min": float(membership_mean.min()) if observed else None,
        },
        "priority": {
            "baseline_weights": dict(PRIORITY_WEIGHTS), "multipliers": [.8, 1.2],
            "scenario_count_including_baseline": len(weight_sets), "top_k": k,
            "alternative_top20_overlap_min": min(r["top20_overlap"] for r in alternative_priorities),
            "alternative_top20_overlap_mean": float(np.mean([r["top20_overlap"] for r in alternative_priorities])),
            "alternative_top20_jaccard_min": min(r["top20_jaccard"] for r in alternative_priorities),
            "always_top20_nodes": int(per_node.priority_top20_frequency.eq(1).sum()),
            "ever_top20_nodes": int(per_node.priority_top20_frequency.gt(0).sum()),
            "seed_discount": .6,
        },
        "limits": [
            "Only Louvain random seeds and individual priority weights are varied.",
            "Roles, role thresholds, data, feature normalization and cluster projection stay fixed.",
            "Stability is not accuracy, calibrated confidence or evidence of wrongdoing.",
            "Isolates are excluded from community analysis; their per-node stability=1 is a technical sentinel.",
            "Node membership stability excludes seed 42; priority frequency includes baseline and 10 variants.",
            "No sensitivity to missing flows, projection choices, resolution or simultaneous weight changes is measured.",
        ],
    }
    return summary, per_node, pd.DataFrame(scenarios)
