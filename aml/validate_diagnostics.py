"""Check saved diagnostic artifacts without recomputing graph experiments."""
from pathlib import Path

import numpy as np
import pandas as pd

from .load import require


STABILITY_NUMERIC = (
    "cluster_stability", "cluster_stability_min", "priority_baseline_rank",
    "priority_top20_frequency", "priority_min_rank", "priority_max_rank",
)
SCOPE = "cluster_stability_scope"
COMPONENTS = ("role", "reach", "volume", "pagerank", "patterns")
SEEDS = (42, 43, 44, 45, 46)


def _numbers(frame, columns):
    require(set(columns).issubset(frame.columns), f"Missing diagnostic columns: {columns}")
    try:
        values = frame[list(columns)].to_numpy(dtype=float)
    except (TypeError, ValueError):
        raise ValueError(f"Nonnumeric diagnostic fields: {columns}") from None
    require(np.isfinite(values).all(), f"Nonfinite diagnostic fields: {columns}")
    return values


def _integers(frame, columns, minimum=0, maximum=None):
    values = _numbers(frame, columns)
    require((values == np.floor(values)).all(), f"Fractional diagnostic counts: {columns}")
    require((values >= minimum).all(), f"Negative or low diagnostic counts: {columns}")
    if maximum is not None:
        require((values <= maximum).all(), f"Excessive diagnostic counts: {columns}")


def _bounds(frame, columns, low=0., high=1.):
    values = _numbers(frame, columns)
    require(((values >= low) & (values <= high)).all(), f"Diagnostic range violation: {columns}")


def _gid_index(frame):
    require("gid" in frame, "Missing diagnostic gid")
    require(not frame.gid.isna().any(), "Empty diagnostic gid")
    gids = frame.gid.map(str)
    require(gids.str.fullmatch(r"[0-9]+").all(), "Diagnostic gid must be an exact integer string")
    require(gids.is_unique, "Duplicate diagnostic gid")
    result = frame.copy()
    result.index = gids
    return result


def validate_diagnostics(nodes_df, out_dir):
    """Raise ValueError for missing, impossible, or inconsistent diagnostics.

    ``nodes_df`` is the already-loaded nodes_roles table. CSV identifiers are
    read as strings; shared numeric fields allow only 12-significant-digit CSV
    rounding. Unused cells in mixed-family sensitivity rows may remain NaN.
    """
    nodes = _gid_index(nodes_df)
    n = len(nodes)
    temporal = ("seed_reach", "temporal_seed_reach_upper", "temporal_seed_reach_strict_days")
    _integers(nodes, temporal)
    require((nodes.temporal_seed_reach_strict_days <= nodes.temporal_seed_reach_upper).all(),
            "Strict temporal reach exceeds upper bound")
    require((nodes.temporal_seed_reach_upper <= nodes.seed_reach).all(),
            "Temporal reach exceeds structural reach")
    _integers(nodes, ("in_deg", "out_deg"))
    isolated = (nodes.in_deg + nodes.out_deg) == 0
    require(SCOPE in nodes, "Missing stability scope")
    expected_scope = np.where(isolated, "isolate_not_assessed", "observed_community")
    require((nodes[SCOPE].to_numpy() == expected_scope).all(), "Stability scope disagrees with isolation")
    _bounds(nodes, ("cluster_stability", "cluster_stability_min", "priority_top20_frequency"))
    require((nodes.cluster_stability_min <= nodes.cluster_stability + 1e-12).all(),
            "Minimum community stability exceeds mean")
    require(nodes.loc[isolated, ["cluster_stability", "cluster_stability_min"]].eq(1).all().all(),
            "Isolate stability must use the unassessed sentinel 1")
    rank_columns = ("priority_min_rank", "priority_baseline_rank", "priority_max_rank")
    _integers(nodes, rank_columns, minimum=1, maximum=n)
    require(nodes.priority_baseline_rank.is_unique, "Duplicate baseline diagnostic rank")
    require(((nodes.priority_min_rank <= nodes.priority_baseline_rank)
             & (nodes.priority_baseline_rank <= nodes.priority_max_rank)).all(), "Invalid diagnostic rank interval")

    out_dir = Path(out_dir)
    for name in ("node_stability.csv", "sensitivity.csv"):
        require((out_dir / name).is_file(), f"Missing diagnostic artifact: {name}")
    stability = _gid_index(pd.read_csv(out_dir / "node_stability.csv", dtype={"gid": "string"}))
    require(set(stability.index) == set(nodes.index), "Stability gid coverage mismatch")
    stability = stability.loc[nodes.index]
    _integers(stability, rank_columns, minimum=1, maximum=n)
    _bounds(stability, ("cluster_stability", "cluster_stability_min", "priority_top20_frequency"))
    require(SCOPE in stability, "Missing exported stability scope")
    require((stability[SCOPE] == nodes[SCOPE]).all(), "Exported stability scope mismatch")
    require(stability.loc[isolated, ["cluster_stability", "cluster_stability_min"]].eq(1).all().all(),
            "Exported isolate stability must use sentinel 1")
    exported = _numbers(stability, STABILITY_NUMERIC)
    expected = _numbers(nodes, STABILITY_NUMERIC)
    require(np.allclose(exported, expected, rtol=1e-11, atol=1e-12), "Exported stability values mismatch")

    scenarios = pd.read_csv(out_dir / "sensitivity.csv")
    require({"family", "scenario"}.issubset(scenarios.columns), "Missing sensitivity identity columns")
    require(len(scenarios) == 16 and scenarios.scenario.is_unique, "Expected 16 unique sensitivity scenarios")
    require(scenarios.family.isin(["louvain", "priority"]).all(), "Unknown sensitivity family")
    louvain = scenarios.loc[scenarios.family == "louvain"].copy()
    priority = scenarios.loc[scenarios.family == "priority"].copy()
    require(set(louvain.scenario) == {f"louvain_seed_{seed}" for seed in SEEDS}, "Incorrect Louvain scenarios")
    expected_priority = {"priority_baseline"} | {
        f"priority_{component}_{direction}" for component in COMPONENTS for direction in ("minus20", "plus20")}
    require(set(priority.scenario) == expected_priority, "Incorrect priority scenarios")
    _integers(louvain, ("seed", "n_communities", "evaluated_nodes"))
    require((louvain.evaluated_nodes == int((~isolated).sum())).all(), "Incorrect Louvain evaluated node count")
    require((louvain.n_communities <= louvain.evaluated_nodes).all(), "Too many Louvain communities")
    require(((louvain.evaluated_nodes == 0) | (louvain.n_communities >= 1)).all(), "Missing observed communities")
    require(all(row.scenario == f"louvain_seed_{int(row.seed)}" for row in louvain.itertuples()),
            "Louvain seed and scenario disagree")
    _bounds(louvain, ("ari", "modularity"), low=-1.)
    _bounds(louvain, ("pair_agreement",))
    _integers(priority, ("top20_size", "top20_overlap"), maximum=min(20, n))
    require(priority.top20_size.eq(min(20, n)).all(), "Incorrect sensitivity top size")
    _bounds(priority, ("top20_overlap_fraction", "top20_jaccard"))
    weights = [f"weight_{component}" for component in COMPONENTS]
    _bounds(priority, weights)
    require(np.allclose(priority[weights].sum(axis=1), 1., rtol=0, atol=1e-10), "Sensitivity weights must sum to one")
    require("changed_component" in priority, "Missing changed priority component")
    _numbers(priority, ("multiplier",))
    for row in priority.itertuples():
        if row.scenario == "priority_baseline":
            require(row.changed_component == "none" and row.multiplier == 1., "Invalid baseline scenario metadata")
        else:
            require(row.changed_component in COMPONENTS, "Unknown changed priority component")
            direction = "minus20" if row.multiplier == .8 else "plus20" if row.multiplier == 1.2 else None
            require(direction is not None and row.scenario == f"priority_{row.changed_component}_{direction}",
                    "Priority scenario metadata mismatch")
        expected_fraction = row.top20_overlap / row.top20_size if row.top20_size else 1.
        union = 2 * row.top20_size - row.top20_overlap
        expected_jaccard = row.top20_overlap / union if union else 1.
        require(np.isclose(row.top20_overlap_fraction, expected_fraction, rtol=1e-11, atol=1e-12)
                and np.isclose(row.top20_jaccard, expected_jaccard, rtol=1e-11, atol=1e-12),
                "Sensitivity overlap metrics disagree")
