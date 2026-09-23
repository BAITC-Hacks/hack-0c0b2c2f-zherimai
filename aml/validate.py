"""Mechanical output contract plus high-risk domain invariants."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .load import require
from .roles import ROLES
from .validate_diagnostics import validate_diagnostics


def validate_outputs(data_dir: Path, out_dir: Path, max_seconds=300):
    source = pd.read_parquet(data_dir / "nodes.parquet")
    edges = pd.read_parquet(data_dir / "edges.parquet")
    nodes = pd.read_csv(out_dir / "nodes_roles.csv", dtype={"gid": "int64"})
    clusters = pd.read_csv(out_dir / "clusters.csv")
    top = pd.read_csv(out_dir / "top_nodes.csv", dtype={"gid": "int64"})
    required = [
        (nodes, ["gid", "role", "role_score", "cluster_id", "priority_score", "evidence",
                 "depth", "is_seed", "in_deg", "out_deg", "in_kzt", "out_kzt",
                 "in_tx", "out_tx", "truncated_by_depth"]),
        (clusters, ["cluster_id", "n_nodes", "n_seed", "sum_kzt_internal", "top_gids", "hypothesis"]),
        (top, ["rank", "gid", "role", "priority_score", "why"]),
    ]
    for table, columns in required:
        require(set(columns).issubset(table.columns), f"Missing output columns: {columns}")
        require(not table[columns].isna().any().any(), f"Empty mandatory cells: {columns}")
    require(len(nodes) == len(source), "Output node count differs from input")
    require(nodes.gid.is_unique and set(nodes.gid) == set(source.gid), "Output gid coverage mismatch")
    # Independently reconstruct trusted attributes in output row order. Domain
    # guards below must not trust depth/seed/degree copied into the output CSV.
    truth = source.set_index("gid").loc[nodes.gid].reset_index()
    require(np.array_equal(nodes.depth, truth.depth), "Source depth mismatch")
    require(pd.api.types.is_bool_dtype(nodes.is_seed)
            and np.array_equal(nodes.is_seed, truth.is_seed), "Source is_seed mismatch")
    for prefix, endpoint, counterpart in (("in", "dst", "src"), ("out", "src", "dst")):
        grouped = edges.groupby(endpoint)
        aggregates = {f"{prefix}_deg": grouped[counterpart].nunique(),
                      f"{prefix}_kzt": grouped.sum_kzt.sum(),
                      f"{prefix}_tx": grouped.n_tx.sum()}
        for column, aggregate in aggregates.items():
            truth[column] = truth.gid.map(aggregate).fillna(0)
            observed = pd.to_numeric(nodes[column], errors="coerce").to_numpy(dtype=float)
            require(np.isfinite(observed).all(), f"Invalid source metric: {column}")
            expected_values = truth[column].to_numpy()
            # Amount tolerance matches the input aggregate check (0.01 KZT).
            equal = (np.allclose(observed, expected_values, rtol=0, atol=.01)
                     if column.endswith("_kzt") else np.array_equal(observed, expected_values))
            require(equal, f"Source {column} mismatch")
    source_isolated = (truth.in_deg + truth.out_deg) == 0
    source_truncated = (truth.depth == 4) & (truth.out_deg == 0)
    require(pd.api.types.is_bool_dtype(nodes.truncated_by_depth)
            and np.array_equal(nodes.truncated_by_depth, source_truncated), "Source depth-cutoff flag mismatch")
    if "flags" in nodes:
        flag_sets = nodes["flags"].fillna("").astype(str).map(lambda value: set(value.split(";")))
        for flag, expected_flags in (("truncated_depth4", source_truncated),
                                     ("no_observed_edges", source_isolated), ("seed", truth.is_seed)):
            require(np.array_equal(flag_sets.map(lambda values: flag in values), expected_flags),
                    f"Source {flag} flag mismatch")
    require(nodes.role.isin(ROLES).all(), "Unknown role")
    for col in ("role_score", "priority_score"):
        require(np.isfinite(nodes[col]).all() and nodes[col].between(0, 1).all(), f"Invalid {col}")
    require(nodes.evidence.str.len().between(1, 200).all(), "Evidence outside 1..200 chars")
    require(nodes.evidence.str.contains(r"\d", regex=True).all(), "Evidence without numbers")
    require(clusters.cluster_id.is_unique and set(nodes.cluster_id) == set(clusters.cluster_id), "Cluster id mismatch")
    require((nodes.cluster_id % 1 == 0).all(), "Fractional cluster id")
    require(len(top) >= min(20, len(nodes)), "Less than 20 top nodes")
    require(top.gid.is_unique and set(top.gid).issubset(set(nodes.gid)), "Invalid top gids")
    expected = nodes.sort_values(["priority_score", "gid"], ascending=[False, True]).head(len(top))
    require(list(top.gid) == list(expected.gid), "Top order or membership mismatch")
    require(list(top['rank']) == list(range(1, len(top) + 1)), "Top ranks must be consecutive")
    require(np.allclose(top.priority_score, expected.priority_score), "Top priorities differ")
    require(list(top.role) == list(expected.role), "Top roles differ")
    require(top.why.astype(str).str.strip().str.len().gt(0).all(), "Empty top reason")
    require(not ((truth.depth == 4) & (nodes.role == "terminal")).any(), "Depth-4 false terminal")
    require(not (truth.is_seed & (nodes.role == "transit")).any(), "Seed balance used for transit")
    require((nodes.loc[source_isolated, "role"] == "peripheral").all(), "Isolated node role")
    by_gid = nodes.set_index("gid")
    truth_by_gid = truth.set_index("gid")
    internal = edges.assign(sc=edges.src.map(by_gid.cluster_id), dc=edges.dst.map(by_gid.cluster_id))
    internal = internal.loc[internal.sc == internal.dc].groupby("sc").sum_kzt.sum()
    for c in clusters.itertuples(index=False):
        members = nodes.loc[nodes.cluster_id == c.cluster_id]
        require(c.n_nodes == len(members), "Incorrect cluster size")
        require(c.n_seed == int(truth_by_gid.loc[members.gid, "is_seed"].sum()), "Incorrect cluster seed count")
        require(abs(c.sum_kzt_internal - internal.get(c.cluster_id, 0.)) < .02, "Incorrect internal turnover")
        require(all(int(g) in set(members.gid) for g in str(c.top_gids).split(";")), "Invalid cluster top gids")
    require((out_dir / "network.html").is_file(), "Missing offline network viewer")
    validate_diagnostics(nodes, out_dir)
    report = json.loads((out_dir / "report.json").read_text())
    require(0 <= report["runtime_seconds"] < max_seconds, "Pipeline exceeds time limit")
    require(report["n_nodes"] == len(source), "Report count mismatch")
    return {"status": "PASS", "n_nodes": len(nodes), "n_clusters": len(clusters), "n_top": len(top), "runtime_seconds": report["runtime_seconds"]}
