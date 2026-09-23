#!/usr/bin/env python3
"""One reproducible offline run from raw parquet to validated analyst artifacts."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time
PROCESS_STARTED = time.perf_counter()
import networkx as nx
from aml.load import load_data
from aml.features import compute_features
from aml.clusters import assign_clusters, summarize_clusters
from aml.roles import assign_roles
from aml.priority import rank_nodes
from aml.explain import add_explanations, top_nodes, data_requests
from aml.resilience import analyze_resilience
from aml.sensitivity import analyze_sensitivity
from aml.viewer import build_viewer
from aml.validate import validate_outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("out"))
    args = parser.parse_args()
    started = PROCESS_STARTED
    args.out.mkdir(parents=True, exist_ok=True)
    nodes, edges, tx, graph = load_data(args.data)
    print(f"Input validated: {len(nodes)} nodes / {len(edges)} edges / {len(tx)} transactions", flush=True)
    features = compute_features(nodes, edges, tx, graph)
    features, projection = assign_clusters(features, graph)
    features, thresholds = assign_roles(features, graph)
    features = add_explanations(rank_nodes(features)).sort_values("gid").reset_index(drop=True)
    sensitivity, stability, scenarios = analyze_sensitivity(features, graph, projection)
    features = features.merge(stability, on="gid", how="left", validate="one_to_one")
    clusters = summarize_clusters(features, edges)
    mandatory = ["gid", "role", "role_score", "cluster_id", "priority_score", "evidence"]
    features = features[mandatory + [c for c in features.columns if c not in mandatory]]
    outputs = {"nodes_roles": features, "clusters": clusters, "top_nodes": top_nodes(features),
               "data_requests": data_requests(features), "resilience": analyze_resilience(features, graph),
               "node_stability": stability, "sensitivity": scenarios}
    for name, frame in outputs.items():
        frame.to_csv(args.out / f"{name}.csv", index=False, float_format="%.12g")
    build_viewer(features, edges, clusters, args.out / "network.html")
    report = {"n_nodes": len(nodes), "n_edges": len(edges), "n_transactions": len(tx),
              "sum_kzt": float(edges.sum_kzt.sum()), "n_seed": int(nodes.is_seed.sum()),
              "weak_components_including_isolates": nx.number_weakly_connected_components(graph),
              "isolated_nodes": len(list(nx.isolates(graph))),
              "truncated_depth4": int(features.truncated_by_depth.sum()),
              "n_clusters_including_isolate_group": len(clusters),
              "communities_with_multiple_seeds": int((clusters.loc[clusters.cluster_id != 0, "n_seed"] > 1).sum()),
              "role_counts": {k: int(v) for k, v in features.role.value_counts().items()},
              "temporal_reach": {
                  "nodes_with_upper_below_static": int((features.temporal_seed_reach_upper < features.seed_reach).sum()),
                  "nodes_with_strict_below_upper": int((features.temporal_seed_reach_strict_days < features.temporal_seed_reach_upper).sum()),
                  "interpretation": "Date-compatible paths only; same-day order is unknown, amounts and provenance are not traced"},
              "sensitivity": sensitivity,
              "thresholds": thresholds, "runtime_seconds": round(time.perf_counter() - started, 4),
              "python": platform.python_version(),
              "packages": {p: importlib.metadata.version(p) for p in ("pandas", "pyarrow", "networkx", "numpy", "scipy")},
              "input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.data.glob("*.parquet"))},
              "limits": ["Observed internal-bank July 2026 transfers >=5000 KZT only", "Depth-4 outgoing flows not collected", "Scores are rule strength and review priority, not crime probabilities", "Date-only temporal compatibility is not provenance of funds", "No labelled ground truth; precision/recall cannot be claimed"]}
    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    result = validate_outputs(args.data, args.out)
    # Include validation time in the deadline check as well.
    report["runtime_seconds"] = round(time.perf_counter() - started, 4)
    if report["runtime_seconds"] >= 300:
        raise RuntimeError("Full pipeline exceeded 5 minutes")
    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({**result, "runtime_seconds": report["runtime_seconds"], "role_counts": report["role_counts"]}, ensure_ascii=False, indent=2))
    print(f"Open {args.out / 'network.html'} in any modern browser (no server, no internet).")


if __name__ == "__main__":
    main()
