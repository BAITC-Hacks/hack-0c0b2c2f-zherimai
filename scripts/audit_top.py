"""Read-only independent checks of saved features against original transactions.

Run from the repository root: .venv/bin/python scripts/audit_top.py
No aml implementation is imported. Temporal compatibility is independently
checked by integer-capacity maximum flow over daily input/output amounts.
"""
from collections import Counter, defaultdict, deque
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


def main():
    root = Path(__file__).resolve().parents[1]
    nodes = pd.read_parquet(root / "data/nodes.parquet")
    edges = pd.read_parquet(root / "data/edges.parquet")
    tx = pd.read_parquet(root / "data/transactions.parquet")
    tx["date"] = pd.to_datetime(tx.date)
    saved = pd.read_csv(root / "out/nodes_roles.csv").set_index("gid")
    top = pd.read_csv(root / "out/top_nodes.csv").head(20)
    seeds = set(nodes.loc[nodes.is_seed, "gid"])
    adjacency = defaultdict(set)
    predecessors = defaultdict(set)
    for src, dst in zip(tx.src, tx.dst):
        adjacency[int(src)].add(int(dst))
        predecessors[int(dst)].add(int(src))

    def reachable(seed, removed=frozenset()):
        if seed in removed:
            return set()
        seen = {seed}
        queue = deque([seed])
        while queue:
            for target in adjacency[queue.popleft()] - removed - seen:
                seen.add(target)
                queue.append(target)
        return seen - {seed}

    reach_counts = Counter(g for seed in seeds for g in reachable(seed))
    # Diagnostic only: structural reachability can join transfers in reverse
    # date order. These alternatives restrict paths to increasing dates, or
    # permit any possible within-day ordering as an optimistic upper bound.
    # Neither construction establishes that the same money followed the path.
    strict_date_reach, possible_same_day_reach = Counter(), Counter()
    day_groups = []
    for _, frame in tx.sort_values("date").groupby("date"):
        day_adjacency = defaultdict(set)
        for src, dst in zip(frame.src, frame.dst):
            day_adjacency[int(src)].add(int(dst))
        day_groups.append(day_adjacency)
    for seed in seeds:
        strict_seen, possible_seen = {seed}, {seed}
        for day_adjacency in day_groups:
            strict_seen |= {target for source in tuple(strict_seen) for target in day_adjacency[source]}
            queue = deque(possible_seen)
            while queue:
                for target in day_adjacency[queue.popleft()] - possible_seen:
                    possible_seen.add(target)
                    queue.append(target)
        strict_date_reach.update(strict_seen - {seed})
        possible_same_day_reach.update(possible_seen - {seed})
    expected = pd.DataFrame(index=nodes.gid)
    for prefix, column, other in (("in", "dst", "src"), ("out", "src", "dst")):
        grouped = tx.groupby(column)
        expected[f"{prefix}_deg"] = grouped[other].nunique()
        expected[f"{prefix}_kzt"] = grouped.sum_kzt.sum()
        expected[f"{prefix}_tx"] = grouped.size()
    expected["seed_reach"] = pd.Series(reach_counts)
    expected["seed_payers"] = pd.Series({g: len(predecessors[g] & seeds) for g in nodes.gid})
    expected["max_payers_same_day"] = tx.groupby(["dst", "date"]).src.nunique().groupby("dst").max()
    for column, counts in (("temporal_seed_reach_upper", possible_same_day_reach),
                           ("temporal_seed_reach_strict_days", strict_date_reach)):
        if column in saved:
            expected[column] = pd.Series(counts)
    expected = expected.fillna(0)
    mismatches = {}
    if not saved.index.is_unique or set(saved.index) != set(nodes.gid):
        mismatches["node_coverage"] = True
    if ((saved.depth == 4) & saved.role.eq("terminal")).any():
        mismatches["terminal_at_depth4"] = True
    if not saved[["role_score", "priority_score"]].apply(lambda s: s.between(0, 1)).all().all():
        mismatches["score_range"] = True
    for column in expected:
        actual = saved.loc[expected.index, column]
        bad = ~np.isclose(actual, expected[column], atol=1e-6, rtol=0)
        if bad.any():
            mismatches[column] = [str(g) for g in expected.index[bad]]

    # Matching over dates and integer cents; unlike production FIFO, use a
    # bipartite maximum flow with only 1- and 2-day input->output links.
    incoming = {g: frame.groupby("date").sum_kzt.sum() for g, frame in tx.groupby("dst")}
    outgoing = {g: frame.groupby("date").sum_kzt.sum() for g, frame in tx.groupby("src")}
    max_temporal_error = 0.0
    temporal_mismatches = []
    for gid in nodes.gid:
        ins, outs = incoming.get(gid), outgoing.get(gid)
        share = 0.0
        if ins is not None and outs is not None:
            graph = nx.DiGraph()
            total_cents = 0
            for day, amount in ins.items():
                cents = int(round(amount * 100))
                total_cents += cents
                graph.add_edge("source", ("in", day), capacity=cents)
            for day, amount in outs.items():
                graph.add_edge(("out", day), "sink", capacity=int(round(amount * 100)))
            for day in ins.index:
                for lag in (1, 2):
                    outday = day + pd.Timedelta(days=lag)
                    if outday in outs.index:
                        graph.add_edge(("in", day), ("out", outday), capacity=total_cents)
            share = nx.maximum_flow_value(graph, "source", "sink") / total_cents
        error = abs(share - saved.loc[gid, "fast_out_share"])
        max_temporal_error = max(max_temporal_error, error)
        if error > 1e-9:
            temporal_mismatches.append(str(gid))
    if temporal_mismatches:
        mismatches["fast_out_share_maxflow"] = temporal_mismatches

    structural = {g for g in nodes.gid if len(predecessors[g]) >= 5 or len(adjacency[g]) >= 10}
    for gid, row in saved.iterrows():
        expected_neighbors = len((predecessors[gid] | adjacency[gid]) & structural)
        if expected_neighbors != row.structural_neighbors:
            mismatches.setdefault("structural_neighbors", []).append(str(gid))

    targets = set(saved.index[saved.base_role == "consolidator"])
    baseline = {(seed, target) for seed in seeds for target in reachable(seed) & targets}
    resilience = pd.read_csv(root / "out/resilience.csv")
    ranked_nonseed = saved.loc[~saved.is_seed].reset_index().sort_values(
        ["priority_score", "gid"], ascending=[False, True])
    resilience_rows = []
    for row in resilience.itertuples(index=False):
        removed = set(ranked_nonseed.head(row.removed_n).gid)
        after = {(seed, target) for seed in seeds for target in reachable(seed, removed) & (targets - removed)}
        surviving_baseline = {(seed, target) for seed, target in baseline if target not in removed}
        metrics = {"baseline_seed_target_pairs": len(baseline),
                   "remaining_seed_target_pairs": len(after),
                   "removed_consolidators": len(removed & targets),
                   "lost_pairs_to_surviving_targets": len(surviving_baseline - after),
                   "seeds_losing_all_targets": len({s for s, _ in baseline} - {s for s, _ in after})}
        for name, value in metrics.items():
            if value != getattr(row, name):
                mismatches.setdefault("resilience", []).append(f"N={row.removed_n}:{name}")
        resilience_rows.append({"removed_n": row.removed_n, **metrics})

    evidence_lengths = saved.evidence.str.len()
    components = ["priority_role", "priority_reach", "priority_volume", "priority_pagerank", "priority_patterns"]
    score_from_components = (saved[components].sum(axis=1) * saved.seed_discount).round(8)
    if not np.allclose(score_from_components, saved.priority_score, atol=1e-8, rtol=0):
        mismatches["priority_components"] = True
    full_ranking = saved.reset_index().sort_values(["priority_score", "gid"], ascending=[False, True])
    if list(full_ranking.head(20).gid) != list(top.gid):
        mismatches["top_order"] = True
    details = saved.loc[top.gid].copy()
    details["gid"] = details.index.map(str)
    details["rank"] = list(range(1, 21))
    details["strict_date_seed_reach"] = [strict_date_reach[g] for g in details.index]
    details["same_day_possible_seed_reach"] = [possible_same_day_reach[g] for g in details.index]
    proof_columns = ["rank", "gid", "role", "base_role", "in_deg", "out_deg", "in_kzt", "out_kzt",
                     "seed_payers", "seed_reach", "fast_out_share", "structural_neighbors",
                     "betweenness", "priority_score", "role_score", "unobserved_funding",
                     "strict_date_seed_reach", "same_day_possible_seed_reach"]
    aggregate = tx.groupby(["src", "dst"]).agg(sum_kzt=("sum_kzt", "sum"), n_tx=("sum_kzt", "size"))
    aligned = edges.set_index(["src", "dst"]).sort_index()
    aggregate = aggregate.sort_index()
    if not aggregate.index.equals(aligned.index):
        mismatches["edge_pairs"] = True
    elif not np.allclose(aggregate.sum_kzt, aligned.sum_kzt, rtol=0, atol=0.01) or not (aggregate.n_tx == aligned.n_tx).all():
        mismatches["edge_aggregates"] = True
    output = {
        "input_nodes": len(nodes), "input_transactions": len(tx),
        "mismatches": mismatches, "max_temporal_share_error": max_temporal_error,
        "evidence_max_length": int(evidence_lengths.max()),
        "evidence_at_200_chars": int(evidence_lengths.eq(200).sum()),
        "top20_coordinators": int(details.role.eq("coordinator").sum()),
        "top20_unobserved_funding": int(details.unobserved_funding.sum()),
        "top20_both_collection_and_distribution": int(((details.in_deg >= 5) & (details.out_deg >= 10)).sum()),
        "top20_in_directed_cycle": int(details.in_cycle.sum()),
        "isolated_nodes_preserved": int(((saved.in_deg + saved.out_deg) == 0).sum()),
        "terminals_at_depth4": int(((saved.depth == 4) & saved.role.eq("terminal")).sum()),
        "top20": details[proof_columns].to_dict("records"),
        "resilience": resilience_rows,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
