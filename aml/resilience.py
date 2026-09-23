"""Static graph sensitivity, not a causal claim about real-world intervention."""
import networkx as nx
import pandas as pd


def reachable_pairs(graph, seeds, targets):
    return {(seed, target) for seed in seeds if seed in graph
            for target in (nx.descendants(graph, seed) & targets) if target != seed}


def analyze_resilience(df, graph):
    seeds = set(df.loc[df.is_seed, "gid"])
    targets = set(df.loc[df.base_role == "consolidator", "gid"])
    candidates = df.loc[~df.is_seed].sort_values(["priority_score", "gid"], ascending=[False, True])
    baseline = reachable_pairs(graph, seeds, targets)
    rows = []
    for count in (0, 5, 10, 20):
        removed = set(candidates.head(count).gid)
        remaining = graph.copy()
        remaining.remove_nodes_from(removed)
        after = reachable_pairs(remaining, seeds, targets - removed)
        surviving_baseline = {(s, t) for s, t in baseline if t not in removed}
        rows.append({"removed_n": len(removed), "removed_gids": ";".join(str(int(g)) for g in candidates.head(count).gid) or "none",
                     "baseline_seed_target_pairs": len(baseline), "remaining_seed_target_pairs": len(after),
                     "lost_pair_share": (len(baseline - after) / len(baseline)) if baseline else 0.,
                     "removed_consolidators": len(removed & targets),
                     "lost_pairs_to_surviving_targets": len(surviving_baseline - after),
                     "seeds_losing_all_targets": len({s for s, _ in baseline} - {s for s, _ in after}),
                     "weak_components_remaining": nx.number_weakly_connected_components(remaining) if remaining else 0})
    return pd.DataFrame(rows)
