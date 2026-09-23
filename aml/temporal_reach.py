"""Date-compatible reachability, without tracing amounts or money provenance."""
from collections import defaultdict, deque

import pandas as pd


def temporal_seed_reach(nodes, tx):
    """Count other seeds having a date-compatible directed path to every node.

    ``temporal_seed_reach_upper`` permits a possible ordering of all transfers
    on the same day. This is an upper bound because intraday order is unknown.
    ``temporal_seed_reach_strict_days`` requires each next transfer to occur on
    a later day, so only the previous-day reachability snapshot can propagate.

    Amounts and balances are deliberately ignored. Neither measure proves that
    funds from a seed reached a client. Inputs are the validated node and
    transaction tables; each seed starts reachable before the observed dates,
    and never counts itself even when a directed cycle returns to it.
    """
    gids = [int(gid) for gid in nodes.gid]
    upper_counts = dict.fromkeys(gids, 0)
    strict_counts = dict.fromkeys(gids, 0)
    dated = tx[["src", "dst", "date"]].copy()
    dated["date"] = pd.to_datetime(dated.date, errors="raise").dt.normalize()
    daily_graphs = []
    for _, day in dated.groupby("date", sort=True):
        adjacency = defaultdict(set)
        for src, dst in zip(day.src, day.dst):
            adjacency[int(src)].add(int(dst))
        daily_graphs.append((dict(adjacency), set(adjacency)))

    seeds = sorted(int(gid) for gid in nodes.loc[nodes.is_seed, "gid"])
    for seed in seeds:
        upper_seen, strict_seen = {seed}, {seed}
        for adjacency, sources in daily_graphs:
            # Compute all additions before changing strict_seen: paths may
            # traverse only one edge per date in this conservative variant.
            strict_additions = {target for source in (strict_seen & sources)
                                for target in adjacency[source]}
            strict_seen.update(strict_additions)

            # Closure represents a possible intraday ordering, not an observed
            # ordering. Seeding the queue from all previously reached sources
            # also permits an arbitrarily long wait between transfers.
            queue = deque(upper_seen & sources)
            while queue:
                for target in adjacency.get(queue.popleft(), ()):
                    if target not in upper_seen:
                        upper_seen.add(target)
                        queue.append(target)
        for gid in upper_seen - {seed}:
            upper_counts[gid] += 1
        for gid in strict_seen - {seed}:
            strict_counts[gid] += 1

    result = nodes[["gid"]].copy()
    result["temporal_seed_reach_upper"] = result.gid.map(upper_counts).astype("int64")
    result["temporal_seed_reach_strict_days"] = result.gid.map(strict_counts).astype("int64")
    return result
