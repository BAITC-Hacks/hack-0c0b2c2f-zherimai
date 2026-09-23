"""Observed metrics, with explicit missing-observation flags."""
import networkx as nx
import numpy as np
import pandas as pd


def temporal_match(incoming, outgoing):
    """Conservative FIFO amount match on days 1..2; never reuse outgoing amount.

    Daily dates cannot order same-day events. This is a temporal compatibility
    indicator, NOT tracing the origin of funds or estimating a client's balance.
    """
    incoming = sorted(incoming)
    outgoing = [[day, float(amount)] for day, amount in sorted(outgoing)]
    total = sum(amount for _, amount in incoming)
    matched, weighted_lag = 0., 0.
    for day, amount in incoming:
        remaining = float(amount)
        for entry in outgoing:
            lag = (entry[0] - day).days
            if lag < 1 or entry[1] <= 0:
                continue
            if lag > 2:
                break
            take = min(remaining, entry[1])
            entry[1] -= take
            remaining -= take
            matched += take
            weighted_lag += take * lag
            if remaining <= 0:
                break
    return (matched / total if total else 0., weighted_lag / matched if matched else 0.)


def compute_features(nodes, edges, tx, graph):
    df = nodes.copy()
    for name, values in {
        "in_deg": dict(graph.in_degree()), "out_deg": dict(graph.out_degree()),
        "in_kzt": dict(graph.in_degree(weight="sum_kzt")),
        "out_kzt": dict(graph.out_degree(weight="sum_kzt")),
        "in_tx": dict(graph.in_degree(weight="n_tx")),
        "out_tx": dict(graph.out_degree(weight="n_tx")),
        "pagerank": nx.pagerank(graph, weight="sum_kzt", tol=1e-10, max_iter=1000),
        "betweenness": nx.betweenness_centrality(graph, weight=None),
    }.items():
        df[name] = df.gid.map(values).fillna(0)
    for col in ("in_deg", "out_deg", "in_tx", "out_tx"):
        df[col] = df[col].astype(int)
    df["pass_through_defined"] = df.in_kzt > 0
    df["pass_through"] = np.divide(df.out_kzt, df.in_kzt, out=np.zeros(len(df)), where=df.in_kzt > 0)
    df["balance_comparable"] = df.pass_through_defined & ~df.is_seed
    df["truncated_by_depth"] = (df.depth == 4) & (df.out_deg == 0)
    df["unobserved_funding"] = (df.out_kzt > df.in_kzt * 1.2) & (df.out_kzt > 0)
    seeds = set(int(g) for g in df.loc[df.is_seed, "gid"])
    reach = dict.fromkeys(graph, 0)
    for seed in sorted(seeds):
        for gid in nx.descendants(graph, seed):
            reach[gid] += 1
    df["seed_reach"] = df.gid.map(reach).astype(int)
    df["seed_payers"] = [len(set(graph.predecessors(int(g))) & seeds) for g in df.gid]
    cyclic = set()
    for comp in nx.strongly_connected_components(graph):
        if len(comp) > 1:
            cyclic.update(comp)
        elif graph.has_edge(next(iter(comp)), next(iter(comp))):
            cyclic.update(comp)
    df["in_cycle"] = df.gid.isin(cyclic)
    df["reciprocal_neighbors"] = [len(set(graph.predecessors(int(g))) & set(graph.successors(int(g)))) for g in df.gid]
    df["repeat_routes"] = [sum(graph[a][b]["n_tx"] > 1 for a, b in set(graph.in_edges(int(g))) | set(graph.out_edges(int(g)))) for g in df.gid]
    incoming = {g: list(zip(f.date, f.sum_kzt)) for g, f in tx.groupby("dst")}
    outgoing = {g: list(zip(f.date, f.sum_kzt)) for g, f in tx.groupby("src")}
    matches = [temporal_match(incoming.get(g, []), outgoing.get(g, [])) for g in df.gid]
    df["fast_out_share"] = [m[0] for m in matches]
    df["matched_lag_days"] = [m[1] for m in matches]
    sync = tx.groupby(["dst", "date"]).src.nunique().groupby("dst").max()
    df["max_payers_same_day"] = df.gid.map(sync).fillna(0).astype(int)
    incident = pd.concat([tx.rename(columns={"src": "gid"})[["gid", "date", "sum_kzt"]],
                          tx.rename(columns={"dst": "gid"})[["gid", "date", "sum_kzt"]]])
    daily = incident.groupby(["gid", "date"]).sum_kzt.sum()
    bursts = daily.groupby("gid").max() / daily.groupby("gid").sum()
    df["burst_share"] = df.gid.map(bursts).fillna(0)
    near = incident.assign(near=incident.sum_kzt.between(5000, 10000, inclusive="left")).groupby("gid").near.mean()
    df["near_threshold_share"] = df.gid.map(near).fillna(0)
    for prefix in ("in", "out"):
        df[f"{prefix}_avg_kzt"] = np.divide(df[f"{prefix}_kzt"], df[f"{prefix}_tx"], out=np.zeros(len(df)), where=df[f"{prefix}_tx"] > 0)
    df["volume_depth_percentile"] = np.log1p(df.in_kzt + df.out_kzt).groupby(df.depth).rank(pct=True, method="average")
    return df
