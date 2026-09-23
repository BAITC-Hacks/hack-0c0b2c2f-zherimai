"""Strict input checks; all clients are preserved, including isolated seeds."""
from pathlib import Path
import networkx as nx
import numpy as np
import pandas as pd


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_data(folder: Path):
    tables = {name: pd.read_parquet(folder / f"{name}.parquet")
              for name in ("nodes", "edges", "transactions")}
    required = {"nodes": {"gid", "depth", "is_seed"},
                "edges": {"src", "dst", "sum_kzt", "n_tx", "depth"},
                "transactions": {"src", "dst", "date", "sum_kzt"}}
    for name, columns in required.items():
        require(columns.issubset(tables[name].columns), f"{name}: missing columns")
        require(not tables[name][sorted(columns)].isna().any().any(), f"{name}: null input")
    nodes, edges, tx = (tables[n] for n in ("nodes", "edges", "transactions"))
    for table, cols in ((nodes, ["gid", "depth"]), (edges, ["src", "dst", "n_tx", "depth"]), (tx, ["src", "dst"])):
        for col in cols:
            require(pd.api.types.is_integer_dtype(table[col]), f"{col}: integer dtype required")
    require(pd.api.types.is_bool_dtype(nodes.is_seed), "is_seed must be boolean")
    require(nodes.gid.is_unique, "Duplicate node gid")
    require(not edges.duplicated(["src", "dst"]).any(), "Duplicate directed edge")
    require(nodes.depth.between(0, 4).all(), "Node depth outside 0..4")
    require(edges.depth.between(1, 4).all(), "Edge depth outside 1..4")
    require((nodes.is_seed == (nodes.depth == 0)).all(), "Seed/depth mismatch")
    gids = set(nodes.gid)
    for table in (edges, tx):
        require(set(table.src).union(table.dst).issubset(gids), "Unknown endpoint gid")
        require(np.isfinite(table.sum_kzt).all() and (table.sum_kzt > 0).all(), "Invalid amount")
    require((edges.n_tx > 0).all(), "n_tx must be positive")
    tx["date"] = pd.to_datetime(tx.date, errors="raise")
    require(tx.date.between("2026-07-01", "2026-07-31").all(), "Transactions outside July 2026")
    require((tx.sum_kzt >= 5000).all(), "Transaction below dataset threshold")
    aggregate = tx.groupby(["src", "dst"], as_index=False).agg(total=("sum_kzt", "sum"), count=("sum_kzt", "size"))
    merged = edges.merge(aggregate, on=["src", "dst"], how="outer", indicator=True, validate="one_to_one")
    require(merged._merge.eq("both").all(), "Transaction/edge pairs differ")
    require(np.allclose(merged.sum_kzt, merged.total, rtol=0, atol=.01), "Transaction/edge amounts differ")
    require((merged.n_tx == merged['count']).all(), "Transaction/edge counts differ")
    nodes = nodes.sort_values("gid").reset_index(drop=True)
    edges = edges.sort_values(["src", "dst"]).reset_index(drop=True)
    tx = tx.sort_values(["date", "src", "dst", "sum_kzt"]).reset_index(drop=True)
    graph = nx.DiGraph()
    graph.add_nodes_from(int(g) for g in nodes.gid)
    for r in edges.itertuples(index=False):
        graph.add_edge(int(r.src), int(r.dst), sum_kzt=float(r.sum_kzt), n_tx=int(r.n_tx), depth=int(r.depth))
    return nodes, edges, tx, graph
