"""Only community detection ignores edge direction; all flow metrics stay directed."""
import math
import networkx as nx
import pandas as pd


def assign_clusters(df, graph):
    projection = nx.Graph()
    projection.add_nodes_from(g for g in graph if graph.degree(g) > 0)
    for src, dst, data in graph.edges(data=True):
        weight = float(data["sum_kzt"])
        if projection.has_edge(src, dst):
            projection[src][dst]["sum_kzt"] += weight
        else:
            projection.add_edge(src, dst, sum_kzt=weight)
    for _, _, data in projection.edges(data=True):
        data["weight"] = math.log1p(data["sum_kzt"])
    if projection.number_of_edges():
        communities = nx.community.louvain_communities(projection, weight="weight", seed=42, resolution=1)
    else:
        communities = [{g} for g in projection]
    communities = sorted(communities, key=lambda c: (-len(c), min(c)))
    mapping = {g: i for i, community in enumerate(communities, 1) for g in community}
    result = df.copy()
    result["cluster_id"] = result.gid.map(mapping).fillna(0).astype(int)
    lookup = dict(zip(result.gid, result.cluster_id))
    result["neighbor_clusters"] = [len({lookup[n] for n in set(graph.predecessors(int(g))) | set(graph.successors(int(g)))}) for g in result.gid]
    return result, projection


def summarize_clusters(df, edges):
    membership = dict(zip(df.gid, df.cluster_id))
    internal = {}
    for e in edges.itertuples(index=False):
        cluster = membership[e.src]
        if cluster == membership[e.dst]:
            internal[cluster] = internal.get(cluster, 0.) + float(e.sum_kzt)
    rows = []
    for cluster, part in df.groupby("cluster_id", sort=True):
        roles = part.role.value_counts()
        if cluster == 0:
            hypothesis = "Нет переводов в выгрузке; техническая группа изолятов, не сообщество."
        elif ((part.base_role == "consolidator") & (part.seed_payers >= 3)).any():
            hypothesis = "Признаки совместного сбора средств от нескольких seed; проверить назначение переводов."
        elif (part.base_role == "distributor").any():
            hypothesis = "Признаки веерного распределения; проверить экономическое основание выплат."
        elif roles.get("transit", 0) >= 2:
            hypothesis = "Возможный транзитный коридор; уточнить последовательность по времени."
        else:
            hypothesis = "Связанная активность в ограниченной выборке; общность цели не установлена."
        top = part.sort_values(["priority_score", "gid"], ascending=[False, True]).head(5)
        rows.append({"cluster_id": int(cluster), "n_nodes": len(part), "n_seed": int(part.is_seed.sum()),
                     "sum_kzt_internal": internal.get(cluster, 0.), "top_gids": ";".join(str(int(g)) for g in top.gid),
                     "hypothesis": hypothesis})
    return pd.DataFrame(rows)
