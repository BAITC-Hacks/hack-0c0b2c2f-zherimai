"""Offline tool-based query assistant; deterministic routing, no language model."""
import argparse
import json
from pathlib import Path
import re
import networkx as nx
import pandas as pd
from .load import load_data


class AnalystAssistant:
    def __init__(self, data_dir=Path("data"), out_dir=Path("out")):
        self.nodes = pd.read_csv(out_dir / "nodes_roles.csv", dtype={"gid": str}).set_index("gid")
        self.clusters = pd.read_csv(out_dir / "clusters.csv")
        _, _, _, self.graph = load_data(data_dir)

    def get_node(self, gid):
        gid = str(gid)
        if gid not in self.nodes.index:
            return {"error": "gid отсутствует в наборе", "gid": gid}
        row = self.nodes.loc[gid]
        values = json.loads(row.to_json(force_ascii=False))
        values["gid"] = gid
        values["incoming_gids"] = [str(g) for g in sorted(self.graph.predecessors(int(gid)))]
        values["outgoing_gids"] = [str(g) for g in sorted(self.graph.successors(int(gid)))]
        values["limitation"] = "Роль — гипотеза по выборке, не вывод о виновности. Баланс и назначение переводов неизвестны."
        return values

    def paths(self, source, target):
        if str(source) not in self.nodes.index or str(target) not in self.nodes.index:
            return {"error": "Один из gid отсутствует"}
        try:
            path = nx.shortest_path(self.graph, int(source), int(target))
        except nx.NetworkXNoPath:
            return {"path": [], "explanation": "В наблюдаемом направленном графе путь отсутствует"}
        return {"path": [str(g) for g in path], "hops": len(path) - 1,
                "limitation": "Структурный путь не подтверждает временную последовательность и происхождение денег"}

    def common_receivers(self, gids):
        gids = list(dict.fromkeys(str(g) for g in gids))
        if len(gids) < 2:
            return {"error": "Укажите минимум два разных gid"}
        if any(str(g) not in self.nodes.index for g in gids):
            return {"error": "Один из gid отсутствует"}
        common = set.intersection(*(set(self.graph.successors(int(g))) for g in gids))
        return {"source_gids": [str(g) for g in gids], "common_receiver_gids": [str(g) for g in sorted(common)]}

    def cluster_summary(self, cluster_id):
        rows = self.clusters.loc[self.clusters.cluster_id == int(cluster_id)]
        if rows.empty:
            return {"error": "Кластер отсутствует"}
        return json.loads(rows.iloc[0].to_json(force_ascii=False))

    def answer(self, question):
        gids = re.findall(r"(?<!\d)\d{15,20}(?!\d)", question)
        q = question.lower()
        cluster = re.search(r"(?:кластер|cluster)\s*#?\s*(\d{1,5})(?!\d)", q)
        if cluster:
            tool, result = "cluster_summary", self.cluster_summary(int(cluster.group(1)))
        elif len(gids) >= 2 and any(word in q for word in ("общ", "common", "собира", "сбор")):
            tool, result = "common_receivers", self.common_receivers(gids)
        elif len(gids) >= 2 and any(word in q for word in ("путь", "пути", "path", "связ")):
            tool, result = "paths", self.paths(gids[0], gids[1])
        elif gids:
            tool, result = "get_node", [self.get_node(g) for g in gids]
        else:
            tool, result = "help", {"examples": ["Объясни <gid>", "Путь от <gid> до <gid>", "Общие получатели <gid> и <gid>", "Кластер <номер>"],
                                     "message": "Это локальный помощник по правилам, без LLM. Укажите gid или номер кластера; произвольные вопросы не поддерживаются."}
        return {"mode": "offline_deterministic", "tool": tool, "result": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("out"))
    parser.add_argument("--question", required=True)
    args = parser.parse_args()
    print(json.dumps(AnalystAssistant(args.data, args.out).answer(args.question), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
