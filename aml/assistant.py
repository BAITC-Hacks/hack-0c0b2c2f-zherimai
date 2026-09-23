"""Offline tool-based query assistant; deterministic routing, no language model."""
import argparse
import json
from pathlib import Path
import re
import networkx as nx
import pandas as pd
from .load import load_data


GID_PATTERN = r"[0-9]{15,20}"
GID_LIST_PATTERN = rf"{GID_PATTERN}(?:(?:\s*[,;]\s*|\s+(?:и|and)\s+|\s+){GID_PATTERN})*"
HELP_EXAMPLES = ["Объясни <gid>", "Путь от <gid> до <gid>",
                 "Общие получатели <gid> и <gid>", "Кластер <номер>"]


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
        # A gid in arbitrary prose is not a request to explain that node.
        # Match the whole supported command so malformed or extra arguments
        # cannot silently turn into a different query.
        q = question.strip().rstrip(".?!").strip()
        tool, result = "help", {
            "examples": list(HELP_EXAMPLES),
            "message": "Это локальный помощник по правилам, без LLM. Поддерживаются только команды из примеров или полный gid; произвольные вопросы не поддерживаются.",
        }
        if re.match(r"(?:кластер|cluster)\b", q, re.IGNORECASE):
            tool = "cluster_summary"
            cluster = re.fullmatch(r"(?:кластер|cluster)\s*[#№]?\s*([0-9]{1,5})", q, re.IGNORECASE)
            result = (self.cluster_summary(int(cluster.group(1))) if cluster else
                      {"error": "Укажите целый неотрицательный номер: «Кластер <номер>»."})
        elif re.match(r"(?:путь|пути|path)\b", q, re.IGNORECASE):
            tool = "paths"
            path = re.fullmatch(
                rf"(?:путь|пути|path)\s+(?:(?:от|from)\s+)?({GID_PATTERN})"
                rf"\s+(?:(?:до|к|to|->|→)\s*)?({GID_PATTERN})", q, re.IGNORECASE)
            result = (self.paths(*path.groups()) if path else
                      {"error": "Укажите ровно два gid: «Путь от <gid> до <gid>»."})
        elif re.match(r"(?:общие\s+получатели|common(?:\s+receivers)?)\b", q, re.IGNORECASE):
            tool = "common_receivers"
            common = re.fullmatch(
                rf"(?:общие\s+получатели|common(?:\s+receivers)?)\s+({GID_LIST_PATTERN})",
                q, re.IGNORECASE)
            result = (self.common_receivers(re.findall(GID_PATTERN, common.group(1))) if common else
                      {"error": "Укажите минимум два разных gid: «Общие получатели <gid> и <gid>»."})
        else:
            node = re.fullmatch(
                rf"(?:(?:объясни|покажи|узел|explain|show|node)\s+)?({GID_LIST_PATTERN})",
                q, re.IGNORECASE)
            if node:
                tool, result = "get_node", [self.get_node(g) for g in re.findall(GID_PATTERN, node.group(1))]
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
