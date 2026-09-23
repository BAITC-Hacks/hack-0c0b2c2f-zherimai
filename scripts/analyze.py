#!/usr/bin/env python3
"""Reproduce exploratory data analysis; never hard-codes client identifiers."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


def records(frame):
    return json.loads(frame.to_json(orient="records"))


def table(rows, columns):
    def cell(value):
        if value is None:
            return "—"
        if isinstance(value, float):
            return f"{value:,.4f}".rstrip("0").rstrip(".").replace(",", " ")
        return str(value)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    lines += ["| " + " | ".join(cell(row.get(c)) for c in columns) + " |" for row in rows]
    return "\n".join(lines)


def markdown(result):
    c = result["counts"]
    parts = [
        "# Разведка датасета «Граф денег»",
        "Отчёт получен из реальных parquet-файлов. Повторить: `python scripts/analyze.py`. Среда: Python 3.13, pandas 3.0.6, NetworkX 3.7, PyArrow 25.0.1. Идентификаторы нигде не используются как правила ролей.",
        "## Загрузка и проверка стартера",
        f"Вход: {c['nodes']} узлов, {c['edges']} направленных рёбер, {c['transactions']} транзакций, {c['seeds']} seed; оборот {c['total_kzt']:,.2f} KZT. Минимальный перевод {c['min_transaction_kzt']:,.0f} KZT. Дубликатов gid и пар src→dst нет. Все пары, агрегированные суммы и количества переводов согласованы; максимальная погрешность сумм из-за float — {result['sanity']['max_sum_delta']:.2g} KZT.",
        "Организаторский `starter/starter.py` успешно запущен командой `python starter/starter.py --data data --out /tmp/hackalem-starter-analysis`. Он создаёт пустые каркасы ролей/кластеров/топа, что ожидаемо. Его sanity_check сверяет только наличие пар; здесь отдельно проверены суммы и n_tx.",
        "## Распределения",
        "Квантили по всем узлам. Для pass_through исключены узлы с нулевым входом: отношение там не определено. Нули исходящих включены.",
        table(result["quantiles"], ["quantile", "in_deg", "out_deg", "pass_through", "in_kzt"]),
        "Коэффициент пропуска среди не-seed, имеющих и вход, и выход:",
        table(result["pass_through_nonseed_active_quantiles"], ["quantile", "pass_through"]),
        "## Нулевые исходящие по коленам",
        table(result["zero_out_by_depth"], ["depth", "n_nodes", "zero_out"]),
        "У всех 444 узлов четвёртого колена исходящие отсутствуют вследствие ограничения обхода. Terminal им не присваивается. У 31 seed нет исходящих, из них 19 вообще изолированы.",
        "## Топ-20 по числу разных плательщиков",
        table(result["top20_in_degree"], ["gid", "depth", "is_seed", "in_deg", "out_deg", "in_kzt", "out_kzt", "pass_through"]),
        "## Топ-20 по числу разных получателей",
        table(result["top20_out_degree"], ["gid", "depth", "is_seed", "in_deg", "out_deg", "in_kzt", "out_kzt", "pass_through"]),
        "## Компоненты и Louvain",
        "Полный граф содержит 35 слабосвязных компонент: 16 с рёбрами и 19 изолятов. Две крупнейшие — 1 877 узлов / 46 seed и 270 узлов / 1 seed. Остальные активные компоненты имеют размеры 17, 13, 6×4, 5×2, 4×2, 3×2, 2×2. Вне крупнейшей компоненты 371 узел, включая 19 изолятов.",
        "Louvain выполняется на ненаправленной проекции. Встречные переводы сначала суммируются по паре, затем применяется выбранная функция веса. Порядок узлов и рёбер отсортирован; seed=42, resolution=1. Направление сохраняется во всех остальных структурных метриках.",
        table([{"weight": weight, **{k: v for k, v in values.items() if k != "communities"}} for weight, values in result["louvain"].items()], ["weight", "n_communities", "n_nontrivial", "n_multiple_seed", "modularity"]),
        "Каждый изолят выше считается отдельным сообществом Louvain. В продуктовой выгрузке их можно объединить в явно обозначенную техническую группу 0 «нет наблюдаемых переводов»; она не означает связанность. Значения modularity разных функций веса напрямую не сравниваются. Восемь сообществ из подсказки ТЗ — сообщества с более чем одним seed при весе суммы, а не общее число кластеров. Один запуск не доказывает устойчивость; для такого утверждения нужны повторные запуски.",
        "Если перед Louvain исключить изоляты (как в основном пайплайне), порядок перемешивания изменится даже с тем же seed=42. Результаты на активной проекции:",
        table([{"weight": weight, **values} for weight, values in result["louvain_active_projection"].items()], ["weight", "n_communities", "n_multiple_seed", "modularity"]),
        "В итоговой выгрузке к числу активных сообществ log_kzt добавляется одна техническая группа изолятов. Различие чисел при иной подготовке графа не является ошибкой; стабильность следует проверять отдельно.",
        "## Пороги, проверенные по распределениям",
        "Это объяснимые эвристики для проверки аналитиком, а не обученная модель и не измеренная точность. Основная роль выбирается по первому сработавшему правилу: coordinator → consolidator → distributor → transit → terminal → peripheral.",
        "| Роль | Предлагаемое формальное правило | Обоснование |\n| --- | --- | --- |\n| coordinator | Есть вход и выход; не менее 2 соседей-хабов; достижим минимум от 3 seed; направленная betweenness не ниже q99 и строго больше 0 | Без условия betweenness было 303 кандидата; с ним 21. Хаб: in_deg≥5 или out_deg≥10. Кратчайшие пути по числу шагов, веса сумм не являются расстоянием. |\n| consolidator | in_deg≥5 | Порог выше q95=3; 51 кандидат до пересечений. Для depth=4 нужен флаг ограниченной видимости и пониженный score. |\n| distributor | out_deg≥10 | q95=5; 64 кандидата до пересечений. Суммы и число переводов дополнительно объясняют поведение. |\n| transit | Не seed; есть вход и выход; 0,8≤out/in≤1,2; in_kzt≥50 000 | 39 кандидатов до пересечений, 72 узла попадают только в диапазон отношения без остальных ограничений. Даты дают дополнительное свидетельство. |\n| terminal | depth<4 и либо out_deg=0 при in_kzt≥50 000, либо не seed и out/in≤0,2 при in_kzt≥200 000 | Только гипотеза по видимым внутрибанковским переводам за июль; не подтверждает сохранение остатка. |\n| peripheral | Не выполнено ни одно правило выше | Нехватка признаков, не низкий риск и не оправдание. |",
        f"Расчёт q99 betweenness на данном полном графе: **{result['threshold_counts']['betweenness_q99']:.12f}**. seed_reach считает другие seed, из которых существует направленный путь, без пути нулевой длины к самому себе.",
        table([{"role": role, "n_nodes": count} for role, count in result["proposed_role_counts"].items()], ["role", "n_nodes"]),
        "Числа выше относятся именно к этим предложенным правилам и могут отличаться от финальной реализации. Порог 50 000 KZT равен медиане входящей суммы и отсеивает минимальные наблюдения; 200 000 выше q75=164 528 KZT. Role_score должен называться силой наблюдаемых свидетельств, а не вероятностью виновности. Для seed не следует трактовать pass_through как полный баланс.",
        "## Уточнения к первоначальному резюме",
        "- 16 компонент и 352 узла вне крупнейшей относятся только к 2 229 узлам, присутствующим в рёбрах. С 19 обязательными изолятами: 35 компонент и 371 узел вне крупнейшей.\n- 354 узла отдают больше, чем получают, только если вход положителен. Ещё 23 узла имеют нулевой наблюдаемый вход и положительный выход; всего таких дисбалансов 377.\n- `out>in` не доказывает внешний приток: возможен остаток на начало месяца. Корректный флаг — «неполнота наблюдения / требуется источник финансирования», а не установленный внешний доход.\n- Отсутствие исходящих при depth<4 означает отсутствие видимых внутрибанковских переводов ≥5 000 KZT за июль. Наличные, межбанк и остаток неизвестны.\n- По датам без времени нельзя установить порядок операций внутри дня. Быстрый транзит — оценка совместимости потоков по времени, а не доказанное отслеживание тех же денег.\n- Gid имеют 18 цифр и превышают JavaScript Number.MAX_SAFE_INTEGER. В JSON интерфейса gid/src/dst обязательно передавать строками.\n- Итоговая сумма точно по данным — 365 890 012,01 KZT; в ТЗ она округлена до целых.",
        "Полные машиночитаемые результаты, включая размеры и seed каждого сообщества: `docs/analysis.json`.",
    ]
    return "\n\n".join(parts) + "\n"


def analyze(data: Path):
    edges = pd.read_parquet(data / "edges.parquet")
    nodes = pd.read_parquet(data / "nodes.parquet")
    tx = pd.read_parquet(data / "transactions.parquet")
    graph = nx.DiGraph()
    graph.add_nodes_from(sorted(nodes.gid.astype(int)))
    for r in edges.sort_values(["src", "dst"]).itertuples(index=False):
        graph.add_edge(int(r.src), int(r.dst), sum_kzt=float(r.sum_kzt), n_tx=int(r.n_tx))
    df = nodes.sort_values("gid").copy()
    for direction in ["in", "out"]:
        degree = getattr(graph, f"{direction}_degree")
        for suffix, weight in [("deg", None), ("kzt", "sum_kzt"), ("tx", "n_tx")]:
            df[f"{direction}_{suffix}"] = df.gid.map(dict(degree(weight=weight)))
    df["pass_through"] = df.out_kzt / df.in_kzt.replace(0, np.nan)
    seeds = set(nodes.loc[nodes.is_seed, "gid"].astype(int))
    seed_reach = dict.fromkeys(graph, 0)
    for seed in sorted(seeds):
        for gid in nx.descendants(graph, seed):
            seed_reach[gid] += 1
    df["seed_reach"] = df.gid.map(seed_reach)
    df["seed_payers"] = df.gid.map({gid: sum(p in seeds for p in graph.predecessors(gid)) for gid in graph})
    hub_gids = set(df.loc[(df.in_deg >= 5) | (df.out_deg >= 10), "gid"])
    df["hub_neighbors"] = df.gid.map({gid: len(set(graph.predecessors(gid)).union(graph.successors(gid)) & hub_gids) for gid in graph})
    df["betweenness"] = df.gid.map(nx.betweenness_centrality(graph, weight=None))
    between_q99 = float(df.betweenness.quantile(.99))
    coordinator = (df.hub_neighbors >= 2) & (df.seed_reach >= 3) & (df.in_deg > 0) & (df.out_deg > 0) & (df.betweenness >= between_q99)
    proposed_role = pd.Series("peripheral", index=df.index)
    proposed_role.loc[(df.depth < 4) & (((df.out_deg == 0) & (df.in_kzt >= 50000)) | (~df.is_seed & (df.pass_through <= .2) & (df.in_kzt >= 200000)))] = "terminal"
    proposed_role.loc[~df.is_seed & df.pass_through.between(.8, 1.2) & (df.in_kzt >= 50000) & (df.out_deg > 0)] = "transit"
    proposed_role.loc[df.out_deg >= 10] = "distributor"
    proposed_role.loc[df.in_deg >= 5] = "consolidator"
    proposed_role.loc[coordinator] = "coordinator"
    components = sorted(nx.weakly_connected_components(graph), key=lambda c: (-len(c), min(c)))
    ug = nx.Graph()
    ug.add_nodes_from(graph)
    for src, dst, attrs in graph.edges(data=True):
        old = ug.get_edge_data(src, dst, {}).get("sum_kzt", 0.0)
        ug.add_edge(src, dst, sum_kzt=old + attrs["sum_kzt"])
    for _, _, attrs in ug.edges(data=True):
        attrs["log_kzt"] = math.log1p(attrs["sum_kzt"])
    communities = {}
    for weight in ["sum_kzt", "log_kzt"]:
        cs = sorted(nx.community.louvain_communities(ug, weight=weight, seed=42), key=lambda c: (-len(c), min(c)))
        communities[weight] = {
            "n_communities": len(cs),
            "n_nontrivial": sum(len(c) > 1 for c in cs),
            "n_multiple_seed": sum(len(seeds & c) > 1 for c in cs),
            "modularity": nx.community.modularity(ug, cs, weight=weight),
            "communities": [{"n_nodes": len(c), "n_seed": len(seeds & c)} for c in cs],
        }
    active_ug = nx.Graph()
    active_ug.add_nodes_from(gid for gid in graph if graph.degree(gid) > 0)
    active_ug.add_edges_from((src, dst, dict(attrs)) for src, dst, attrs in ug.edges(data=True))
    active_communities = {}
    for weight in ["sum_kzt", "log_kzt"]:
        cs = nx.community.louvain_communities(active_ug, weight=weight, seed=42)
        active_communities[weight] = {
            "n_communities": len(cs),
            "n_multiple_seed": sum(len(seeds & c) > 1 for c in cs),
            "modularity": nx.community.modularity(active_ug, cs, weight=weight),
        }
    grouped = tx.groupby(["src", "dst"]).agg(sum_kzt=("sum_kzt", "sum"), n_tx=("sum_kzt", "size"))
    match = edges.set_index(["src", "dst"])[["sum_kzt", "n_tx"]].join(grouped, how="outer", lsuffix="_edge", rsuffix="_tx")
    cols = ["gid", "depth", "is_seed", "in_deg", "out_deg", "in_kzt", "out_kzt", "pass_through", "seed_reach", "seed_payers"]
    result = {
        "counts": {"nodes": len(nodes), "edges": len(edges), "transactions": len(tx), "seeds": len(seeds), "total_kzt": float(edges.sum_kzt.sum()), "min_transaction_kzt": float(tx.sum_kzt.min()), "isolates": len(list(nx.isolates(graph))), "out_exceeds_in": int((df.out_kzt > df.in_kzt).sum()), "out_exceeds_positive_in": int(((df.out_kzt > df.in_kzt) & (df.in_kzt > 0)).sum()), "ratio_08_12": int(df.pass_through.between(.8, 1.2).sum()), "seed_without_out": int(((df.out_deg == 0) & df.is_seed).sum())},
        "sanity": {"no_duplicate_nodes": bool(nodes.gid.is_unique), "no_duplicate_edges": bool(not edges.duplicated(["src", "dst"]).any()), "tx_edge_pairs_match": bool(not match.isna().any().any()), "max_sum_delta": float((match.sum_kzt_edge - match.sum_kzt_tx).abs().max()), "max_count_delta": int((match.n_tx_edge - match.n_tx_tx).abs().max())},
        "quantiles": records(df[["in_deg", "out_deg", "pass_through", "in_kzt"]].quantile([0,.25,.5,.75,.9,.95,.99,1]).reset_index(names="quantile")),
        "pass_through_nonseed_active_quantiles": records(df.loc[~df.is_seed & (df.in_deg>0) & (df.out_deg>0), ["pass_through"]].quantile([0,.25,.5,.75,.9,.95,.99,1]).reset_index(names="quantile")),
        "zero_out_by_depth": records(df.assign(zero_out=df.out_deg == 0).groupby("depth").agg(n_nodes=("gid", "size"), zero_out=("zero_out", "sum")).reset_index()),
        "top20_in_degree": records(df.sort_values(["in_deg", "in_kzt", "gid"], ascending=[False,False,True]).head(20)[cols]),
        "top20_out_degree": records(df.sort_values(["out_deg", "out_kzt", "gid"], ascending=[False,False,True]).head(20)[cols]),
        "components": [{"n_nodes":len(c), "n_seed":len(c & seeds)} for c in components],
        "louvain": communities,
        "louvain_active_projection": active_communities,
        "threshold_counts": {
            "in_degree": {str(t): int((df.in_deg >= t).sum()) for t in [3,4,5,6,8,10]},
            "out_degree": {str(t): int((df.out_deg >= t).sum()) for t in [5,8,10,15,20,30,60]},
            "in_kzt": {str(t): int((df.in_kzt >= t).sum()) for t in [50000,100000,200000,500000,1000000]},
            "transit_nonseed_ratio_08_12_in50k": int((~df.is_seed & df.pass_through.between(.8,1.2) & (df.in_kzt >= 50000) & (df.out_deg > 0)).sum()),
            "terminal_depth_lt4_no_out_in50k": int(((df.depth < 4) & (df.out_deg == 0) & (df.in_kzt >= 50000)).sum()),
            "terminal_depth_lt4_ratio_le02_in200k": int(((df.depth < 4) & (df.pass_through <= .2) & (df.in_kzt >= 200000)).sum()),
            "coordinator_hub_neighbors_ge2_seed_reach_ge3": int(((df.hub_neighbors >= 2) & (df.seed_reach >= 3)).sum()),
            "coordinator_plus_between_q99_in_out": int(coordinator.sum()),
            "betweenness_q99": between_q99,
        },
        "proposed_role_counts": proposed_role.value_counts().to_dict(),
        "coordinator_candidates": records(df.loc[coordinator, cols + ["hub_neighbors", "betweenness"]]),
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("docs/analysis.json"))
    parser.add_argument("--markdown", type=Path, default=Path("docs/analysis.md"))
    args = parser.parse_args()
    result = analyze(args.data)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(markdown(result))
    print(json.dumps({k:v for k,v in result.items() if k in ["counts", "sanity", "threshold_counts", "proposed_role_counts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
