"""Every explanation is computed from observed metrics; no generated client facts."""
import pandas as pd

ROLE_NAMES = {"coordinator": "структурная связность", "consolidator": "консолидация",
              "distributor": "распределение", "transit": "транзит", "terminal": "получатель", "peripheral": "периферия"}


def explanation(r):
    if r.role == "coordinator":
        text = f"Связывает {r.structural_neighbors} узлов сбора/рассылки; структурно достижим от {r.seed_reach} seed (без учёта дат); betweenness={r.betweenness:.5f}. Гипотеза координации."
    elif r.role == "consolidator":
        text = f"Вход от {r.in_deg} плательщиков (seed: {r.seed_payers}); {r.in_kzt:,.0f} KZT; исходящих связей {r.out_deg}. Признаки консолидации."
    elif r.role == "distributor":
        text = f"Переводы {r.out_deg} получателям; {r.out_tx} операций; {r.out_kzt:,.0f} KZT. Признаки веерного распределения."
    elif r.role == "transit":
        text = f"Вход {r.in_kzt:,.0f} KZT; выход/вход={r.pass_through:.2f}; сопоставлено за 1–2 дня {r.fast_out_share:.0%}. Признаки транзита."
    elif r.role == "terminal":
        text = f"Колено {r.depth}; вход {r.in_kzt:,.0f} KZT; исходящих связей {r.out_deg}, выход/вход={r.pass_through:.2f}. Получатель в видимой выборке."
    else:
        text = f"Колено {r.depth}; входящих связей {r.in_deg}, исходящих {r.out_deg}; вход {r.in_kzt:,.0f} KZT. Пороги значимых ролей не достигнуты."
    if r.truncated_by_depth:
        text = f"4-е колено: исходящие не выгружены. Вход {r.in_deg} плательщиков, {r.in_kzt:,.0f} KZT. " + ("Признаки консолидации." if r.role == "consolidator" else "Роль за границей обхода не установлена.")
    elif r.is_seed and r.in_deg + r.out_deg == 0:
        text = "Seed; 0 связей и 0 KZT в выгрузке. Данных для поведенческой роли недостаточно."
    return text[:200]


def add_explanations(df):
    result = df.copy()
    result["evidence"] = [explanation(r) for r in result.itertuples(index=False)]
    result["next_request"] = [next_request(r) for r in result.itertuples(index=False)]
    return result


def next_request(r):
    if r.truncated_by_depth:
        return "Запросить исходящие следующего колена, наличные и межбанк."
    if r.is_seed or r.unobserved_funding:
        return "Запросить входящие вне выборки и остаток на начало июля; уточнить назначение платежей."
    if r.role == "terminal":
        return "Запросить наличные, межбанк и переводы ниже 5 000 KZT."
    if r.role == "transit":
        return "Запросить точное время и назначение переводов для проверки последовательности."
    return "Проверить назначение переводов и экономическое основание связей."


def priority_explanation(r):
    return (f"Слагаемые приоритета: роль {r.priority_role:.3f}, структурная достижимость {r.priority_reach:.3f}, "
            f"объём {r.priority_volume:.3f}, PageRank {r.priority_pagerank:.3f}, "
            f"паттерны {r.priority_patterns:.3f}; множитель seed {r.seed_discount:.1f}.")


def top_nodes(df, count=50):
    top = df.sort_values(["priority_score", "gid"], ascending=[False, True]).head(count)
    rows = []
    for rank, r in enumerate(top.itertuples(index=False), 1):
        patterns = getattr(r, "matched_roles", r.role)
        flow = (f"Вход: {r.in_deg} плательщиков, {r.in_kzt:,.0f} KZT; "
                f"выход: {r.out_deg} получателей, {r.out_kzt:,.0f} KZT.")
        temporal = (f"Достижимость от seed по датам: {r.temporal_seed_reach_strict_days} "
                    f"со строго растущими днями; до {r.temporal_seed_reach_upper} при допустимом порядке внутри дня. "
                    "Это совместимость маршрута с датами, не происхождение денег. "
                    if hasattr(r, "temporal_seed_reach_upper") else "")
        rows.append({"rank": rank, "gid": r.gid, "role": r.role, "priority_score": r.priority_score,
                     "why": f"{r.evidence} {flow} Сработавшие правила: {patterns}. "
                            f"{temporal}{priority_explanation(r)} {next_request(r)}"})
    return pd.DataFrame(rows)


def data_requests(df):
    rows = []
    top = df.sort_values(["priority_score", "gid"], ascending=[False, True])
    for r in top.itertuples(index=False):
        if r.truncated_by_depth:
            reason, request = "Обрыв обхода на глубине 4", "Исходящие переводы клиента и следующее колено за июль"
        elif r.is_seed or r.unobserved_funding:
            reason, request = "Неполный наблюдаемый баланс", "Входящие вне выборки и остаток на начало июля"
        elif r.role == "terminal":
            reason, request = "Не видны внешние выходы", "Наличные, межбанк, переводы ниже 5000 KZT"
        else:
            continue
        rows.append({"gid": r.gid, "priority_score": r.priority_score, "reason": reason, "request": request})
    return pd.DataFrame(rows, columns=["gid", "priority_score", "reason", "request"])
