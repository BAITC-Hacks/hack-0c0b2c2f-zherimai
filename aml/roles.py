"""Transparent structural roles. Scores measure rule strength, not guilt."""
import numpy as np

ROLES = ("coordinator", "consolidator", "distributor", "transit", "terminal", "peripheral")
THRESHOLDS = {
    "consolidator_in_deg": 5, "distributor_out_deg": 10,
    "transit_ratio_min": .8, "transit_ratio_max": 1.2, "transit_in_kzt": 50000.,
    "terminal_in_kzt": 50000., "retention_in_kzt": 200000., "retention_ratio_max": .2,
    "coordinator_structural_neighbors": 2, "coordinator_seed_reach": 3,
    "coordinator_betweenness_quantile": .99,
}


def assign_roles(df, graph):
    result = df.copy()
    t = THRESHOLDS
    consolidate = result.in_deg >= t["consolidator_in_deg"]
    distribute = result.out_deg >= t["distributor_out_deg"]
    transit = (~result.is_seed & (result.in_deg > 0) & (result.out_deg > 0)
               & result.pass_through.between(t["transit_ratio_min"], t["transit_ratio_max"])
               & (result.in_kzt >= t["transit_in_kzt"]))
    terminal = ((result.depth < 4) & (result.in_kzt >= t["terminal_in_kzt"])
                & ((result.out_deg == 0) | (~result.is_seed & (result.pass_through <= t["retention_ratio_max"])
                                          & (result.in_kzt >= t["retention_in_kzt"]))))
    result["base_role"] = np.select([consolidate, distribute, transit, terminal],
                                    ["consolidator", "distributor", "transit", "terminal"], default="peripheral")
    structural = set(result.loc[consolidate | distribute, "gid"])
    result["structural_neighbors"] = [len((set(graph.predecessors(int(g))) | set(graph.successors(int(g)))) & structural) for g in result.gid]
    bw_threshold = float(result.betweenness.quantile(t["coordinator_betweenness_quantile"]))
    coordinator = ((result.structural_neighbors >= t["coordinator_structural_neighbors"])
                   & (result.seed_reach >= t["coordinator_seed_reach"])
                   & (result.in_deg > 0) & (result.out_deg > 0)
                   & (result.betweenness >= bw_threshold) & (result.betweenness > 0))
    result["role"] = result.base_role.where(~coordinator, "coordinator")
    scores = []
    for r in result.itertuples(index=False):
        if r.role == "coordinator":
            score = .5 + .5 * min(1., max(0., r.betweenness / max(bw_threshold, 1e-15) - 1.))
        elif r.role == "consolidator":
            score = .5 + .5 * min(1., (r.in_deg - 5) / 10)
        elif r.role == "distributor":
            score = .5 + .5 * min(1., (r.out_deg - 10) / 30)
        elif r.role == "transit":
            score = .5 + .25 * max(0., 1 - abs(r.pass_through - 1) / .2) + .25 * r.fast_out_share
        elif r.role == "terminal":
            score = .5 + .5 * min(1., max(0., r.in_kzt / 50000 - 1) / 9)
        else:
            # Strength of a residual assignment is deliberately low with missing observations.
            score = .25 if r.truncated_by_depth or r.in_deg + r.out_deg == 0 else .5
        if r.truncated_by_depth and r.role == "consolidator":
            score *= .8
        scores.append(round(min(1., max(0., score)), 6))
    result["role_score"] = scores
    result["flags"] = [";".join(flag for flag, active in [
        ("seed", r.is_seed), ("truncated_depth4", r.truncated_by_depth),
        ("unobserved_funding", r.unobserved_funding),
        ("fast_temporal_match", r.fast_out_share >= .5),
        ("sync_inflow", r.max_payers_same_day >= 3), ("directed_cycle", r.in_cycle),
        ("no_observed_edges", r.in_deg + r.out_deg == 0),
    ] if active) or "none" for r in result.itertuples(index=False)]
    return result, {**t, "coordinator_betweenness_actual": bw_threshold}
