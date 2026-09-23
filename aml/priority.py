"""Rank review candidates using disclosed, deterministic weights."""
import numpy as np

ROLE_WEIGHTS = {"coordinator": 1., "consolidator": .9, "distributor": .8,
                "transit": .6, "terminal": .5, "peripheral": .1}


def rank_nodes(df):
    result = df.copy()
    reach = np.log1p(result.seed_reach)
    volume = np.log1p(result.in_kzt + result.out_kzt)
    result["priority_role"] = .30 * result.role.map(ROLE_WEIGHTS)
    result["priority_reach"] = .25 * reach / max(float(reach.max()), 1.)
    result["priority_volume"] = .20 * volume / max(float(volume.max()), 1.)
    result["priority_pagerank"] = .15 * result.pagerank.rank(method="average", pct=True)
    signals = (result.fast_out_share.ge(.5).astype(float) + result.max_payers_same_day.ge(3).astype(float)
               + result.in_cycle.astype(float)) / 3
    result["priority_patterns"] = .10 * signals
    total = result[["priority_role", "priority_reach", "priority_volume", "priority_pagerank", "priority_patterns"]].sum(axis=1)
    result["seed_discount"] = np.where(result.is_seed, .6, 1.)
    result["priority_score"] = (total * result.seed_discount).clip(0, 1).round(8)
    return result
