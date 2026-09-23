"""Build a self-contained, offline canvas dashboard from pipeline tables."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .roles import THRESHOLDS


def _records(frame: pd.DataFrame, identifiers: tuple[str, ...]) -> list[dict]:
    """Keep opaque identifiers lossless; let pandas normalize numpy and NA values."""
    copy = frame.copy()
    for column in identifiers:
        if column in copy:
            copy[column] = copy[column].map(lambda value: "" if pd.isna(value) else str(value))
    return json.loads(copy.to_json(orient="records", date_format="iso"))


def build_viewer(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    clusters_df: pd.DataFrame,
    out_path: Path,
) -> Path:
    """Write one HTML file: no CDN, server, keys, or browser network access needed.

    Coordinates are generated deterministically in-browser in linear time. The
    overview groups nodes by cluster, while an ego view uses 1–2 hop rings.
    """
    out_path = Path(out_path)
    payload = {
        "nodes": _records(nodes_df, ("gid", "cluster_id")),
        "edges": _records(edges_df, ("src", "dst")),
        "clusters": _records(clusters_df, ("cluster_id",)),
        "thresholds": {
            **THRESHOLDS,
            "coordinator_betweenness_actual": float(
                nodes_df.betweenness.quantile(THRESHOLDS["coordinator_betweenness_quantile"])
            ) if "betweenness" in nodes_df and len(nodes_df) else 0.0,
        },
    }
    # Do not let any dataset text close the embedded JSON script element.
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    encoded = encoded.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    template = Path(__file__).with_name("viewer_template.html").read_text(encoding="utf-8")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(template.replace("__AML_DATA__", encoded), encoding="utf-8")
    return out_path
