import os
import numpy as np
import pandas as pd

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

@dataclass
class PreprocessConfig:
    dataset_dir: str = "dataset"
    outputs_dir: str = "outputs"
    drop_missing_labels: bool = True

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def build_subgraph_feature_table(cfg: PreprocessConfig) -> Tuple[pd.DataFrame, pd.Series]:
    project_root = Path(__file__).resolve().parents[1]
    dataset_dir = project_root / cfg.dataset_dir
    outputs_dir = project_root / cfg.outputs_dir
    _ensure_dir(outputs_dir)

    nodes_path = dataset_dir / "nodes.csv"
    edges_path = dataset_dir / "edges.csv"
    cc_path = dataset_dir / "connected_components.csv"

    nodes = pd.read_csv(nodes_path)
    edges = pd.read_csv(edges_path)
    # handle duplicate ccId header if present
    cc = pd.read_csv(cc_path)

    # manually fix duplicate column names (e.g. ccId appearing twice)
    if cc.columns.duplicated().any():
        new_cols = []
        counts = {}
        for c in cc.columns:
            if c in counts:
                counts[c] += 1
                new_cols.append(f"{c}_{counts[c]}")
            else:
                counts[c] = 0
                new_cols.append(c)
        cc.columns = new_cols

    # validate expected columns
    for col in ["clId", "ccId"]:
        if col not in nodes.columns:
            raise ValueError(f"nodes.csv must contain '{col}'. Found: {list(nodes.columns)}")
    for col in ["clId1", "clId2", "txId"]:
        if col not in edges.columns:
            raise ValueError(f"edges.csv must contain '{col}'. Found: {list(edges.columns)}")

    # connected_components should have ccId and ccLabel
    if "ccId" not in cc.columns:
        # sometimes duplicated becomes ccId.1 etc. we pick first column that starts with ccId
        cand = [c for c in cc.columns if c.lower().startswith("ccid")]
        if not cand:
            raise ValueError(f"connected_components.csv must contain ccId. Found: {list(cc.columns)}")
        cc = cc.rename(columns={cand[0]: "ccId"})
    if "ccLabel" not in cc.columns:
        raise ValueError(f"connected_components.csv must contain ccLabel. Found: {list(cc.columns)}")

    # convert id's to string for safe merges
    nodes = nodes.copy()
    nodes["clId"] = nodes["clId"].astype(str)
    nodes["ccId"] = nodes["ccId"].astype(str)

    edges = edges.copy()
    edges["clId1"] = edges["clId1"].astype(str)
    edges["clId2"] = edges["clId2"].astype(str)
    edges["txId"] = edges["txId"].astype(str)

    cc = cc[["ccId", "ccLabel"]].drop_duplicates(subset=["ccId"]).copy()
    cc["ccId"] = cc["ccId"].astype(str)

    # node counts per component
    node_counts = nodes.groupby("ccId")["clId"].nunique().rename("num_nodes").reset_index()

    # map each clId -> ccId
    cl_to_cc = nodes[["clId", "ccId"]].drop_duplicates()

    # attach ccId to each edge endpoint
    e = edges.merge(cl_to_cc, left_on="clId1", right_on="clId", how="inner").rename(columns={"ccId": "cc1"})
    e = e.drop(columns=["clId"])
    e = e.merge(cl_to_cc, left_on="clId2", right_on="clId", how="inner").rename(columns={"ccId": "cc2"})
    e = e.drop(columns=["clId"])

    # internal edges: both endpoints in same component
    e_in = e[e["cc1"] == e["cc2"]].copy()
    e_in = e_in.rename(columns={"cc1": "ccId"})
    e_in = e_in.drop(columns=["cc2"])

    # edge count per component
    edge_counts = e_in.groupby("ccId").size().rename("num_edges").reset_index()

    # unique tx count per component
    tx_counts = e_in.groupby("ccId")["txId"].nunique().rename("unique_txs").reset_index()

    # degree features within component
    deg = pd.concat(
        [
            e_in[["ccId", "clId1"]].rename(columns={"clId1": "clId"}),
            e_in[["ccId", "clId2"]].rename(columns={"clId2": "clId"}),
        ],
        ignore_index=True,
    )
    deg_counts = deg.groupby(["ccId", "clId"]).size().rename("degree").reset_index()
    deg_stats = deg_counts.groupby("ccId")["degree"].agg(["mean", "max", "std"]).reset_index()
    deg_stats = deg_stats.rename(columns={"mean": "degree_mean", "max": "degree_max", "std": "degree_std"})
    deg_stats["degree_std"] = deg_stats["degree_std"].fillna(0)

    # combine features
    X = node_counts.merge(edge_counts, on="ccId", how="left").merge(tx_counts, on="ccId", how="left").merge(deg_stats, on="ccId", how="left")
    X["num_edges"] = X["num_edges"].fillna(0)
    X["unique_txs"] = X["unique_txs"].fillna(0)
    X["degree_mean"] = X["degree_mean"].fillna(0)
    X["degree_max"] = X["degree_max"].fillna(0)
    X["degree_std"] = X["degree_std"].fillna(0)

    # ratio features (help ML a lot)
    X["edges_per_node"] = X["num_edges"] / X["num_nodes"].replace(0, np.nan)
    X["txs_per_edge"] = X["unique_txs"] / X["num_edges"].replace(0, np.nan)
    X["txs_per_node"] = X["unique_txs"] / X["num_nodes"].replace(0, np.nan)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # labels
    data = X.merge(cc, on="ccId", how="left")
    y = data["ccLabel"]
    X_out = data.drop(columns=["ccLabel"]).rename(columns={"ccId": "component_id"})

    if cfg.drop_missing_labels:
        mask = ~y.isna()
        X_out = X_out.loc[mask].reset_index(drop=True)
        y = y.loc[mask].reset_index(drop=True)

    # save engineered dataset
    save_df = X_out.copy()
    save_df["label"] = y
    save_path = outputs_dir / "subgraph_features.csv"
    save_df.to_csv(save_path, index=False)

    return X_out, y
