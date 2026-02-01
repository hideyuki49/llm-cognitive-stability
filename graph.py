#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SCRC figure generator (PDF)
- Parse SCRC summary logs (*.txt)
- Compute per-model metric means (across tasks + deltas, NaNs ignored)
- Output:
  1) fig_scale_curve.pdf  : SCRC composite score vs model params
  2) fig_radar_profiles.pdf: Radar chart of metric profiles for selected models
  3) scrc_aggregates.csv   : Aggregated table you can paste into LaTeX

Usage:
  python make_figs.py --input_dir . --output_dir .

Notes:
- Expects lines like:
    === t1_constraints_format: ... ===
    δ=0 CIR=1.000 CASm=0.889 RRS=nan FUV=nan AWR=1.000 CQS=1.000 label=UNK
- Robust to extra noise lines (RAW_OUT etc.)
"""

from __future__ import annotations
import argparse
import math
import os
import re
import glob
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


METRICS = ["CIR", "CASm", "FUV", "AWR", "CQS", "RRS"]

# ---- Regex patterns ----
RE_TASK = re.compile(r"^===\s+(t[0-9a-zA-Z_]+)\s*:")
RE_DELTA = re.compile(r"^δ\s*=\s*([0-9]+)\s+(.*)$")
RE_KV = re.compile(r"(CIR|CASm|RRS|FUV|AWR|CQS)=([^\s]+)")


def parse_float(x: str) -> float:
    x = x.strip()
    if x.lower() == "nan":
        return float("nan")
    try:
        return float(x)
    except ValueError:
        return float("nan")


@dataclass
class ModelInfo:
    name: str
    params_b: Optional[float]  # billions, e.g., 7.0 for 7B
    path: str


def infer_model_name_from_filename(path: str) -> str:
    base = os.path.basename(path)
    # strip extension
    base = re.sub(r"\.txt$", "", base)
    # common suffixes
    base = base.replace("_summary", "")
    return base


def infer_params_b(model_name: str) -> Optional[float]:
    """
    Heuristic parsing:
      - "...-0.5B..." -> 0.5
      - "...-7B..."   -> 7
      - "Mixtral-8x7B" -> treat as 56B total? we return 56 for x-axis comparability
      - "Qwen2-57B-A14B" -> treat as 57
    """
    # Mixtral-8x7B
    m = re.search(r"(\d+)\s*x\s*(\d+(?:\.\d+)?)B", model_name, flags=re.IGNORECASE)
    if m:
        a = float(m.group(1))
        b = float(m.group(2))
        return a * b  # total experts*size as a rough scale axis

    # Qwen2-57B-A14B style
    m = re.search(r"(\d+(?:\.\d+)?)B", model_name)
    if m:
        return float(m.group(1))

    return None


def parse_summary_file(path: str) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      model, task, delta, CIR, CASm, FUV, AWR, CQS, RRS
    One row per (task, delta).
    """
    model = infer_model_name_from_filename(path)
    rows = []
    current_task = None

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")

            mt = RE_TASK.match(line)
            if mt:
                current_task = mt.group(1)
                continue

            md = RE_DELTA.match(line)
            if md and current_task is not None:
                delta = int(md.group(1))
                rest = md.group(2)
                kv = {k: float("nan") for k in METRICS}
                for k, v in RE_KV.findall(rest):
                    kv[k] = parse_float(v)

                rows.append({
                    "model": model,
                    "task": current_task,
                    "delta": delta,
                    **kv
                })

    if not rows:
        return pd.DataFrame(columns=["model", "task", "delta"] + METRICS)

    return pd.DataFrame(rows)


def nanmean_safe(arr: np.ndarray) -> float:
    if arr.size == 0:
        return float("nan")
    if np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def build_aggregates(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      - model_metrics: per-model mean of each metric across tasks+deltas
      - task_metrics : per-(model, task) mean of each metric across deltas
    """
    model_metrics = (
        df.groupby("model")[METRICS]
          .agg(lambda s: np.nanmean(s.values))
          .reset_index()
    )

    task_metrics = (
        df.groupby(["model", "task"])[METRICS]
          .agg(lambda s: np.nanmean(s.values))
          .reset_index()
    )

    return model_metrics, task_metrics


def compute_composite_score(model_metrics: pd.DataFrame,
                            weights: Optional[Dict[str, float]] = None) -> pd.Series:
    """
    Composite score for scale curve.
    Default: unweighted mean of available metrics (NaN ignored).
    """
    if weights is None:
        weights = {m: 1.0 for m in METRICS}

    w = np.array([weights[m] for m in METRICS], dtype=float)
    w = w / w.sum()

    scores = []
    for _, row in model_metrics.iterrows():
        vals = np.array([row[m] for m in METRICS], dtype=float)
        mask = ~np.isnan(vals)
        if not np.any(mask):
            scores.append(float("nan"))
            continue
        # re-normalize weights over available metrics
        ww = w[mask]
        ww = ww / ww.sum()
        scores.append(float(np.sum(vals[mask] * ww)))
    return pd.Series(scores, index=model_metrics.index, name="SCRC")


def make_scale_curve(model_metrics: pd.DataFrame, outpath: str) -> None:
    # infer params
    params = []
    for name in model_metrics["model"].tolist():
        params.append(infer_params_b(name))
    model_metrics = model_metrics.copy()
    model_metrics["params_b"] = params

    # drop unknown params for curve
    plot_df = model_metrics.dropna(subset=["params_b"]).copy()
    plot_df["SCRC"] = compute_composite_score(plot_df)

    # sort by params
    plot_df = plot_df.sort_values("params_b")

    plt.figure(figsize=(7.2, 4.2))
    plt.plot(plot_df["params_b"], plot_df["SCRC"], marker="o")
    for _, r in plot_df.iterrows():
        # short label on points
        label = r["model"]
        label = label.replace("-Instruct", "")
        plt.annotate(label, (r["params_b"], r["SCRC"]), textcoords="offset points", xytext=(-20, 4), fontsize=6)

    plt.xscale("log")
    plt.xlabel("Model scale (B parameters, log scale; MoE approximated)")
    plt.ylabel("SCRC composite score (mean over metrics)")
    plt.title("SCRC vs Model Scale")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def make_radar(model_metrics: pd.DataFrame, outpath: str, models: List[str]) -> None:
    # filter chosen models
    df = model_metrics.set_index("model")
    chosen = []
    for m in models:
        if m in df.index:
            chosen.append(m)
        else:
            # try fuzzy contains
            hits = [idx for idx in df.index if m.lower() in idx.lower()]
            if hits:
                chosen.append(hits[0])

    if not chosen:
        raise ValueError("No matching models for radar. Check --radar_models.")

    # radar setup
    labels = METRICS
    n = len(labels)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False).tolist()
    angles += angles[:1]  # close loop

    fig = plt.figure(figsize=(6.8, 6.2))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0.0, 1.05)

    for m in chosen:
        vals = df.loc[m, METRICS].values.astype(float).tolist()
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=2, label=m.replace("_summary", ""))
        ax.fill(angles, vals, alpha=0.10)

    ax.set_title("SCRC Metric Profiles (Radar)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=1, fontsize=8)
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, default=".", help="Directory containing *_summary.txt and related logs.")
    ap.add_argument("--output_dir", type=str, default=".", help="Directory to write PDFs/CSVs.")
    ap.add_argument("--pattern", type=str, default="*.txt", help="Glob pattern for input files (default: *.txt).")
    ap.add_argument(
        "--radar_models",
        type=str,
        default="Qwen2.5-0.5B,Qwen2.5-7B,Qwen2.5-32B,Mixtral-8x7B",
        help="Comma-separated model name substrings for radar chart."
    )
    args = ap.parse_args()

    in_glob = os.path.join(args.input_dir, args.pattern)
    paths = sorted(glob.glob(in_glob))
    if not paths:
        raise FileNotFoundError(f"No files matched: {in_glob}")

    # parse all files
    dfs = []
    for p in paths:
        df = parse_summary_file(p)
        if len(df) > 0:
            dfs.append(df)

    if not dfs:
        raise RuntimeError("No parseable SCRC rows found. Check file format and pattern.")

    df_all = pd.concat(dfs, ignore_index=True)

    model_metrics, task_metrics = build_aggregates(df_all)

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # write aggregates for LaTeX tables
    agg_csv = os.path.join(out_dir, "scrc_aggregates.csv")
    model_metrics.to_csv(agg_csv, index=False)

    # make figures
    scale_pdf = os.path.join(out_dir, "fig_scale_curve.pdf")
    radar_pdf = os.path.join(out_dir, "fig_radar_profiles.pdf")

    make_scale_curve(model_metrics, scale_pdf)

    radar_models = [s.strip() for s in args.radar_models.split(",") if s.strip()]
    make_radar(model_metrics, radar_pdf, radar_models)

    print("Wrote:")
    print("  ", agg_csv)
    print("  ", scale_pdf)
    print("  ", radar_pdf)


if __name__ == "__main__":
    main()
