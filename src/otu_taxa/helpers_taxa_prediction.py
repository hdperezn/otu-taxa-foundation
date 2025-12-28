# --- imports ---
import os, re, json, glob, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- helpers ---
def norm_id(s: str) -> str:
    return re.sub(r"\s+", "", str(s)).strip()

def is_unknown(name: str) -> bool:
    s = str(name).strip(); sl = s.lower()
    return (s == "") or ("__unknown" in sl) or (sl == "unknown") or s.startswith("Unknown_")

def load_removed_species_list(path: str) -> set:
    ext = os.path.splitext(path)[1].lower()
    with open(path) as f:
        items = json.load(f) if ext == ".json" else [line.strip() for line in f if line.strip()]
    cleaned = []
    for s in items:
        s = s.split("(", 1)[0].strip()
        if s: cleaned.append(s)
    return set(cleaned)

def load_sintax_table(path: str) -> pd.DataFrame:
    # supports either: col 0=otu_id, col 3=taxonomy   OR   col 0=otu_id, col 1=raw string with confidences
    try:
        df = pd.read_csv(path, sep="\t", header=None, engine="python",
                         usecols=[0,3], names=["otu_id","taxonomy"], dtype=str)
    except Exception:
        df_raw = pd.read_csv(path, sep="\t", header=None, engine="python",
                             usecols=[0,1], names=["otu_id","raw_sintax"], dtype=str)
        def drop_conf(s):
            if pd.isna(s): return ""
            parts = [p.split("(",1)[0].strip() for p in s.strip().rstrip(";").split(",") if ":" in p]
            return ",".join(parts)
        df_raw["taxonomy"] = df_raw["raw_sintax"].apply(drop_conf)
        df = df_raw[["otu_id","taxonomy"]]
    df["taxonomy"] = df["taxonomy"].fillna("")
    return df

def parse_tax_to_cols(series: pd.Series) -> pd.DataFrame:
    ranks = ["k","p","c","o","f","g","s"]
    def to_dict(tax):
        out = {r:"" for r in ranks}
        for part in str(tax).strip().rstrip(";").split(","):
            if ":" not in part: continue
            r, name = part.split(":",1)
            r = r.strip(); name = name.split("(",1)[0].strip()
            if r in out: out[r] = name
        return pd.Series(out)
    return series.apply(to_dict)

# def load_otu_vocab_map(ds_dir: str):
#     # finds first otu*_vocab.json
#     cands = sorted(glob.glob(os.path.join(ds_dir, "**", "otu*_vocab.json"), recursive=True))
#     if not cands:
#         raise FileNotFoundError(f"No otu*_vocab.json found under {ds_dir}")
#     path = cands[0]
#     with open(path) as f: obj = json.load(f)
#     if isinstance(obj, list):
#         name2id = {str(name): i for i, name in enumerate(obj)}
#     elif isinstance(obj, dict):
#         name2id = {str(k): int(v) for k, v in obj.items()}
#     else:
#         raise ValueError("Unsupported OTU vocab format (expected list or dict).")
#     id2name = {v:k for k,v in name2id.items()}
#     return name2id, id2name






def load_vocab_json(DATASET_DIR: str ,path_pattern: str):
    cands = sorted(glob.glob(os.path.join(DATASET_DIR, "**", path_pattern), recursive=True))
    if not cands:
        raise FileNotFoundError(f"No file matching '{path_pattern}' under {DATASET_DIR}")
    path = cands[0]
    with open(path) as f:
        obj = json.load(f)
    if isinstance(obj, list):
        name2id = {str(name): i for i, name in enumerate(obj)}
    elif isinstance(obj, dict):
        # favor name->id dicts; coerce to int indices
        name2id = {str(k): int(v) for k, v in obj.items()}
    else:
        raise TypeError(f"Unsupported JSON format in {path}")
    id2name = {v:k for k,v in name2id.items()}
    return name2id, id2name, path


# ploting accuracy: Number of predictions per OTU vs accuracy

import json, pandas as pd, numpy as np, matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, ScalarFormatter
from pathlib import Path
from typing import Tuple, Optional, Union

def evaluate_and_plot_predictions(
    jsonl_path: Union[str, Path],
    test_list_path: Union[str, Path],
    title_prefix: str = "Per-OTU Accuracy vs # Predictions (TEST)",
    make_plot: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict, Optional[plt.Figure]]:
    """
    Load predictions (JSONL), filter to TEST sample_ids (txt list),
    extract top-1 predictions and correctness, aggregate per OTU,
    compute summary metrics, and optionally plot accuracy vs #predictions (log X).

    Returns: (df_filtered, per_otu_df, summary_dict, fig_or_None)
    """
    # --- load predictions ---
    df = pd.read_json(str(jsonl_path), lines=True)

    # --- load fixed test ids and filter ---
    with open(str(test_list_path), "r", encoding="utf-8") as f:
        test_ids = set(line.strip() for line in f if line.strip())
    df = df[df["sample_id"].isin(test_ids)].copy()

    # --- extract top-1 + correctness ---
    df = df[df["pred_topk"].apply(lambda x: isinstance(x, list) and len(x) > 0)].copy()

    def _top1_id(lst):
        try:
            return int(lst[0].get("tax_id"))
        except Exception:
            return None

    df["pred_top1_id"] = df["pred_topk"].apply(_top1_id)
    df = df.dropna(subset=["pred_top1_id"]).copy()
    df["pred_top1_id"] = df["pred_top1_id"].astype(int)
    df["correct"] = (df["pred_top1_id"] == df["true_tax_id"]).astype(int)

    # --- per-OTU aggregation (on TEST subset only) ---
    g = (
        df.groupby("otu_name")
          .agg(accuracy=("correct", "mean"),
               n_pred=("correct", "size"),
               n_unique_samples=("sample_id", "nunique"))
          .reset_index()
    )

    # --- summary + optional plot ---
    fig = None
    if g.empty:
        summary = {
            "n_samples": 0, "n_preds": 0, "n_otus": 0,
            "n_correct": 0, "micro_acc": float("nan"),
            "mean_per_otu": float("nan")
        }
        print("[WARN] No predictions after filtering to the test set.")
        return df, g, summary, fig

    total_preds = len(df)
    n_correct   = int(df["correct"].sum())
    micro_acc   = (n_correct / total_preds) if total_preds > 0 else float("nan")
    mean_per_otu = float(g["accuracy"].mean())
    n_samples = int(df["sample_id"].nunique())
    n_otus    = int(g.shape[0])

    summary = {
        "n_samples": n_samples, "n_preds": total_preds, "n_otus": n_otus,
        "n_correct": n_correct, "micro_acc": micro_acc, "mean_per_otu": mean_per_otu
    }

    if make_plot:
        g_sorted = g.sort_values(
            ["n_unique_samples", "n_pred", "accuracy"],
            ascending=[False, False, False]
        )
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.scatter(g_sorted["n_pred"], g_sorted["accuracy"], alpha=0.9)
        ax.set_xlabel("# predictions per OTU (n_pred) [log]")
        ax.set_ylabel("Accuracy (Top-1 taxonomy)")
        ax.set_title(f"{title_prefix}: {n_otus} OTUs, "
                     f"correct={n_correct}/{total_preds} ({micro_acc:.2%})")
        ax.set_ylim(-0.1, 1.1)
        ax.set_xscale("log")
        xmin = max(1, int(g_sorted["n_pred"].min()))
        xmax = int(g_sorted["n_pred"].max() * 1.1)
        ax.set_xlim(xmin, xmax)
        ax.xaxis.set_major_locator(LogLocator(base=10.0))
        ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=tuple(range(2, 10))))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.tick_params(axis='x', which='major', length=6)
        ax.tick_params(axis='x', which='minor', length=3)
        ax.grid(True, which="both", axis="x", linewidth=0.5, alpha=0.3)
        ax.grid(True, which="major", axis="y", linewidth=0.5, alpha=0.3)
        ax.axhline(mean_per_otu, linestyle="--", color="tab:gray")
        ax.text(xmax, mean_per_otu, " mean (per-OTU, TEST) = {:.2f}".format(mean_per_otu),
                va="bottom", ha="right")
        plt.tight_layout()
        plt.show()

    print("[INFO] TEST subset: {} unique samples, {} predictions, {} OTUs | "
          "correct={}/{} ({:.2%})".format(
              n_samples, total_preds, n_otus, n_correct, total_preds, micro_acc
          ))

    return df, g, summary, fig