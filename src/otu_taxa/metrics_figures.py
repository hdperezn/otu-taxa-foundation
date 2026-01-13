####################################
######## Accuracy per OTU Plot #####
####################################


from typing import Union, Tuple, Optional, Set
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, ScalarFormatter

def evaluate_and_plot_predictions_multilabel(
    jsonl_path: Union[str, Path],
    title_prefix: str = "Per-OTU Accuracy vs # Predictions (multilabel TEST)",
    make_plot: bool = True,
    # Optional: if you want to enforce filtering (e.g., sanity check)
    test_ids: Optional[Set[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict, Optional[plt.Figure]]:
    """
    Multilabel-aware evaluation:

    For each prediction row:
      - Infer the TRUE rank letter from `true_tax_name` (k,p,c,o,f,g,s).
      - Take the model's prediction at that SAME rank from `pred_by_rank[rank]`.
      - Correct if predicted tax_id == true_tax_id.

    Aggregate per OTU and plot Accuracy vs # predictions per OTU (log X).

    Expected JSONL columns:
      - sample_id, otu_name
      - true_tax_id, true_tax_name
      - pred_by_rank: dict like {"g": {"tax_id": ..., "tax_name": ...}, ...}
    """
    jsonl_path = Path(jsonl_path)

    # --- load predictions ---
    df = pd.read_json(str(jsonl_path), lines=True)

    # --- optional filter by provided test IDs ---
    if test_ids is not None:
        test_ids = set(map(str, test_ids))
        df["sample_id"] = df["sample_id"].astype(str)
        df = df[df["sample_id"].isin(test_ids)].copy()

    if df.empty:
        summary = {
            "n_samples": 0, "n_preds": 0, "n_otus": 0,
            "n_correct": 0, "micro_acc": float("nan"),
            "mean_per_otu": float("nan")
        }
        print("[WARN] No predictions to evaluate (empty after loading/filtering).")
        return df, pd.DataFrame(), summary, None

    # --- infer TRUE rank letter from true_tax_name ---
    def _true_rank_letter(name):
        if not isinstance(name, str) or len(name) == 0:
            return None
        ch = name[0].lower()
        return ch if ch in {"k","p","c","o","f","g","s"} else None

    df["true_rank"] = df["true_tax_name"].apply(_true_rank_letter)
    df = df.dropna(subset=["true_rank"]).copy()

    if df.empty:
        summary = {
            "n_samples": 0, "n_preds": 0, "n_otus": 0,
            "n_correct": 0, "micro_acc": float("nan"),
            "mean_per_otu": float("nan")
        }
        print("[WARN] No rows with a valid rank letter in true_tax_name.")
        return df, pd.DataFrame(), summary, None

    # --- extract predicted id at that same rank from pred_by_rank ---
    def _pred_id_for_true_rank(row):
        true_rank = row["true_rank"]
        pb = row.get("pred_by_rank", None)
        if not isinstance(pb, dict):
            return None
        pr = pb.get(true_rank, None)
        if not isinstance(pr, dict):
            return None
        try:
            return int(pr.get("tax_id"))
        except Exception:
            return None

    df["pred_rank_id"] = df.apply(_pred_id_for_true_rank, axis=1)
    df = df.dropna(subset=["pred_rank_id"]).copy()

    if df.empty:
        summary = {
            "n_samples": 0, "n_preds": 0, "n_otus": 0,
            "n_correct": 0, "micro_acc": float("nan"),
            "mean_per_otu": float("nan")
        }
        print("[WARN] No rows with a usable pred_by_rank entry for the true rank.")
        return df, pd.DataFrame(), summary, None

    df["pred_rank_id"] = df["pred_rank_id"].astype(int)

    # --- correctness ---
    df["correct"] = (df["pred_rank_id"] == df["true_tax_id"]).astype(int)

    # --- per-OTU aggregation ---
    g = (
        df.groupby("otu_name")
          .agg(accuracy=("correct", "mean"),
               n_pred=("correct", "size"),
               n_unique_samples=("sample_id", "nunique"))
          .reset_index()
    )

    fig = None
    if g.empty:
        summary = {
            "n_samples": int(df["sample_id"].nunique()),
            "n_preds": len(df),
            "n_otus": 0,
            "n_correct": int(df["correct"].sum()),
            "micro_acc": float("nan"),
            "mean_per_otu": float("nan"),
        }
        print("[WARN] No per-OTU groups after aggregation.")
        return df, g, summary, fig

    # --- summary metrics ---
    total_preds = len(df)
    n_correct   = int(df["correct"].sum())
    micro_acc   = (n_correct / total_preds) if total_preds > 0 else float("nan")
    mean_per_otu = float(g["accuracy"].mean())
    n_samples = int(df["sample_id"].nunique())
    n_otus    = int(g.shape[0])

    summary = {
        "n_samples": n_samples,
        "n_preds": total_preds,
        "n_otus": n_otus,
        "n_correct": n_correct,
        "micro_acc": micro_acc,
        "mean_per_otu": mean_per_otu,
    }

    # --- plot ---
    if make_plot:
        g_sorted = g.sort_values(
            ["n_unique_samples", "n_pred", "accuracy"],
            ascending=[False, False, False]
        )
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.scatter(g_sorted["n_pred"], g_sorted["accuracy"], alpha=0.2)
        ax.set_xlabel("# predictions per OTU (n_pred) [log]")
        ax.set_ylabel("Accuracy (per true rank)")
        ax.set_title(
            f"{title_prefix}: {n_otus} OTUs, "
            f"correct={n_correct}/{total_preds} ({micro_acc:.2%})"
        )
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
        ax.text(
            xmax, mean_per_otu,
            f" mean (per-OTU) = {mean_per_otu:.2f}",
            va="bottom", ha="right"
        )
        plt.tight_layout()
        plt.show()

    print(
        "[INFO] multilabel eval: {} unique samples, {} predictions, {} OTUs | "
        "correct={}/{} ({:.2%})".format(
            n_samples, total_preds, n_otus, n_correct, total_preds, micro_acc
        )
    )

    return df, g, summary, fig

####################################
######## Accuracy per level    #####
####################################


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union, Tuple, Optional, Set, Dict, List

def plot_per_level_accuracy_hierarchical_from_arrays(
    jsonl_path: Union[str, Path],
    ancestor_at_rank: Dict[str, List[int]],   # rank -> dense list len T_base with ancestor_id or -1
    test_ids: Optional[Set[str]] = None,
    ranks=("k","p","c","o","f","g","s"),
    level_labels=None,
    otu_col: str = "otu_name",
    title: str = "Hierarchical model: per-level accuracy",
    make_plot: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict, Optional[plt.Figure]]:
    """
    Per-level accuracy for hierarchical predictions.

    For each row and each rank r:
      gold_r = ancestor_at_rank[r][true_tax_id]  (or -1 if missing)
      pred_r = pred_by_rank[r]["tax_id"]
      Compare pred_r vs gold_r only when gold_r != -1

    Returns:
      df (row-level), summary_df (rank-level), overall dict, fig
    """
    jsonl_path = Path(jsonl_path)
    df = pd.read_json(str(jsonl_path), lines=True)

    if level_labels is None:
        level_labels = {"k":"kingdom","p":"phylum","c":"class",
                        "o":"order","f":"family","g":"genus","s":"species"}

    # optional strict filtering
    if test_ids is not None and "sample_id" in df.columns:
        test_ids = set(map(str, test_ids))
        df["sample_id"] = df["sample_id"].astype(str)
        df = df[df["sample_id"].isin(test_ids)].copy()

    if df.empty:
        print("[WARN] No rows to evaluate after loading/filtering.")
        return df, pd.DataFrame(), {
            "n_samples": 0,
            "n_rows": 0,
            "micro_overall": float("nan"),
            "mean_macro_overall": float("nan"),
        }, None

    # ensure true_tax_id is int
    df["true_tax_id"] = df["true_tax_id"].astype(int)

    # extract pred_{r} columns from pred_by_rank
    for r in ranks:
        pred_col = f"pred_{r}"
        if pred_col not in df.columns:
            def _get_pred(d):
                if not isinstance(d, dict):
                    return None
                x = d.get(r, None)
                if not isinstance(x, dict):
                    return None
                try:
                    return int(x.get("tax_id"))
                except Exception:
                    return None
            df[pred_col] = df["pred_by_rank"].apply(_get_pred) if "pred_by_rank" in df.columns else None

    # gold_{r} from dense ancestor arrays
    for r in ranks:
        gold_col = f"gold_{r}"
        arr = ancestor_at_rank[r]
        df[gold_col] = df["true_tax_id"].apply(lambda t: arr[t] if (0 <= t < len(arr)) else -1)

    # compute micro & macro per rank
    def _metrics_at_rank(df_local: pd.DataFrame, r: str):
        pred_col = f"pred_{r}"
        gold_col = f"gold_{r}"

        # only rows where gold exists
        sub = df_local[df_local[gold_col] != -1].copy()
        if len(sub) == 0:
            return np.nan, np.nan, 0, 0

        pred_vals = sub[pred_col].fillna(-1).astype("int64").to_numpy()
        gold_vals = sub[gold_col].astype("int64").to_numpy()
        corr = (pred_vals == gold_vals)
        micro = float(corr.mean())

        # macro over OTUs
        def _per_otu_acc(s):
            idx = s.index
            s_pred = s.fillna(-1).astype("int64").to_numpy()
            s_gold = sub.loc[idx, gold_col].astype("int64").to_numpy()
            return float((s_pred == s_gold).mean())

        macro = float(sub.groupby(otu_col)[pred_col].apply(_per_otu_acc).mean())
        n_otus = int(sub[otu_col].nunique())
        n_rows = int(len(sub))
        return micro, macro, n_otus, n_rows

    micro_arr, macro_arr, otus_used, rows_used = [], [], [], []
    for r in ranks:
        m, M, n_otus, n_rows = _metrics_at_rank(df, r)
        micro_arr.append(m)
        macro_arr.append(M)
        otus_used.append(n_otus)
        rows_used.append(n_rows)

    overall = {
        "n_samples": int(df["sample_id"].nunique()) if "sample_id" in df.columns else None,
        "n_rows": int(len(df)),
        "micro_overall": float(np.nanmean(micro_arr)),
        "mean_macro_overall": float(np.nanmean(macro_arr)),
    }

    fig = None
    if make_plot:
        x = [level_labels[r] for r in ranks]
        fig, ax1 = plt.subplots(figsize=(9.5, 5))

        ax1.plot(x, micro_arr, marker="o", linewidth=2, label="Micro accuracy")
        ax1.plot(x, macro_arr, marker=".", linestyle="--", linewidth=2, label="Macro accuracy")

        for xi, yi in zip(x, macro_arr):
            if np.isfinite(yi):
                ax1.text(xi, yi + 0.03, f"{yi:.3f}", ha="center", va="bottom", fontsize=9)

        ax1.set_ylim(0.0, 1.02)
        ax1.set_ylabel("Accuracy")
        ax1.set_xlabel("Taxonomic level")
        ax1.set_title(title)
        ax1.grid(True, axis="y", alpha=0.3)

        ax2 = ax1.twinx()
        ax2.bar(x, otus_used, alpha=0.25, label="# OTUs")
        ax2.set_ylabel("Number of OTUs")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

        plt.tight_layout()
        plt.show()

    summary_df = pd.DataFrame({
        "level": [level_labels[r] for r in ranks],
        "n_rows_used": rows_used,
        "n_otus_used": otus_used,
        "micro_acc": [round(v,4) if np.isfinite(v) else None for v in micro_arr],
        "macro_acc": [round(v,4) if np.isfinite(v) else None for v in macro_arr],
    })

    print(
        "[INFO] overall micro (avg over levels) = {:.3f}, mean macro (avg over levels) = {:.3f}".format(
            overall["micro_overall"], overall["mean_macro_overall"]
        )
    )

    return df, summary_df, overall, fig


################################################
######## Error Origin Matrix and RankAcc #######
################################################


from matplotlib.gridspec import GridSpec


RANKS = ("k","p","c","o","f","g","s")


# # ------------------------------------------------------------
# # 0) Build gold ancestor arrays 
# # ------------------------------------------------------------
# def build_ancestor_arrays_all_ranks(M_np, tax_vocab_unk, ranks=RANKS, missing_value=-1):
#     """
#     Returns dict rank -> dense list length T_base:
#       ancestor_at_rank[r][t] = ancestor tax_id at rank r, else -1.
#     """
#     ancestor_at_rank = {}
#     for r in ranks:
#         ancestor_at_rank[r] = build_tax2ancestor_at_rank(
#             M_np=M_np,
#             tax_vocab_list=tax_vocab_unk,
#             target_rank=r,
#             missing_value=missing_value,
#         )
#     return ancestor_at_rank


# ------------------------------------------------------------
# 1) Load predictions + add pred_{r} + gold_{r}
# ------------------------------------------------------------
def load_predictions_with_pred_gold(
    jsonl_path,
    ancestor_at_rank,
    ranks=RANKS,
):
    df = pd.read_json(str(Path(jsonl_path)), lines=True)
    if df.empty:
        raise ValueError("Empty predictions file.")

    # ensure ints where needed
    df["true_tax_id"] = df["true_tax_id"].astype(int)

    # pred_{r} from pred_by_rank
    for r in ranks:
        pred_col = f"pred_{r}"
        if pred_col not in df.columns:
            def _get_pred(d):
                if not isinstance(d, dict):
                    return None
                x = d.get(r, None)
                if not isinstance(x, dict):
                    return None
                try:
                    return int(x.get("tax_id"))
                except Exception:
                    return None
            df[pred_col] = df["pred_by_rank"].apply(_get_pred) if "pred_by_rank" in df.columns else None

    # gold_{r} from dense arrays
    for r in ranks:
        gold_col = f"gold_{r}"
        arr = ancestor_at_rank[r]
        df[gold_col] = df["true_tax_id"].apply(lambda t: arr[t] if 0 <= t < len(arr) else -1)

    return df


# ------------------------------------------------------------
# 2) Correctness flags + deepest-rank + rankACC
# ------------------------------------------------------------
def _ensure_ok_columns(df, ranks=RANKS):
    """
    ok_{r} in {1.0, 0.0, NaN}:
      - NaN if gold_r missing (gold_r == -1)
      - 1.0 if pred_r == gold_r
      - 0.0 otherwise
    """
    df = df.copy()
    for r in ranks:
        pred = df[f"pred_{r}"]
        gold = df[f"gold_{r}"]

        has = (gold != -1)
        ok = pd.Series(np.nan, index=df.index, dtype="float")
        ok.loc[has] = (
            pred.loc[has].fillna(-1).astype("int64").to_numpy()
            ==
            gold.loc[has].fillna(-1).astype("int64").to_numpy()
        ).astype(float)

        df[f"ok_{r}"] = ok
    return df


def _deepest_gold_rank(row, ranks=RANKS):
    for r in reversed(ranks):
        if row.get(f"gold_{r}", -1) != -1:
            return r
    return None


def _first_wrong_rank_up_to_target(row, target_rank, ranks=RANKS):
    t_idx = ranks.index(target_rank)
    for r in ranks[: t_idx + 1]:
        if row[f"gold_{r}"] == -1:
            continue
        ok = row[f"ok_{r}"]  # 1.0/0.0/NaN
        if (not pd.isna(ok)) and (ok < 0.5):
            return r
    return None  # fully correct up to target


def _rankACC_for_row(row, target_rank, ranks=RANKS):
    """
    rankACC = m/D where D = index(target)+1, m = # consecutive correct from root until first error.
    Missing gold within k..target is skipped safely.
    """
    D = ranks.index(target_rank) + 1
    m = 0
    for r in ranks[:D]:
        if row[f"gold_{r}"] == -1:
            continue
        ok = row[f"ok_{r}"]
        if pd.isna(ok):
            continue
        if ok >= 0.5:
            m += 1
        else:
            break
    return float(m / max(1, D))


# ------------------------------------------------------------
# 3) Strict matrix + per-rank mean rankACC 
# ------------------------------------------------------------
def strict_error_origin_matrix_and_rankACC(df, ranks=RANKS):
    df2 = _ensure_ok_columns(df, ranks=ranks).copy()

    # deepest rank per row
    df2["deepest_rank"] = df2.apply(lambda row: _deepest_gold_rank(row, ranks=ranks), axis=1)
    df2 = df2[df2["deepest_rank"].notna()].copy()

    counts = pd.DataFrame(0, index=ranks, columns=ranks, dtype=int)
    meta_rows = []

    for target in ranks:
        sub = df2[df2["deepest_rank"] == target].copy()
        n_preds = int(len(sub))

        if n_preds == 0:
            meta_rows.append({
                "rank": target,
                "n_preds_strict": 0,
                "n_errors_strict": 0,
                "strict_error_rate": np.nan,
                "rankACC_mean": np.nan,
            })
            continue

        # rankACC per row
        sub["rankACC"] = sub.apply(lambda row: _rankACC_for_row(row, target, ranks=ranks), axis=1)
        rankACC_mean = float(sub["rankACC"].mean())

        # errors are rows not fully correct up to target
        err_mask = sub["rankACC"] < 0.999999
        errs = sub[err_mask].copy()
        n_errors = int(len(errs))

        if n_errors > 0:
            errs["first_wrong"] = errs.apply(
                lambda row: _first_wrong_rank_up_to_target(row, target, ranks=ranks),
                axis=1
            )
            vc = errs["first_wrong"].value_counts(dropna=True)
            for col_rank, cnt in vc.items():
                counts.loc[target, col_rank] = int(cnt)

        meta_rows.append({
            "rank": target,
            "n_preds_strict": n_preds,
            "n_errors_strict": n_errors,
            "strict_error_rate": float(n_errors / max(1, n_preds)),
            "rankACC_mean": rankACC_mean,
        })

    meta = pd.DataFrame(meta_rows).set_index("rank")
    return counts, meta


# ------------------------------------------------------------
# 4) Plot 
# ------------------------------------------------------------
def plot_strict_matrix_with_rankACC(
    counts_strict,
    meta_strict,
    title,
    ranks=RANKS,
    df=None,
    otu_col="otu_id",
    show_zeros_lower=True,
):
    ranks = list(ranks)

    mat = counts_strict.loc[ranks, ranks].fillna(0).astype(int)
    n_preds = meta_strict.loc[ranks, "n_preds_strict"].fillna(0).astype(int).to_numpy()
    rankacc = meta_strict.loc[ranks, "rankACC_mean"].to_numpy()
    rankacc = np.nan_to_num(rankacc, nan=0.0)

    extra = ""
    if df is not None:
        if otu_col in df.columns:
            n_otus = int(df[otu_col].nunique())
        elif "otu_name" in df.columns:
            n_otus = int(df["otu_name"].nunique())
        else:
            n_otus = None
        n_predictions = int(len(df))
        extra = f"OTUs={n_otus} | N predictions={n_predictions}" if n_otus is not None else f"N predictions={n_predictions}"

    fig = plt.figure(figsize=(10.0, 6.6))
    gs = GridSpec(
        2, 3,
        height_ratios=[0.75, 6.0],
        width_ratios=[20, 3.2, 1.1],
        hspace=0.06,
        wspace=0.08
    )

    # top denominators
    ax_top = fig.add_subplot(gs[0, 0])
    ax_top.bar(range(len(ranks)), n_preds, color="lightgray", edgecolor="none")
    ax_top.set_xlim(-0.5, len(ranks) - 0.5)
    ax_top.set_ylabel("# pred")
    ax_top.set_yticks([])
    ax_top.tick_params(axis="y", left=False, labelleft=False)
    ax_top.spines["left"].set_visible(False)
    ax_top.set_xticks([])
    ax_top.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_top.spines["right"].set_visible(False)
    ax_top.spines["top"].set_visible(False)

    ymax = max(1, int(n_preds.max()) if len(n_preds) else 1)
    ax_top.set_ylim(0, ymax * 1.10)
    for i, n in enumerate(n_preds):
        ax_top.text(i, n + 0.01 * ymax, f"{n}", ha="center", va="bottom", fontsize=9)

    # heatmap
    ax_hm = fig.add_subplot(gs[1, 0], sharex=ax_top)
    data = mat.to_numpy()
    im = ax_hm.imshow(data, aspect="auto", cmap="Blues")

    ax_hm.set_xticks(range(len(ranks)))
    ax_hm.set_xticklabels(ranks)
    ax_hm.set_yticks(range(len(ranks)))
    ax_hm.set_yticklabels(ranks)
    ax_hm.set_xlabel("First wrong rank (error origin)")
    ax_hm.set_ylabel("Deepest labeled rank (strict rows)")

    vmax = int(data.max()) if data.size else 1
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = int(data[i, j])
            is_lower = (j <= i)
            if v > 0:
                ax_hm.text(
                    j, i, f"{v}",
                    ha="center", va="center", fontsize=8,
                    color="white" if v > 0.6 * vmax else "black",
                )
            elif show_zeros_lower and is_lower:
                ax_hm.text(j, i, "0", ha="center", va="center", fontsize=7, color="0.65")

    # rankACC bars
    ax_ra = fig.add_subplot(gs[1, 1], sharey=ax_hm)
    y = np.arange(len(ranks))
    ax_ra.barh(y, rankacc, color="lightgray", edgecolor="none")
    ax_ra.set_xlim(0.0, 1.0)
    ax_ra.set_xlabel("rankACC")
    ax_ra.tick_params(axis="y", left=False, labelleft=False)
    for i, v in enumerate(rankacc):
        ax_ra.text(0.03, i, f"{v*100:.1f}%", va="center", ha="left", fontsize=9, color="black")
    ax_ra.spines["top"].set_visible(False)
    ax_ra.spines["right"].set_visible(False)

    # colorbar
    ax_cb = fig.add_subplot(gs[1, 2])
    cbar = fig.colorbar(im, cax=ax_cb)
    cbar.set_label("Number of errors")

    fig.suptitle(f"{title}\n{extra}", y=0.985, fontsize=13, linespacing=1.3)
    return fig, (ax_top, ax_hm, ax_ra, ax_cb)


# ------------------------------------------------------------
# 5) Global rankACC (one scalar)
# ------------------------------------------------------------
def compute_global_rankACC(df, ranks=RANKS):
    df2 = _ensure_ok_columns(df, ranks=ranks).copy()
    df2["deepest_rank"] = df2.apply(lambda row: _deepest_gold_rank(row, ranks=ranks), axis=1)
    df2 = df2[df2["deepest_rank"].notna()].copy()

    df2["rankACC"] = df2.apply(lambda row: _rankACC_for_row(row, row["deepest_rank"], ranks=ranks), axis=1)
    return float(df2["rankACC"].mean()), df2



def plot_error_origin_matrix_rowpct_colored(
    counts,
    meta,
    title,
    ranks=RANKS,
    df=None,
    otu_col="otu_id",
    show_zeros_lower=True,
    show_percent_text: bool = True,     # annotate each cell with (row %)
    vmax_pct: float = 5.0,              # fixed scale upper bound in percentage points
    vmin_pct: float = 0.0,              # fixed scale lower bound
):
    ranks = list(ranks)

    # counts: errors per (deepest_rank=row, first_wrong=col)
    mat_counts = counts.loc[ranks, ranks].fillna(0).astype(int)
    data_counts = mat_counts.to_numpy()

    # denominators for row %: number of predictions whose deepest rank is that row
    n_preds = meta.loc[ranks, "n_preds_strict"].fillna(0).astype(int).to_numpy()

    # rankACC panel
    rankacc = meta.loc[ranks, "rankACC_mean"].to_numpy()
    rankacc = np.nan_to_num(rankacc, nan=0.0)

    # extra info line
    extra = ""
    if df is not None:
        if otu_col in df.columns:
            n_otus = int(df[otu_col].nunique())
        elif "otu_name" in df.columns:
            n_otus = int(df["otu_name"].nunique())
        else:
            n_otus = None
        n_predictions = int(len(df))
        extra = f"OTUs={n_otus} | N predictions={n_predictions}" if n_otus is not None else f"N predictions={n_predictions}"

    # -------------------------------------------------
    # Build the COLOR matrix: row-normalized percentage
    # pct[i,j] = 100 * count[i,j] / n_preds[i]
    # -------------------------------------------------
    data_pct = np.zeros_like(data_counts, dtype=float)
    for i in range(data_counts.shape[0]):
        denom = float(n_preds[i])
        if denom > 0:
            data_pct[i, :] = 100.0 * (data_counts[i, :] / denom)
        else:
            data_pct[i, :] = 0.0

    fig = plt.figure(figsize=(10.6, 6.8))
    gs = GridSpec(
        2, 3,
        height_ratios=[0.75, 6.0],
        width_ratios=[20, 3.2, 1.1],
        hspace=0.06,
        wspace=0.10
    )

    # -------------------------
    # Top denominators
    # -------------------------
    ax_top = fig.add_subplot(gs[0, 0])
    ax_top.bar(range(len(ranks)), n_preds, color="lightgray", edgecolor="none")
    ax_top.set_xlim(-0.5, len(ranks) - 0.5)
    ax_top.set_ylabel("# predictions")
    ax_top.set_yticks([])
    ax_top.tick_params(axis="y", left=False, labelleft=False)
    ax_top.spines["left"].set_visible(False)
    ax_top.set_xticks([])
    ax_top.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_top.spines["right"].set_visible(False)
    ax_top.spines["top"].set_visible(False)

    ymax = max(1, int(n_preds.max()) if len(n_preds) else 1)
    ax_top.set_ylim(0, ymax * 1.12)
    for i, n in enumerate(n_preds):
        ax_top.text(i, n + 0.01 * ymax, f"{n}", ha="center", va="bottom", fontsize=9)

    # -------------------------
    # Heatmap colored by ROW %
    # -------------------------
    ax_hm = fig.add_subplot(gs[1, 0], sharex=ax_top)

    im = ax_hm.imshow(
        data_pct,
        aspect="auto",
        cmap="Blues",
        vmin=vmin_pct,
        vmax=vmax_pct,
    )

    ax_hm.set_xticks(range(len(ranks)))
    ax_hm.set_xticklabels(ranks)
    ax_hm.set_yticks(range(len(ranks)))
    ax_hm.set_yticklabels(ranks)

    ax_hm.set_xlabel("First incorrect prediction")
    ax_hm.set_ylabel("Deepest available label")

    # annotate: show counts and optionally row %
    for i in range(data_counts.shape[0]):
        for j in range(data_counts.shape[1]):
            v = int(data_counts[i, j])
            is_lower = (j <= i)

            if v > 0:
                pct = float(data_pct[i, j])
                if show_percent_text:
                    txt = f"{v}\n({pct:.2f}%)"
                else:
                    txt = f"{v}"

                # choose text color based on pct scale (not counts)
                txt_color = "white" if pct > 0.6 * vmax_pct else "black"

                ax_hm.text(
                    j, i, txt,
                    ha="center", va="center",
                    fontsize=8,
                    color=txt_color,
                    linespacing=0.9,
                )
            elif show_zeros_lower and is_lower:
                ax_hm.text(j, i, "0", ha="center", va="center", fontsize=7, color="0.65")

    # -------------------------
    # rankACC bars
    # -------------------------
    ax_ra = fig.add_subplot(gs[1, 1], sharey=ax_hm)
    y = np.arange(len(ranks))
    ax_ra.barh(y, rankacc, color="lightgray", edgecolor="none")
    ax_ra.set_xlim(0.0, 1.0)
    ax_ra.set_xlabel("rankACC")
    ax_ra.tick_params(axis="y", left=False, labelleft=False)
    for i, v in enumerate(rankacc):
        ax_ra.text(0.04, i, f"{v*100:.1f}%", va="center", ha="left", fontsize=9, color="black")
    ax_ra.spines["top"].set_visible(False)
    ax_ra.spines["right"].set_visible(False)

    # -------------------------
    # Colorbar (now in %)
    # -------------------------
    ax_cb = fig.add_subplot(gs[1, 2])
    cbar = fig.colorbar(im, cax=ax_cb)
    cbar.set_label("Error rate per row (%)")

    # Title
    #fig.suptitle(f"{title}\n{extra}", y=0.985, fontsize=13, linespacing=1.3)

    return fig, (ax_top, ax_hm, ax_ra, ax_cb)



RANKS = ("k","p","c","o","f","g","s")

def build_accuracy_vs_rankACC_table(df, ranks=RANKS):
    """
    Returns a DataFrame with:
      rows  : k, p, c, o, f, g, s, Global
      cols  : standard_accuracy, rankACC
    """

    # --------------------------------------------------
    # Ensure required columns exist
    # --------------------------------------------------
    df2 = _ensure_ok_columns(df, ranks=ranks).copy()

    if "deepest_rank" not in df2.columns:
        df2["deepest_rank"] = df2.apply(
            lambda row: _deepest_gold_rank(row, ranks=ranks), axis=1
        )

    # --------------------------------------------------
    # Per-rank metrics
    # --------------------------------------------------
    rows = []

    for r in ranks:
        # rows whose deepest available label is r
        sub = df2[df2["deepest_rank"] == r]

        if len(sub) == 0:
            rows.append({
                "rank": r,
                "standard_accuracy": np.nan,
                "rankACC": np.nan,
                "n_rows": 0,
            })
            continue

        # ---- standard accuracy at rank r (micro) ----
        ok_col = f"ok_{r}"
        valid = sub[ok_col].notna()

        if valid.any():
            standard_acc = float(sub.loc[valid, ok_col].mean())
        else:
            standard_acc = np.nan

        # ---- rankACC (same definition as your strict metric) ----
        sub_rankacc = sub.apply(
            lambda row: _rankACC_for_row(row, r, ranks=ranks), axis=1
        )
        rankacc_mean = float(sub_rankacc.mean())

        rows.append({
            "rank": r,
            "standard_accuracy": standard_acc,
            "rankACC": rankacc_mean,
            "n_rows": int(len(sub)),
        })

    per_rank_df = pd.DataFrame(rows).set_index("rank")

    # --------------------------------------------------
    # Global metrics
    # --------------------------------------------------
    global_rankACC, df_with_rankacc = compute_global_rankACC(df2, ranks=ranks)

    # global standard accuracy = exact match at deepest rank
    def _deepest_ok(row):
        r = row["deepest_rank"]
        if r is None:
            return np.nan
        return row.get(f"ok_{r}", np.nan)

    global_standard_acc = float(
        df2.apply(_deepest_ok, axis=1).dropna().mean()
    )

    global_row = pd.DataFrame(
        [{
            "standard_accuracy": global_standard_acc,
            "rankACC": global_rankACC,
            "n_rows": int(len(df2)),
        }],
        index=["Global"]
    )

    # --------------------------------------------------
    # Final table
    # --------------------------------------------------
    table = pd.concat([per_rank_df, global_row], axis=0)

    return table
