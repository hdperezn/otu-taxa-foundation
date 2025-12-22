import torch
import torch.nn.functional as F

from otu_taxa.joint_hier_loss_metrics_unk import (
    hierarchical_accuracy_f1_per_rank,
    deepest_taxonomy_accuracy_f1,
    unk_prediction_metrics_per_rank,
)

# ----------------------------
# Small utilities / helpers
# ----------------------------

RANKS = ["k", "p", "c", "o", "f", "g", "s"]


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else float("nan")


class LossAverager:
    """Tracks simple running means."""
    def __init__(self):
        self.sum_loss = 0.0
        self.sum_loss_otu = 0.0
        self.sum_loss_tax = 0.0
        self.sum_loss_tree = 0.0
        self.n_batches = 0

    def update(self, out: dict):
        self.sum_loss += float(out.get("loss", 0.0))
        self.sum_loss_otu += float(out.get("loss_otu", 0.0))
        self.sum_loss_tax += float(out.get("loss_tax", 0.0))
        self.sum_loss_tree += float(out.get("loss_tree", 0.0)) if "loss_tree" in out else 0.0
        self.n_batches += 1

    def summarize(self) -> dict:
        n = max(1, self.n_batches)
        return {
            "loss": self.sum_loss / n,
            "loss_otu": self.sum_loss_otu / n,
            "loss_tax": self.sum_loss_tax / n,
            "loss_tree": self.sum_loss_tree / n,
            "seen": self.n_batches,
        }

    def reset(self):
        self.__init__()


class DeepestMetricsAverager:
    """
    Aggregates deepest taxonomy acc as a position-weighted mean,
    and deepest F1 as a mean over batches (matching your previous behavior).
    Also supports JOINT and TAX-ONLY subsets.
    """
    def __init__(self):
        # overall
        self.deep_acc_num = 0.0
        self.deep_n = 0
        self.deep_f1_sum = 0.0
        self.deep_f1_cnt = 0

        # joint
        self.deep_joint_acc_num = 0.0
        self.deep_joint_n = 0
        self.deep_joint_f1_sum = 0.0
        self.deep_joint_f1_cnt = 0

        # tax-only
        self.deep_only_acc_num = 0.0
        self.deep_only_n = 0
        self.deep_only_f1_sum = 0.0
        self.deep_only_f1_cnt = 0

        # bookkeeping: how many supervised taxonomy positions we processed (windowing)
        self.tax_supervised_positions = 0

    def update_from_logits(
        self,
        *,
        logits_tax_full: torch.Tensor,     # [B, L, T]
        labels_taxa: torch.Tensor,         # [B, L]
        labels_otu: torch.Tensor,          # [B, L]
        attention_mask: torch.Tensor,      # [B, L]
        M_tensor: torch.Tensor,
        rank_idx: torch.Tensor,
        T_base: int,
        ignore_index: int,
    ):
        # Count supervised taxonomy positions in THIS batch for windowing
        valid_tax_mask = (labels_taxa != ignore_index) & attention_mask.bool()
        n_valid_tax_pos = int(valid_tax_mask.sum().item())
        self.tax_supervised_positions += n_valid_tax_pos

        # ---------- Overall deepest ----------
        deep = deepest_taxonomy_accuracy_f1(
            logits_tax_full=logits_tax_full,
            labels_taxa=labels_taxa,
            attention_mask=attention_mask,
            M_tensor=M_tensor,
            rank_idx=rank_idx,
            T_base=T_base,
            ignore_index=ignore_index,
        )
        n_deep = int(deep.get("n_deep", 0))
        if n_deep > 0:
            self.deep_acc_num += float(deep["acc_deep"]) * n_deep
            self.deep_n += n_deep
            self.deep_f1_sum += float(deep["f1_deep"])
            self.deep_f1_cnt += 1

        # ---------- Subsets (optional, but you had them and they are often useful) ----------
        joint_mask = (
            (labels_taxa != ignore_index) &
            (labels_otu != ignore_index) &
            attention_mask.bool()
        )
        tax_only_mask = (
            (labels_taxa != ignore_index) &
            (labels_otu == ignore_index) &
            attention_mask.bool()
        )

        if joint_mask.any():
            lbl_tax_joint = labels_taxa.clone()
            lbl_tax_joint[~joint_mask] = ignore_index
            deep_joint = deepest_taxonomy_accuracy_f1(
                logits_tax_full=logits_tax_full,
                labels_taxa=lbl_tax_joint,
                attention_mask=attention_mask,
                M_tensor=M_tensor,
                rank_idx=rank_idx,
                T_base=T_base,
                ignore_index=ignore_index,
            )
            n_deep_j = int(deep_joint.get("n_deep", 0))
            if n_deep_j > 0:
                self.deep_joint_acc_num += float(deep_joint["acc_deep"]) * n_deep_j
                self.deep_joint_n += n_deep_j
                self.deep_joint_f1_sum += float(deep_joint["f1_deep"])
                self.deep_joint_f1_cnt += 1

        if tax_only_mask.any():
            lbl_tax_only = labels_taxa.clone()
            lbl_tax_only[~tax_only_mask] = ignore_index
            deep_only = deepest_taxonomy_accuracy_f1(
                logits_tax_full=logits_tax_full,
                labels_taxa=lbl_tax_only,
                attention_mask=attention_mask,
                M_tensor=M_tensor,
                rank_idx=rank_idx,
                T_base=T_base,
                ignore_index=ignore_index,
            )
            n_deep_o = int(deep_only.get("n_deep", 0))
            if n_deep_o > 0:
                self.deep_only_acc_num += float(deep_only["acc_deep"]) * n_deep_o
                self.deep_only_n += n_deep_o
                self.deep_only_f1_sum += float(deep_only["f1_deep"])
                self.deep_only_f1_cnt += 1

    def summarize(self, *, prefix: str = "tax") -> dict:
        out = {}

        # overall
        out[f"{prefix}_acc_deep"] = _safe_div(self.deep_acc_num, self.deep_n)
        out[f"{prefix}_f1_deep"] = _safe_div(self.deep_f1_sum, self.deep_f1_cnt)
        out[f"{prefix}_n_deep"] = int(self.deep_n)

        # joint
        out[f"{prefix}_acc_deep_joint"] = _safe_div(self.deep_joint_acc_num, self.deep_joint_n)
        out[f"{prefix}_f1_deep_joint"] = _safe_div(self.deep_joint_f1_sum, self.deep_joint_f1_cnt)
        out[f"{prefix}_n_deep_joint"] = int(self.deep_joint_n)

        # tax-only
        out[f"{prefix}_acc_deep_only"] = _safe_div(self.deep_only_acc_num, self.deep_only_n)
        out[f"{prefix}_f1_deep_only"] = _safe_div(self.deep_only_f1_sum, self.deep_only_f1_cnt)
        out[f"{prefix}_n_deep_only"] = int(self.deep_only_n)

        # window bookkeeping
        out[f"{prefix}_tax_supervised_positions"] = int(self.tax_supervised_positions)
        return out

    def reset(self):
        self.__init__()


def _maybe_log_window(
    *,
    logger,
    split: str,
    global_step: int,
    epoch: int,
    loss_win: LossAverager,
    deep_win: DeepestMetricsAverager,
    force: bool = False,
    metric_every_positions: int = 1000,
):
    """
    Logs and resets window stats if we crossed the window threshold
    (or if force=True).
    """
    if logger is None:
        return

    if (not force) and (deep_win.tax_supervised_positions < metric_every_positions):
        return

    stats = {}
    stats.update(loss_win.summarize())
    stats.update(deep_win.summarize(prefix="tax"))

    logger.log(
        split=f"{split}_step",
        step=global_step,
        epoch=epoch,
        **stats,
    )

    loss_win.reset()
    deep_win.reset()


# ----------------------------
# Refactored run_epoch
# ----------------------------

def run_epoch(
    *,
    model,
    dataloader,
    device,
    IGNORE_INDEX,
    split: str,                # "train", "val", "test"
    epoch: int,
    global_step: int,
    # --- tree structures needed for deepest metrics ---
    M_tensor,                  # [T_base, T_base] descendant matrix
    rank_idx,                  # [T_base] rank index 0..6 for k..s
    T_base: int,               # base taxonomy size (no pad/mask)
    optimizer=None,
    scheduler=None,
    scaler=None,
    grad_accum_steps: int = 1,
    max_grad_norm: float = 1.0,
    logger=None,
    deterministic_masks: bool = False,
    # NEW: windowed logging control
    metric_every_positions: int = 1000,   # ~ "1k samples" worth of supervised TAX positions
    progress_every_steps: int = 50,
):
    """
    Epoch runner (same training logic), but:
      - Only deepest taxonomy metrics are computed (overall + joint + tax-only).
      - No per-rank metrics, no UNK metrics.
      - Metrics + loss are logged every ~metric_every_positions supervised TAX positions,
        and once again at end-of-epoch (epoch aggregates).
    """

    is_train = optimizer is not None
    amp_on = (device == "cuda")

    # mode setup
    if is_train:
        model.train()
        optimizer.zero_grad(set_to_none=True)
    else:
        model.eval()
        if deterministic_masks:
            import random
            random.seed(123)
        torch.set_grad_enabled(False)

    # epoch accumulators
    loss_epoch = LossAverager()
    deep_epoch = DeepestMetricsAverager()

    # window accumulators (for frequent logging)
    loss_win = LossAverager()
    deep_win = DeepestMetricsAverager()

    for step, batch in enumerate(dataloader, start=1):
        # move batch to device
        for k in ("input_otus", "input_taxa", "attention_mask", "labels_otu", "labels_taxa"):
            batch[k] = batch[k].to(device, non_blocking=True)

        att = batch["attention_mask"]
        lbl_otu = batch["labels_otu"]
        lbl_tax = batch["labels_taxa"]

        valid_otu = ((lbl_otu != IGNORE_INDEX) & att.bool()).sum().item()
        valid_tax = ((lbl_tax != IGNORE_INDEX) & att.bool()).sum().item()

        if is_train:
            # ---- TRAIN (UNCHANGED LOGIC) ----
            with torch.cuda.amp.autocast(enabled=amp_on):
                out = model(
                    input_otus=batch["input_otus"],
                    input_taxa=batch["input_taxa"],
                    attention_mask=batch["attention_mask"],
                    labels_otu=batch["labels_otu"],
                    labels_taxa=batch["labels_taxa"],
                )

                # skip batch if absolutely no supervision, as before
                if valid_otu == 0 and valid_tax == 0:
                    optimizer.zero_grad(set_to_none=True)
                    continue

                loss = out["loss"] / grad_accum_steps

            scaler.scale(loss).backward()

            if step % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
                global_step += 1
                
                if progress_every_steps and (global_step % progress_every_steps == 0):
                    # print only cheap info; avoid any metric computations here
                    print(
                        f"[E{epoch:02d}] {split.upper()} "
                        f"global_step={global_step} "
                        f"batch={step}"
                    )

        else:
            # ---- VAL / TEST ----
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp_on):
                out = model(
                    input_otus=batch["input_otus"],
                    input_taxa=batch["input_taxa"],
                    attention_mask=batch["attention_mask"],
                    labels_otu=batch["labels_otu"],
                    labels_taxa=batch["labels_taxa"],
                )

        # losses (epoch + window)
        loss_epoch.update(out)
        loss_win.update(out)

        # deepest metrics (epoch + window)
        # IMPORTANT: no extra forward pass; use logits already computed.
        logits_tax = out["logits_tax"].detach()
        deep_epoch.update_from_logits(
            logits_tax_full=logits_tax,
            labels_taxa=lbl_tax,
            labels_otu=lbl_otu,
            attention_mask=att,
            M_tensor=M_tensor,
            rank_idx=rank_idx,
            T_base=T_base,
            ignore_index=IGNORE_INDEX,
        )
        deep_win.update_from_logits(
            logits_tax_full=logits_tax,
            labels_taxa=lbl_tax,
            labels_otu=lbl_otu,
            attention_mask=att,
            M_tensor=M_tensor,
            rank_idx=rank_idx,
            T_base=T_base,
            ignore_index=IGNORE_INDEX,
        )

        # frequent logging: every ~metric_every_positions supervised TAX positions
        _maybe_log_window(
            logger=logger,
            split=split,
            global_step=global_step,
            epoch=epoch,
            loss_win=loss_win,
            deep_win=deep_win,
            force=False,
            metric_every_positions=metric_every_positions,
        )

    # restore grad mode after VAL/TEST
    if not is_train:
        torch.set_grad_enabled(True)

    # flush leftover window stats at epoch end (if any)
    _maybe_log_window(
        logger=logger,
        split=split,
        global_step=global_step,
        epoch=epoch,
        loss_win=loss_win,
        deep_win=deep_win,
        force=True,
        metric_every_positions=metric_every_positions,
    )

    # epoch summary
    stats = {}
    stats.update(loss_epoch.summarize())
    stats.update(deep_epoch.summarize(prefix="tax"))

    # stdout (minimal, scalable)
    print(
        f"[E{epoch:02d}] {split.upper()} "
        f"loss={stats['loss']:.4f} "
        f"otu={stats['loss_otu']:.4f} tax={stats['loss_tax']:.4f} "
        f"tree={stats['loss_tree']:.4f} "
        f"acc_tax_deep={stats['tax_acc_deep']:.3f} "
        f"(n_deep={stats['tax_n_deep']})"
    )

    # epoch-level logger
    if logger:
        logger.log(
            split=f"{split}_epoch",
            step=global_step,
            epoch=epoch,
            **stats,
        )

    return stats, global_step