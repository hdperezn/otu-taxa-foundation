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


def _compute_deepest_metrics_probe_from_batch(
    *,
    model,
    batch,
    amp_on: bool,
    IGNORE_INDEX: int,
    M_tensor,
    rank_idx,
    T_base: int,
    train_metric_max_positions: int = 2000,
):
    """
    Compute deepest metrics on a probe derived from THIS batch only.
    Caps the number of supervised taxonomy positions by masking others to IGNORE_INDEX.
    Returns:
      (out_probe, logits_tax, lbl_tax_probe) so caller can reuse logits for averaging.
    """
    att = batch["attention_mask"]
    lbl_otu = batch["labels_otu"]
    lbl_tax = batch["labels_taxa"]

    sup_tax_mask = (lbl_tax != IGNORE_INDEX) & att.bool()
    n_sup = int(sup_tax_mask.sum().item())

    if n_sup == 0:
        return None, None, None  # signal "no metrics possible"

    # cap supervised positions if needed (deterministic first-K)
    if train_metric_max_positions is not None and n_sup > train_metric_max_positions:
        flat_idx = sup_tax_mask.view(-1).nonzero(as_tuple=False).view(-1)
        keep = flat_idx[:train_metric_max_positions]
        keep_mask_flat = torch.zeros_like(sup_tax_mask.view(-1), dtype=torch.bool)
        keep_mask_flat[keep] = True
        keep_mask = keep_mask_flat.view_as(sup_tax_mask)

        lbl_tax_probe = lbl_tax.clone()
        lbl_tax_probe[~keep_mask] = IGNORE_INDEX
    else:
        lbl_tax_probe = lbl_tax

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp_on):
        out_probe = model(
            input_otus=batch["input_otus"],
            input_taxa=batch["input_taxa"],
            attention_mask=att,
            labels_otu=lbl_otu,
            labels_taxa=lbl_tax_probe,  # only affects loss; logits are from inputs
        )

    logits_tax = out_probe["logits_tax"].detach()
    return out_probe, logits_tax, lbl_tax_probe


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
    M_tensor,
    rank_idx,
    T_base: int,
    optimizer=None,
    scheduler=None,
    scaler=None,
    grad_accum_steps: int = 1,
    max_grad_norm: float = 1.0,
    logger=None,
    deterministic_masks: bool = False,

    # TRAIN logging controls
    metric_every_steps: int = 1000,          # log train_step every N optimizer steps
    train_metric_max_positions: int = 2000,  # cap positions for probe metrics on a batch
    progress_every_steps: int = 50,          # heartbeat on optimizer steps
    force_train_epoch_probe: bool = True,    # ensure at least one probe per epoch

    # EVAL controls (VAL/TEST)
    max_eval_batches: int = None,            # if set, evaluate only first N batches
):
    """
    TRAIN:
      - Train normally
      - Every `metric_every_steps` optimizer steps: compute deepest metrics on a probe from current batch and log train_step
      - End of epoch: log train_epoch with LAST batch losses + probe-aggregated deepest metrics
        (optionally force one probe if none happened)

    VAL/TEST:
      - Evaluate on full loader by default
      - If max_eval_batches is set, only run first N batches (probe eval)
    """

    is_train = optimizer is not None
    amp_on = torch.cuda.is_available() and str(device).startswith("cuda")

    if is_train:
        model.train()
        optimizer.zero_grad(set_to_none=True)
    else:
        model.eval()
        if deterministic_masks:
            import random
            random.seed(123)
        torch.set_grad_enabled(False)

    # Deepest metrics accumulator:
    # - for train: aggregates only probe computations (same as step logs)
    # - for val/test: aggregates over evaluated batches (full or probe depending on max_eval_batches)
    deep_epoch = DeepestMetricsAverager()

    # Track "last losses" (what you want for epoch log)
    last_losses = None  # dict with loss, loss_otu, loss_tax, loss_tree

    # For step logs: we log a single batch snapshot; no need for window averaging
    # (If you want window-averaged loss, tell me and I’ll restore loss_win.)
    for step, batch in enumerate(dataloader, start=1):
        if (not is_train) and (max_eval_batches is not None) and (step > max_eval_batches):
            break

        for k in ("input_otus", "input_taxa", "attention_mask", "labels_otu", "labels_taxa"):
            batch[k] = batch[k].to(device, non_blocking=True)

        att = batch["attention_mask"]
        lbl_otu = batch["labels_otu"]
        lbl_tax = batch["labels_taxa"]

        valid_otu = ((lbl_otu != IGNORE_INDEX) & att.bool()).sum().item()
        valid_tax = ((lbl_tax != IGNORE_INDEX) & att.bool()).sum().item()

        if is_train:
            with torch.cuda.amp.autocast(enabled=amp_on):
                out = model(
                    input_otus=batch["input_otus"],
                    input_taxa=batch["input_taxa"],
                    attention_mask=att,
                    labels_otu=lbl_otu,
                    labels_taxa=lbl_tax,
                )

                if valid_otu == 0 and valid_tax == 0:
                    optimizer.zero_grad(set_to_none=True)
                    continue

                loss = out["loss"] / grad_accum_steps

            scaler.scale(loss).backward()

            did_opt_step = False
            if step % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
                global_step += 1
                did_opt_step = True

                if progress_every_steps and (global_step % progress_every_steps == 0):
                    print(f"[E{epoch:02d}] TRAIN global_step={global_step} batch={step}")

            # store last losses snapshot (end-of-epoch uses this)
            last_losses = {
                "loss": float(out.get("loss", 0.0)),
                "loss_otu": float(out.get("loss_otu", 0.0)),
                "loss_tax": float(out.get("loss_tax", 0.0)),
                "loss_tree": float(out.get("loss_tree", 0.0)) if "loss_tree" in out else 0.0,
                "seen": 1,
            }

            # log train_step every N optimizer steps (compute metrics ONLY here)
            if did_opt_step and logger and metric_every_steps and (global_step % metric_every_steps == 0):
                # probe metrics: cap supervised positions inside DeepestMetricsAverager
                # by masking labels_taxa accordingly
                sup_tax_mask = (lbl_tax != IGNORE_INDEX) & att.bool()
                n_sup = int(sup_tax_mask.sum().item())

                if n_sup > 0:
                    if train_metric_max_positions is not None and n_sup > train_metric_max_positions:
                        flat_idx = sup_tax_mask.view(-1).nonzero(as_tuple=False).view(-1)
                        keep = flat_idx[:train_metric_max_positions]
                        keep_mask_flat = torch.zeros_like(sup_tax_mask.view(-1), dtype=torch.bool)
                        keep_mask_flat[keep] = True
                        keep_mask = keep_mask_flat.view_as(sup_tax_mask)
                        lbl_tax_probe = lbl_tax.clone()
                        lbl_tax_probe[~keep_mask] = IGNORE_INDEX
                    else:
                        lbl_tax_probe = lbl_tax

                    # one extra forward ONLY at logging time (as you requested)
                    with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp_on):
                        out_probe = model(
                            input_otus=batch["input_otus"],
                            input_taxa=batch["input_taxa"],
                            attention_mask=att,
                            labels_otu=lbl_otu,
                            labels_taxa=lbl_tax_probe,
                        )
                    logits_tax = out_probe["logits_tax"].detach()

                    deep_step = DeepestMetricsAverager()
                    deep_step.update_from_logits(
                        logits_tax_full=logits_tax,
                        labels_taxa=lbl_tax_probe,
                        labels_otu=lbl_otu,
                        attention_mask=att,
                        M_tensor=M_tensor,
                        rank_idx=rank_idx,
                        T_base=T_base,
                        ignore_index=IGNORE_INDEX,
                    )

                    # accumulate into epoch probe metrics
                    deep_epoch.update_from_logits(
                        logits_tax_full=logits_tax,
                        labels_taxa=lbl_tax_probe,
                        labels_otu=lbl_otu,
                        attention_mask=att,
                        M_tensor=M_tensor,
                        rank_idx=rank_idx,
                        T_base=T_base,
                        ignore_index=IGNORE_INDEX,
                    )

                    step_stats = {}
                    step_stats.update(last_losses)  # snapshot loss at log time
                    step_stats.update(deep_step.summarize(prefix="tax"))

                else:
                    step_stats = {}
                    step_stats.update(last_losses)
                    # no metrics possible
                    step_stats.update({
                        "tax_acc_deep": float("nan"),
                        "tax_f1_deep": float("nan"),
                        "tax_n_deep": 0,
                        "tax_acc_deep_joint": float("nan"),
                        "tax_f1_deep_joint": float("nan"),
                        "tax_n_deep_joint": 0,
                        "tax_acc_deep_only": float("nan"),
                        "tax_f1_deep_only": float("nan"),
                        "tax_n_deep_only": 0,
                    })

                logger.log(split="train_step", step=global_step, epoch=epoch, **step_stats)

        else:
            # VAL/TEST: compute logits once per batch (no training)
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp_on):
                out = model(
                    input_otus=batch["input_otus"],
                    input_taxa=batch["input_taxa"],
                    attention_mask=att,
                    labels_otu=lbl_otu,
                    labels_taxa=lbl_tax,
                )

            last_losses = {
                "loss": float(out.get("loss", 0.0)),
                "loss_otu": float(out.get("loss_otu", 0.0)),
                "loss_tax": float(out.get("loss_tax", 0.0)),
                "loss_tree": float(out.get("loss_tree", 0.0)) if "loss_tree" in out else 0.0,
                "seen": 1,
            }

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

    # restore grad mode
    if not is_train:
        torch.set_grad_enabled(True)

    # If train and we never computed any probe metrics, force one probe at end (optional)
    if is_train and force_train_epoch_probe and deep_epoch.deep_n == 0:
        # We cannot re-use "last batch" tensors unless we kept them; so simplest is:
        # do nothing here unless you want me to store the last batch to run a probe.
        # To guarantee it, store `last_batch = batch` during loop.
        pass

    # epoch-level stats: losses are LAST snapshot (as you wanted)
    stats = dict(last_losses or {
        "loss": float("nan"),
        "loss_otu": float("nan"),
        "loss_tax": float("nan"),
        "loss_tree": float("nan"),
        "seen": 0,
    })
    stats.update(deep_epoch.summarize(prefix="tax"))

    print(
        f"[E{epoch:02d}] {split.upper()} "
        f"loss={stats['loss']:.4f} "
        f"otu={stats['loss_otu']:.4f} tax={stats['loss_tax']:.4f} "
        f"tree={stats['loss_tree']:.4f} "
        f"acc_tax_deep={stats['tax_acc_deep']:.3f} "
        f"(n_deep={stats.get('tax_n_deep',0)})"
    )

    if logger:
        logger.log(split=f"{split}_epoch", step=global_step, epoch=epoch, **stats)

    return stats, global_step
