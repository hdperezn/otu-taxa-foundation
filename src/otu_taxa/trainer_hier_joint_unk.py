import torch
import torch.nn.functional as F

from joint_hier_loss_metrics_unk import (
    hierarchical_accuracy_f1_per_rank,
    deepest_taxonomy_accuracy_f1,
    unk_prediction_metrics_per_rank,
)


# def run_epoch(
#     *,
#     model,
#     dataloader,
#     device,
#     IGNORE_INDEX,
#     split: str,                # "train", "val", "test"
#     epoch: int,
#     global_step: int,
#     # --- tree structures needed for hierarchical metrics ---
#     M_tensor,                  # [T_base, T_base] descendant matrix
#     rank_idx,                  # [T_base] rank index 0..6 for k..s
#     T_base: int,               # base taxonomy size (no pad/mask)
#     unk_ids_by_rank,           # NEW: dict {r: global_id_of_UNK_r}
#     optimizer=None,
#     scheduler=None,
#     scaler=None,
#     grad_accum_steps: int = 1,
#     max_grad_norm: float = 1.0,
#     logger=None,
#     log_every: int = 100,      # (now only affects global_step-based stuff if you reuse it)
#     deterministic_masks: bool = False,
#     compute_train_metrics: bool = False,  # <--- NEW
# ):
#     """
#     Epoch runner for hierarchical taxonomy model.

#     - TRAIN:
#         * First pass: losses + backprop only (no hierarchical metrics).
#         * Optional second pass (if compute_train_metrics=True): eval-only metrics.

#     - VAL / TEST:
#         * Single pass: losses + hierarchical metrics (no backprop).
#     """

#     is_train = optimizer is not None
#     amp_on = (device == "cuda")

#     # -------------------
#     # 1) TRAIN / VAL LOOP
#     # -------------------
#     if is_train:
#         model.train()
#         optimizer.zero_grad(set_to_none=True)
#     else:
#         model.eval()
#         if deterministic_masks:
#             import random
#             random.seed(123)
#         torch.set_grad_enabled(False)

#     # ---- accumulators for losses ----
#     tot_loss = tot_loss_otu = tot_loss_tax = tot_loss_tree = 0.0
#     seen_batches = 0

#     # ---- metric accumulators (will be filled in VAL/TEST, or in a second pass for TRAIN) ----
#     ranks = ['k','p','c','o','f','g','s']

#     rank_acc_num = {ch: 0.0 for ch in ranks}
#     rank_n       = {ch: 0   for ch in ranks}
#     rank_f1_sum  = {ch: 0.0 for ch in ranks}
#     rank_f1_cnt  = {ch: 0   for ch in ranks}

#     deep_acc_num = 0.0
#     deep_n       = 0
#     deep_f1_sum  = 0.0
#     deep_f1_cnt  = 0

#     # NEW: deepest metrics by masking type
#     deep_joint_acc_num = 0.0
#     deep_joint_n       = 0
#     deep_joint_f1_sum  = 0.0
#     deep_joint_f1_cnt  = 0

#     deep_only_acc_num  = 0.0
#     deep_only_n        = 0
#     deep_only_f1_sum   = 0.0
#     deep_only_f1_cnt   = 0

#     # NEW: UNK prediction accumulators (epoch-level)
#     unk_tp_total  = {ch: 0 for ch in ranks}
#     unk_fp_total  = {ch: 0 for ch in ranks}
#     unk_fn_total  = {ch: 0 for ch in ranks}
#     unk_sup_total = {ch: 0 for ch in ranks}

#     # ---- FIRST PASS: training or val/test ----
#     for step, batch in enumerate(dataloader, start=1):

#         # --- move batch to device ---
#         for k in ("input_otus", "input_taxa", "attention_mask", "labels_otu", "labels_taxa"):
#             batch[k] = batch[k].to(device, non_blocking=True)

#         att     = batch["attention_mask"]
#         lbl_otu = batch["labels_otu"]
#         lbl_tax = batch["labels_taxa"]

#         valid_otu = ((lbl_otu != IGNORE_INDEX) & att.bool()).sum().item()
#         valid_tax = ((lbl_tax != IGNORE_INDEX) & att.bool()).sum().item()

#         if is_train:
#             # ---- TRAIN ----
#             with torch.cuda.amp.autocast(enabled=amp_on):
#                 out = model(
#                     input_otus=batch["input_otus"],
#                     input_taxa=batch["input_taxa"],
#                     attention_mask=batch["attention_mask"],
#                     labels_otu=batch["labels_otu"],
#                     labels_taxa=batch["labels_taxa"],
#                 )

#                 # skip batch if absolutely no supervision, as before
#                 if valid_otu == 0 and valid_tax == 0:
#                     optimizer.zero_grad(set_to_none=True)
#                     continue

#                 loss = out["loss"] / grad_accum_steps

#             # backward + optimizer step
#             scaler.scale(loss).backward()

#             if step % grad_accum_steps == 0:
#                 scaler.unscale_(optimizer)
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
#                 scaler.step(optimizer)
#                 scaler.update()
#                 optimizer.zero_grad(set_to_none=True)
#                 if scheduler is not None:
#                     scheduler.step()
#                 global_step += 1

#         else:
#             # ---- VAL / TEST ----
#             with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp_on):
#                 out = model(
#                     input_otus=batch["input_otus"],
#                     input_taxa=batch["input_taxa"],
#                     attention_mask=batch["attention_mask"],
#                     labels_otu=batch["labels_otu"],
#                     labels_taxa=batch["labels_taxa"],
#                 )

#         # ---- accumulate losses ----
#         tot_loss     += float(out["loss"])
#         tot_loss_otu += float(out["loss_otu"])
#         tot_loss_tax += float(out["loss_tax"])
#         if "loss_tree" in out:
#             tot_loss_tree += float(out["loss_tree"])
#         seen_batches += 1

#         # ---- hierarchical taxonomy metrics (per batch) ----
#         # ONLY for val/test in this first pass. Training metrics are optional second pass.
#         if not is_train:
#             logits_tax = out["logits_tax"].detach()

#             # 1) Per-rank metrics
#             per_rank = hierarchical_accuracy_f1_per_rank(
#                 logits_tax_full=logits_tax,
#                 labels_taxa=lbl_tax,
#                 attention_mask=att,
#                 M_tensor=M_tensor,
#                 rank_idx=rank_idx,
#                 T_base=T_base,
#                 ignore_index=IGNORE_INDEX,
#             )

#             for ch in ranks:
#                 n_ch = per_rank[f"n_{ch}"]
#                 if n_ch > 0:
#                     acc_ch = per_rank[f"acc_{ch}"]
#                     f1_ch  = per_rank[f"f1_{ch}"]
#                     rank_acc_num[ch] += acc_ch * n_ch
#                     rank_n[ch]       += n_ch
#                     rank_f1_sum[ch]  += f1_ch
#                     rank_f1_cnt[ch]  += 1

#             # 2) Deepest-known taxonomy metrics (overall)
#             deep = deepest_taxonomy_accuracy_f1(
#                 logits_tax_full=logits_tax,
#                 labels_taxa=lbl_tax,
#                 attention_mask=att,
#                 M_tensor=M_tensor,
#                 rank_idx=rank_idx,
#                 T_base=T_base,
#                 ignore_index=IGNORE_INDEX,
#             )

#             n_deep_batch = deep["n_deep"]
#             if n_deep_batch > 0:
#                 deep_acc_num += deep["acc_deep"] * n_deep_batch
#                 deep_n       += n_deep_batch
#                 deep_f1_sum  += deep["f1_deep"]
#                 deep_f1_cnt  += 1

#             # -------- NEW: deepest metrics by masking type --------
#             # joint positions: OTU and TAX both supervised
#             joint_mask = (
#                 (lbl_tax != IGNORE_INDEX) &
#                 (lbl_otu != IGNORE_INDEX) &
#                 att.bool()
#             )
#             # tax-only positions: TAX supervised, OTU not supervised
#             tax_only_mask = (
#                 (lbl_tax != IGNORE_INDEX) &
#                 (lbl_otu == IGNORE_INDEX) &
#                 att.bool()
#             )

#             # Joint subset
#             if joint_mask.any():
#                 lbl_tax_joint = lbl_tax.clone()
#                 lbl_tax_joint[~joint_mask] = IGNORE_INDEX

#                 deep_joint = deepest_taxonomy_accuracy_f1(
#                     logits_tax_full=logits_tax,
#                     labels_taxa=lbl_tax_joint,
#                     attention_mask=att,
#                     M_tensor=M_tensor,
#                     rank_idx=rank_idx,
#                     T_base=T_base,
#                     ignore_index=IGNORE_INDEX,
#                 )

#                 n_deep_joint_batch = deep_joint["n_deep"]
#                 if n_deep_joint_batch > 0:
#                     deep_joint_acc_num += deep_joint["acc_deep"] * n_deep_joint_batch
#                     deep_joint_n       += n_deep_joint_batch
#                     deep_joint_f1_sum  += deep_joint["f1_deep"]
#                     deep_joint_f1_cnt  += 1

#             # Tax-only subset
#             if tax_only_mask.any():
#                 lbl_tax_only = lbl_tax.clone()
#                 lbl_tax_only[~tax_only_mask] = IGNORE_INDEX

#                 deep_only = deepest_taxonomy_accuracy_f1(
#                     logits_tax_full=logits_tax,
#                     labels_taxa=lbl_tax_only,
#                     attention_mask=att,
#                     M_tensor=M_tensor,
#                     rank_idx=rank_idx,
#                     T_base=T_base,
#                     ignore_index=IGNORE_INDEX,
#                 )

#                 n_deep_only_batch = deep_only["n_deep"]
#                 if n_deep_only_batch > 0:
#                     deep_only_acc_num += deep_only["acc_deep"] * n_deep_only_batch
#                     deep_only_n       += n_deep_only_batch
#                     deep_only_f1_sum  += deep_only["f1_deep"]
#                     deep_only_f1_cnt  += 1

#             # -------- NEW: UNK prediction metrics (per batch, val/test) --------
#             unk_batch = unk_prediction_metrics_per_rank(
#                 logits_tax_full=logits_tax,
#                 labels_taxa=lbl_tax,
#                 attention_mask=att,
#                 M_tensor=M_tensor,
#                 rank_idx=rank_idx,
#                 T_base=T_base,
#                 unk_ids_by_rank=unk_ids_by_rank,
#                 ignore_index=IGNORE_INDEX,
#                 R=7,
#             )

#             for ch in ranks:
#                 sup = unk_batch[f"unk_support_{ch}"]
#                 tp  = unk_batch[f"unk_tp_{ch}"]
#                 fp  = unk_batch[f"unk_fp_{ch}"]
#                 fn  = unk_batch[f"unk_fn_{ch}"]

#                 unk_sup_total[ch] += sup
#                 unk_tp_total[ch]  += tp
#                 unk_fp_total[ch]  += fp
#                 unk_fn_total[ch]  += fn

#     # restore grad mode after VAL/TEST
#     if not is_train:
#         torch.set_grad_enabled(True)

#     # -------------------
#     # 2) OPTIONAL SECOND PASS FOR TRAIN METRICS
#     # -------------------
#     if is_train and compute_train_metrics:
#         model.eval()
#         torch.set_grad_enabled(False)

#         # reset metric accumulators for train metrics
#         rank_acc_num = {ch: 0.0 for ch in ranks}
#         rank_n       = {ch: 0   for ch in ranks}
#         rank_f1_sum  = {ch: 0.0 for ch in ranks}
#         rank_f1_cnt  = {ch: 0   for ch in ranks}
#         deep_acc_num = 0.0
#         deep_n       = 0
#         deep_f1_sum  = 0.0
#         deep_f1_cnt  = 0

#         deep_joint_acc_num = 0.0
#         deep_joint_n       = 0
#         deep_joint_f1_sum  = 0.0
#         deep_joint_f1_cnt  = 0

#         deep_only_acc_num  = 0.0
#         deep_only_n        = 0
#         deep_only_f1_sum   = 0.0
#         deep_only_f1_cnt   = 0

#         # reset UNK accumulators for train-metrics pass
#         unk_tp_total  = {ch: 0 for ch in ranks}
#         unk_fp_total  = {ch: 0 for ch in ranks}
#         unk_fn_total  = {ch: 0 for ch in ranks}
#         unk_sup_total = {ch: 0 for ch in ranks}

#         # limit how much train data we use for metrics
#         # here: up to ~2000 valid taxonomy positions (tokens) per epoch
#         max_train_metric_positions = 2000
#         used_positions = 0

#         for step, batch in enumerate(dataloader, start=1):
#             for k in ("input_otus", "input_taxa", "attention_mask", "labels_otu", "labels_taxa"):
#                 batch[k] = batch[k].to(device, non_blocking=True)

#             att     = batch["attention_mask"]
#             lbl_otu = batch["labels_otu"]
#             lbl_tax = batch["labels_taxa"]

#             # count how many valid taxonomy positions this batch has
#             valid_mask = (lbl_tax != IGNORE_INDEX) & att.bool()
#             n_valid_pos = int(valid_mask.sum().item())
#             if n_valid_pos == 0:
#                 continue

#             # stop if we've already used enough positions for metrics
#             if used_positions >= max_train_metric_positions:
#                 break
#             # clip contribution if this batch would exceed the cap
#             # (we still compute metrics on full batch, but cap the accounting)
#             used_positions += n_valid_pos

#             with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp_on):
#                 out = model(
#                     input_otus=batch["input_otus"],
#                     input_taxa=batch["input_taxa"],
#                     attention_mask=batch["attention_mask"],
#                     labels_otu=batch["labels_otu"],
#                     labels_taxa=batch["labels_taxa"],
#                 )

#             logits_tax = out["logits_tax"].detach()

#             # 1) Per-rank metrics
#             per_rank = hierarchical_accuracy_f1_per_rank(
#                 logits_tax_full=logits_tax,
#                 labels_taxa=lbl_tax,
#                 attention_mask=att,
#                 M_tensor=M_tensor,
#                 rank_idx=rank_idx,
#                 T_base=T_base,
#                 ignore_index=IGNORE_INDEX,
#             )

#             for ch in ranks:
#                 n_ch = per_rank[f"n_{ch}"]
#                 if n_ch > 0:
#                     acc_ch = per_rank[f"acc_{ch}"]
#                     f1_ch  = per_rank[f"f1_{ch}"]
#                     rank_acc_num[ch] += acc_ch * n_ch
#                     rank_n[ch]       += n_ch
#                     rank_f1_sum[ch]  += f1_ch
#                     rank_f1_cnt[ch]  += 1

#             # 2) Deepest-known taxonomy metrics (overall)
#             deep = deepest_taxonomy_accuracy_f1(
#                 logits_tax_full=logits_tax,
#                 labels_taxa=lbl_tax,
#                 attention_mask=att,
#                 M_tensor=M_tensor,
#                 rank_idx=rank_idx,
#                 T_base=T_base,
#                 ignore_index=IGNORE_INDEX,
#             )

#             n_deep_batch = deep["n_deep"]
#             if n_deep_batch > 0:
#                 deep_acc_num += deep["acc_deep"] * n_deep_batch
#                 deep_n       += n_deep_batch
#                 deep_f1_sum  += deep["f1_deep"]
#                 deep_f1_cnt  += 1

#             # -------- NEW: deepest metrics by masking type (train pass) --------
#             joint_mask = (
#                 (lbl_tax != IGNORE_INDEX) &
#                 (lbl_otu != IGNORE_INDEX) &
#                 att.bool()
#             )
#             tax_only_mask = (
#                 (lbl_tax != IGNORE_INDEX) &
#                 (lbl_otu == IGNORE_INDEX) &
#                 att.bool()
#             )

#             if joint_mask.any():
#                 lbl_tax_joint = lbl_tax.clone()
#                 lbl_tax_joint[~joint_mask] = IGNORE_INDEX

#                 deep_joint = deepest_taxonomy_accuracy_f1(
#                     logits_tax_full=logits_tax,
#                     labels_taxa=lbl_tax_joint,
#                     attention_mask=att,
#                     M_tensor=M_tensor,
#                     rank_idx=rank_idx,
#                     T_base=T_base,
#                     ignore_index=IGNORE_INDEX,
#                 )

#                 n_deep_joint_batch = deep_joint["n_deep"]
#                 if n_deep_joint_batch > 0:
#                     deep_joint_acc_num += deep_joint["acc_deep"] * n_deep_joint_batch
#                     deep_joint_n       += n_deep_joint_batch
#                     deep_joint_f1_sum  += deep_joint["f1_deep"]
#                     deep_joint_f1_cnt  += 1

#             if tax_only_mask.any():
#                 lbl_tax_only = lbl_tax.clone()
#                 lbl_tax_only[~tax_only_mask] = IGNORE_INDEX

#                 deep_only = deepest_taxonomy_accuracy_f1(
#                     logits_tax_full=logits_tax,
#                     labels_taxa=lbl_tax_only,
#                     attention_mask=att,
#                     M_tensor=M_tensor,
#                     rank_idx=rank_idx,
#                     T_base=T_base,
#                     ignore_index=IGNORE_INDEX,
#                 )

#                 n_deep_only_batch = deep_only["n_deep"]
#                 if n_deep_only_batch > 0:
#                     deep_only_acc_num += deep_only["acc_deep"] * n_deep_only_batch
#                     deep_only_n       += n_deep_only_batch
#                     deep_only_f1_sum  += deep_only["f1_deep"]
#                     deep_only_f1_cnt  += 1

#             # -------- NEW: UNK prediction metrics (train-metrics pass) --------
#             unk_batch = unk_prediction_metrics_per_rank(
#                 logits_tax_full=logits_tax,
#                 labels_taxa=lbl_tax,
#                 attention_mask=att,
#                 M_tensor=M_tensor,
#                 rank_idx=rank_idx,
#                 T_base=T_base,
#                 unk_ids_by_rank=unk_ids_by_rank,
#                 ignore_index=IGNORE_INDEX,
#                 R=7,
#             )

#             for ch in ranks:
#                 sup = unk_batch[f"unk_support_{ch}"]
#                 tp  = unk_batch[f"unk_tp_{ch}"]
#                 fp  = unk_batch[f"unk_fp_{ch}"]
#                 fn  = unk_batch[f"unk_fn_{ch}"]

#                 unk_sup_total[ch] += sup
#                 unk_tp_total[ch]  += tp
#                 unk_fp_total[ch]  += fp
#                 unk_fn_total[ch]  += fn

#         torch.set_grad_enabled(True)
#         model.train()   # leave model in train mode for next epoch

#     # -------------------
#     # 3) AGGREGATE STATS
#     # -------------------
#     stats = dict(
#         loss      = tot_loss      / max(1, seen_batches),
#         loss_otu  = tot_loss_otu  / max(1, seen_batches),
#         loss_tax  = tot_loss_tax  / max(1, seen_batches),
#         loss_tree = tot_loss_tree / max(1, seen_batches),
#         seen      = seen_batches,
#     )

#     # per-rank final metrics
#     ranks = ['k','p','c','o','f','g','s']
#     for ch in ranks:
#         n_ch = rank_n[ch]
#         if n_ch > 0:
#             acc_ch = rank_acc_num[ch] / n_ch
#         else:
#             acc_ch = float("nan")

#         if rank_f1_cnt[ch] > 0:
#             f1_ch = rank_f1_sum[ch] / rank_f1_cnt[ch]
#         else:
#             f1_ch = float("nan")

#         stats[f"tax_acc_{ch}"] = acc_ch
#         stats[f"tax_f1_{ch}"]  = f1_ch
#         stats[f"tax_n_{ch}"]   = n_ch

#     # deepest-level final metrics (overall)
#     if deep_n > 0:
#         acc_deep = deep_acc_num / deep_n
#     else:
#         acc_deep = float("nan")
#     if deep_f1_cnt > 0:
#         f1_deep = deep_f1_sum / deep_f1_cnt
#     else:
#         f1_deep = float("nan")

#     stats["tax_acc_deep"] = acc_deep
#     stats["tax_f1_deep"]  = f1_deep
#     stats["tax_n_deep"]   = deep_n

#     # NEW: deepest-level metrics by masking type
#     if deep_joint_n > 0:
#         acc_deep_joint = deep_joint_acc_num / deep_joint_n
#     else:
#         acc_deep_joint = float("nan")
#     if deep_joint_f1_cnt > 0:
#         f1_deep_joint = deep_joint_f1_sum / deep_joint_f1_cnt
#     else:
#         f1_deep_joint = float("nan")

#     stats["tax_acc_deep_joint"] = acc_deep_joint
#     stats["tax_f1_deep_joint"]  = f1_deep_joint
#     stats["tax_n_deep_joint"]   = deep_joint_n

#     if deep_only_n > 0:
#         acc_deep_only = deep_only_acc_num / deep_only_n
#     else:
#         acc_deep_only = float("nan")
#     if deep_only_f1_cnt > 0:
#         f1_deep_only = deep_only_f1_sum / deep_only_f1_cnt
#     else:
#         f1_deep_only = float("nan")

#     stats["tax_acc_deep_only"] = acc_deep_only
#     stats["tax_f1_deep_only"]  = f1_deep_only
#     stats["tax_n_deep_only"]   = deep_only_n

#     # NEW: aggregate UNK metrics into stats
#     for ch in ranks:
#         sup = unk_sup_total[ch]
#         tp  = unk_tp_total[ch]
#         fp  = unk_fp_total[ch]
#         fn  = unk_fn_total[ch]

#         if sup > 0:
#             recall = tp / sup
#         else:
#             recall = float("nan")

#         denom = tp + fp
#         if denom > 0:
#             precision = tp / denom
#         else:
#             precision = float("nan")

#         stats[f"unk_recall_{ch}"]    = recall
#         stats[f"unk_precision_{ch}"] = precision
#         stats[f"unk_support_{ch}"]   = sup
#         stats[f"unk_fp_{ch}"]        = fp
#         stats[f"unk_fn_{ch}"]        = fn

#     # ---- summary prints ----
#     print(
#         f"[E{epoch:02d}] {split.upper()} "
#         f"loss={stats['loss']:.4f}  "
#         f"otu={stats['loss_otu']:.4f}  tax={stats['loss_tax']:.4f}  "
#         f"tree={stats['loss_tree']:.4f}  "
#         f"acc_tax_deep={stats['tax_acc_deep']:.3f}"
#     )

#     print(f"[{split.upper()}] per-rank taxonomy metrics:")
#     for ch in ranks:
#         print(
#             f"  {ch}: "
#             f"acc={stats[f'tax_acc_{ch}']:.3f} "
#             f"F1={stats[f'tax_f1_{ch}']:.3f} "
#             f"n={stats[f'tax_n_{ch}']}"
#         )

#     # NEW: UNK prediction metrics, reported separately
#     print(f"[{split.upper()}] UNK prediction metrics per rank:")
#     for ch in ranks:
#         sup = stats[f"unk_support_{ch}"]
#         if sup > 0:
#             print(
#                 f"  {ch}: "
#                 f"recall={stats[f'unk_recall_{ch}']:.3f} "
#                 f"precision={stats[f'unk_precision_{ch}']:.3f} "
#                 f"support={sup} "
#                 f"fp={stats[f'unk_fp_{ch}']} "
#                 f"fn={stats[f'unk_fn_{ch}']}"
#             )

#     if logger:
#         logger.log(
#             split=f"{split}_epoch",
#             step=global_step,
#             epoch=epoch,
#             **stats,
#         )

#     return stats, global_step


import torch
import torch.nn.functional as F

from joint_hier_loss_metrics_unk import (
    hierarchical_accuracy_f1_per_rank,
    deepest_taxonomy_accuracy_f1,
    unk_prediction_metrics_per_rank,
)


def run_epoch(
    *,
    model,
    dataloader,
    device,
    IGNORE_INDEX,
    split: str,                # "train", "val", "test"
    epoch: int,
    global_step: int,
    # --- tree structures needed for hierarchical metrics ---
    M_tensor,                  # [T_base, T_base] descendant matrix
    rank_idx,                  # [T_base] rank index 0..6 for k..s
    T_base: int,               # base taxonomy size (no pad/mask)
    unk_ids_by_rank,           # dict {r: global_id_of_UNK_r}
    optimizer=None,
    scheduler=None,
    scaler=None,
    grad_accum_steps: int = 1,
    max_grad_norm: float = 1.0,
    logger=None,
    log_every: int = 100,
    deterministic_masks: bool = False,
    compute_train_metrics: bool = False,
):
    """
    Epoch runner for hierarchical taxonomy model.

    - TRAIN:
        * First pass: losses + backprop only (no hierarchical metrics).
        * Optional second pass (if compute_train_metrics=True): eval-only metrics.

    - VAL / TEST:
        * Single pass: losses + hierarchical metrics (no backprop).
    """

    is_train = optimizer is not None
    amp_on = (device == "cuda")

    # -------------------
    # 1) TRAIN / VAL LOOP
    # -------------------
    if is_train:
        model.train()
        optimizer.zero_grad(set_to_none=True)
    else:
        model.eval()
        if deterministic_masks:
            import random
            random.seed(123)
        torch.set_grad_enabled(False)

    # ---- accumulators for losses ----
    tot_loss = tot_loss_otu = tot_loss_tax = tot_loss_tree = 0.0
    seen_batches = 0

    # ---- metric accumulators (will be filled in VAL/TEST, or in a second pass for TRAIN) ----
    ranks = ['k','p','c','o','f','g','s']

    rank_acc_num = {ch: 0.0 for ch in ranks}
    rank_n       = {ch: 0   for ch in ranks}
    rank_f1_sum  = {ch: 0.0 for ch in ranks}
    rank_f1_cnt  = {ch: 0   for ch in ranks}

    deep_acc_num = 0.0
    deep_n       = 0
    deep_f1_sum  = 0.0
    deep_f1_cnt  = 0

    # deepest metrics by masking type
    deep_joint_acc_num = 0.0
    deep_joint_n       = 0
    deep_joint_f1_sum  = 0.0
    deep_joint_f1_cnt  = 0

    deep_only_acc_num  = 0.0
    deep_only_n        = 0
    deep_only_f1_sum   = 0.0
    deep_only_f1_cnt   = 0

    # ===== UNK prediction accumulators (epoch-level) =====
    # overall (ALL masked taxa positions)
    unk_tp_total_all  = {ch: 0 for ch in ranks}
    unk_fp_total_all  = {ch: 0 for ch in ranks}
    unk_fn_total_all  = {ch: 0 for ch in ranks}
    unk_sup_total_all = {ch: 0 for ch in ranks}

    # JOINT: OTU+tax both supervised/masked
    unk_tp_total_joint  = {ch: 0 for ch in ranks}
    unk_fp_total_joint  = {ch: 0 for ch in ranks}
    unk_fn_total_joint  = {ch: 0 for ch in ranks}
    unk_sup_total_joint = {ch: 0 for ch in ranks}

    # TAX-ONLY: taxonomy masked, OTU not supervised
    unk_tp_total_only  = {ch: 0 for ch in ranks}
    unk_fp_total_only  = {ch: 0 for ch in ranks}
    unk_fn_total_only  = {ch: 0 for ch in ranks}
    unk_sup_total_only = {ch: 0 for ch in ranks}

    # ---- FIRST PASS: training or val/test ----
    for step, batch in enumerate(dataloader, start=1):

        # --- move batch to device ---
        for k in ("input_otus", "input_taxa", "attention_mask", "labels_otu", "labels_taxa"):
            batch[k] = batch[k].to(device, non_blocking=True)

        att     = batch["attention_mask"]
        lbl_otu = batch["labels_otu"]
        lbl_tax = batch["labels_taxa"]

        valid_otu = ((lbl_otu != IGNORE_INDEX) & att.bool()).sum().item()
        valid_tax = ((lbl_tax != IGNORE_INDEX) & att.bool()).sum().item()

        if is_train:
            # ---- TRAIN ----
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

            # backward + optimizer step
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

        # ---- accumulate losses ----
        tot_loss     += float(out["loss"])
        tot_loss_otu += float(out["loss_otu"])
        tot_loss_tax += float(out["loss_tax"])
        if "loss_tree" in out:
            tot_loss_tree += float(out["loss_tree"])
        seen_batches += 1

        # ---- hierarchical taxonomy metrics (per batch) ----
        if not is_train:
            logits_tax = out["logits_tax"].detach()

            # 1) Per-rank metrics
            per_rank = hierarchical_accuracy_f1_per_rank(
                logits_tax_full=logits_tax,
                labels_taxa=lbl_tax,
                attention_mask=att,
                M_tensor=M_tensor,
                rank_idx=rank_idx,
                T_base=T_base,
                ignore_index=IGNORE_INDEX,
            )

            for ch in ranks:
                n_ch = per_rank[f"n_{ch}"]
                if n_ch > 0:
                    acc_ch = per_rank[f"acc_{ch}"]
                    f1_ch  = per_rank[f"f1_{ch}"]
                    rank_acc_num[ch] += acc_ch * n_ch
                    rank_n[ch]       += n_ch
                    rank_f1_sum[ch]  += f1_ch
                    rank_f1_cnt[ch]  += 1

            # 2) Deepest-known taxonomy metrics (overall)
            deep = deepest_taxonomy_accuracy_f1(
                logits_tax_full=logits_tax,
                labels_taxa=lbl_tax,
                attention_mask=att,
                M_tensor=M_tensor,
                rank_idx=rank_idx,
                T_base=T_base,
                ignore_index=IGNORE_INDEX,
            )

            n_deep_batch = deep["n_deep"]
            if n_deep_batch > 0:
                deep_acc_num += deep["acc_deep"] * n_deep_batch
                deep_n       += n_deep_batch
                deep_f1_sum  += deep["f1_deep"]
                deep_f1_cnt  += 1

            # -------- deepest metrics by masking type --------
            joint_mask = (
                (lbl_tax != IGNORE_INDEX) &
                (lbl_otu != IGNORE_INDEX) &
                att.bool()
            )
            tax_only_mask = (
                (lbl_tax != IGNORE_INDEX) &
                (lbl_otu == IGNORE_INDEX) &
                att.bool()
            )

            # Joint subset
            if joint_mask.any():
                lbl_tax_joint = lbl_tax.clone()
                lbl_tax_joint[~joint_mask] = IGNORE_INDEX

                deep_joint = deepest_taxonomy_accuracy_f1(
                    logits_tax_full=logits_tax,
                    labels_taxa=lbl_tax_joint,
                    attention_mask=att,
                    M_tensor=M_tensor,
                    rank_idx=rank_idx,
                    T_base=T_base,
                    ignore_index=IGNORE_INDEX,
                )

                n_deep_joint_batch = deep_joint["n_deep"]
                if n_deep_joint_batch > 0:
                    deep_joint_acc_num += deep_joint["acc_deep"] * n_deep_joint_batch
                    deep_joint_n       += n_deep_joint_batch
                    deep_joint_f1_sum  += deep_joint["f1_deep"]
                    deep_joint_f1_cnt  += 1

            # Tax-only subset
            if tax_only_mask.any():
                lbl_tax_only = lbl_tax.clone()
                lbl_tax_only[~tax_only_mask] = IGNORE_INDEX

                deep_only = deepest_taxonomy_accuracy_f1(
                    logits_tax_full=logits_tax,
                    labels_taxa=lbl_tax_only,
                    attention_mask=att,
                    M_tensor=M_tensor,
                    rank_idx=rank_idx,
                    T_base=T_base,
                    ignore_index=IGNORE_INDEX,
                )

                n_deep_only_batch = deep_only["n_deep"]
                if n_deep_only_batch > 0:
                    deep_only_acc_num += deep_only["acc_deep"] * n_deep_only_batch
                    deep_only_n       += n_deep_only_batch
                    deep_only_f1_sum  += deep_only["f1_deep"]
                    deep_only_f1_cnt  += 1

            # ================= UNK metrics (VAL/TEST) =================

            # 1) ALL masked taxonomy positions
            unk_all = unk_prediction_metrics_per_rank(
                logits_tax_full=logits_tax,
                labels_taxa=lbl_tax,
                attention_mask=att,
                M_tensor=M_tensor,
                rank_idx=rank_idx,
                T_base=T_base,
                unk_ids_by_rank=unk_ids_by_rank,
                ignore_index=IGNORE_INDEX,
                R=7,
            )

            for ch in ranks:
                sup = unk_all[f"unk_support_{ch}"]
                tp  = unk_all[f"unk_tp_{ch}"]
                fp  = unk_all[f"unk_fp_{ch}"]
                fn  = unk_all[f"unk_fn_{ch}"]

                unk_sup_total_all[ch] += sup
                unk_tp_total_all[ch]  += tp
                unk_fp_total_all[ch]  += fp
                unk_fn_total_all[ch]  += fn

            # 2) JOINT subset (OTU+tax masked)
            if joint_mask.any():
                lbl_tax_joint = lbl_tax.clone()
                lbl_tax_joint[~joint_mask] = IGNORE_INDEX

                unk_joint = unk_prediction_metrics_per_rank(
                    logits_tax_full=logits_tax,
                    labels_taxa=lbl_tax_joint,
                    attention_mask=att,
                    M_tensor=M_tensor,
                    rank_idx=rank_idx,
                    T_base=T_base,
                    unk_ids_by_rank=unk_ids_by_rank,
                    ignore_index=IGNORE_INDEX,
                    R=7,
                )

                for ch in ranks:
                    sup = unk_joint[f"unk_support_{ch}"]
                    tp  = unk_joint[f"unk_tp_{ch}"]
                    fp  = unk_joint[f"unk_fp_{ch}"]
                    fn  = unk_joint[f"unk_fn_{ch}"]

                    unk_sup_total_joint[ch] += sup
                    unk_tp_total_joint[ch]  += tp
                    unk_fp_total_joint[ch]  += fp
                    unk_fn_total_joint[ch]  += fn

            # 3) TAX-ONLY subset
            if tax_only_mask.any():
                lbl_tax_only = lbl_tax.clone()
                lbl_tax_only[~tax_only_mask] = IGNORE_INDEX

                unk_only = unk_prediction_metrics_per_rank(
                    logits_tax_full=logits_tax,
                    labels_taxa=lbl_tax_only,
                    attention_mask=att,
                    M_tensor=M_tensor,
                    rank_idx=rank_idx,
                    T_base=T_base,
                    unk_ids_by_rank=unk_ids_by_rank,
                    ignore_index=IGNORE_INDEX,
                    R=7,
                )

                for ch in ranks:
                    sup = unk_only[f"unk_support_{ch}"]
                    tp  = unk_only[f"unk_tp_{ch}"]
                    fp  = unk_only[f"unk_fp_{ch}"]
                    fn  = unk_only[f"unk_fn_{ch}"]

                    unk_sup_total_only[ch] += sup
                    unk_tp_total_only[ch]  += tp
                    unk_fp_total_only[ch]  += fp
                    unk_fn_total_only[ch]  += fn

    # restore grad mode after VAL/TEST
    if not is_train:
        torch.set_grad_enabled(True)

    # -------------------
    # 2) OPTIONAL SECOND PASS FOR TRAIN METRICS
    # -------------------
    if is_train and compute_train_metrics:
        model.eval()
        torch.set_grad_enabled(False)

        # reset metric accumulators for train metrics
        rank_acc_num = {ch: 0.0 for ch in ranks}
        rank_n       = {ch: 0   for ch in ranks}
        rank_f1_sum  = {ch: 0.0 for ch in ranks}
        rank_f1_cnt  = {ch: 0   for ch in ranks}
        deep_acc_num = 0.0
        deep_n       = 0
        deep_f1_sum  = 0.0
        deep_f1_cnt  = 0

        deep_joint_acc_num = 0.0
        deep_joint_n       = 0
        deep_joint_f1_sum  = 0.0
        deep_joint_f1_cnt  = 0

        deep_only_acc_num  = 0.0
        deep_only_n        = 0
        deep_only_f1_sum   = 0.0
        deep_only_f1_cnt   = 0

        # reset UNK accumulators for train-metrics pass
        unk_tp_total_all  = {ch: 0 for ch in ranks}
        unk_fp_total_all  = {ch: 0 for ch in ranks}
        unk_fn_total_all  = {ch: 0 for ch in ranks}
        unk_sup_total_all = {ch: 0 for ch in ranks}

        unk_tp_total_joint  = {ch: 0 for ch in ranks}
        unk_fp_total_joint  = {ch: 0 for ch in ranks}
        unk_fn_total_joint  = {ch: 0 for ch in ranks}
        unk_sup_total_joint = {ch: 0 for ch in ranks}

        unk_tp_total_only  = {ch: 0 for ch in ranks}
        unk_fp_total_only  = {ch: 0 for ch in ranks}
        unk_fn_total_only  = {ch: 0 for ch in ranks}
        unk_sup_total_only = {ch: 0 for ch in ranks}

        # limit how much train data we use for metrics
        max_train_metric_positions = 2000
        used_positions = 0

        for step, batch in enumerate(dataloader, start=1):
            for k in ("input_otus", "input_taxa", "attention_mask", "labels_otu", "labels_taxa"):
                batch[k] = batch[k].to(device, non_blocking=True)

            att     = batch["attention_mask"]
            lbl_otu = batch["labels_otu"]
            lbl_tax = batch["labels_taxa"]

            valid_mask = (lbl_tax != IGNORE_INDEX) & att.bool()
            n_valid_pos = int(valid_mask.sum().item())
            if n_valid_pos == 0:
                continue

            if used_positions >= max_train_metric_positions:
                break
            used_positions += n_valid_pos

            with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp_on):
                out = model(
                    input_otus=batch["input_otus"],
                    input_taxa=batch["input_taxa"],
                    attention_mask=batch["attention_mask"],
                    labels_otu=batch["labels_otu"],
                    labels_taxa=batch["labels_taxa"],
                )

            logits_tax = out["logits_tax"].detach()

            # 1) Per-rank metrics
            per_rank = hierarchical_accuracy_f1_per_rank(
                logits_tax_full=logits_tax,
                labels_taxa=lbl_tax,
                attention_mask=att,
                M_tensor=M_tensor,
                rank_idx=rank_idx,
                T_base=T_base,
                ignore_index=IGNORE_INDEX,
            )

            for ch in ranks:
                n_ch = per_rank[f"n_{ch}"]
                if n_ch > 0:
                    acc_ch = per_rank[f"acc_{ch}"]
                    f1_ch  = per_rank[f"f1_{ch}"]
                    rank_acc_num[ch] += acc_ch * n_ch
                    rank_n[ch]       += n_ch
                    rank_f1_sum[ch]  += f1_ch
                    rank_f1_cnt[ch]  += 1

            # 2) Deepest-known taxonomy metrics (overall)
            deep = deepest_taxonomy_accuracy_f1(
                logits_tax_full=logits_tax,
                labels_taxa=lbl_tax,
                attention_mask=att,
                M_tensor=M_tensor,
                rank_idx=rank_idx,
                T_base=T_base,
                ignore_index=IGNORE_INDEX,
            )

            n_deep_batch = deep["n_deep"]
            if n_deep_batch > 0:
                deep_acc_num += deep["acc_deep"] * n_deep_batch
                deep_n       += n_deep_batch
                deep_f1_sum  += deep["f1_deep"]
                deep_f1_cnt  += 1

            # deepest metrics by masking type (train-metrics pass)
            joint_mask = (
                (lbl_tax != IGNORE_INDEX) &
                (lbl_otu != IGNORE_INDEX) &
                att.bool()
            )
            tax_only_mask = (
                (lbl_tax != IGNORE_INDEX) &
                (lbl_otu == IGNORE_INDEX) &
                att.bool()
            )

            if joint_mask.any():
                lbl_tax_joint = lbl_tax.clone()
                lbl_tax_joint[~joint_mask] = IGNORE_INDEX

                deep_joint = deepest_taxonomy_accuracy_f1(
                    logits_tax_full=logits_tax,
                    labels_taxa=lbl_tax_joint,
                    attention_mask=att,
                    M_tensor=M_tensor,
                    rank_idx=rank_idx,
                    T_base=T_base,
                    ignore_index=IGNORE_INDEX,
                )

                n_deep_joint_batch = deep_joint["n_deep"]
                if n_deep_joint_batch > 0:
                    deep_joint_acc_num += deep_joint["acc_deep"] * n_deep_joint_batch
                    deep_joint_n       += n_deep_joint_batch
                    deep_joint_f1_sum  += deep_joint["f1_deep"]
                    deep_joint_f1_cnt  += 1

            if tax_only_mask.any():
                lbl_tax_only = lbl_tax.clone()
                lbl_tax_only[~tax_only_mask] = IGNORE_INDEX

                deep_only = deepest_taxonomy_accuracy_f1(
                    logits_tax_full=logits_tax,
                    labels_taxa=lbl_tax_only,
                    attention_mask=att,
                    M_tensor=M_tensor,
                    rank_idx=rank_idx,
                    T_base=T_base,
                    ignore_index=IGNORE_INDEX,
                )

                n_deep_only_batch = deep_only["n_deep"]
                if n_deep_only_batch > 0:
                    deep_only_acc_num += deep_only["acc_deep"] * n_deep_only_batch
                    deep_only_n       += n_deep_only_batch
                    deep_only_f1_sum  += deep_only["f1_deep"]
                    deep_only_f1_cnt  += 1

            # ============ UNK metrics (train-metrics pass) ============

            # ALL
            unk_all = unk_prediction_metrics_per_rank(
                logits_tax_full=logits_tax,
                labels_taxa=lbl_tax,
                attention_mask=att,
                M_tensor=M_tensor,
                rank_idx=rank_idx,
                T_base=T_base,
                unk_ids_by_rank=unk_ids_by_rank,
                ignore_index=IGNORE_INDEX,
                R=7,
            )
            for ch in ranks:
                sup = unk_all[f"unk_support_{ch}"]
                tp  = unk_all[f"unk_tp_{ch}"]
                fp  = unk_all[f"unk_fp_{ch}"]
                fn  = unk_all[f"unk_fn_{ch}"]

                unk_sup_total_all[ch] += sup
                unk_tp_total_all[ch]  += tp
                unk_fp_total_all[ch]  += fp
                unk_fn_total_all[ch]  += fn

            # JOINT
            if joint_mask.any():
                lbl_tax_joint = lbl_tax.clone()
                lbl_tax_joint[~joint_mask] = IGNORE_INDEX

                unk_joint = unk_prediction_metrics_per_rank(
                    logits_tax_full=logits_tax,
                    labels_taxa=lbl_tax_joint,
                    attention_mask=att,
                    M_tensor=M_tensor,
                    rank_idx=rank_idx,
                    T_base=T_base,
                    unk_ids_by_rank=unk_ids_by_rank,
                    ignore_index=IGNORE_INDEX,
                    R=7,
                )
                for ch in ranks:
                    sup = unk_joint[f"unk_support_{ch}"]
                    tp  = unk_joint[f"unk_tp_{ch}"]
                    fp  = unk_joint[f"unk_fp_{ch}"]
                    fn  = unk_joint[f"unk_fn_{ch}"]

                    unk_sup_total_joint[ch] += sup
                    unk_tp_total_joint[ch]  += tp
                    unk_fp_total_joint[ch]  += fp
                    unk_fn_total_joint[ch]  += fn

            # TAX-ONLY
            if tax_only_mask.any():
                lbl_tax_only = lbl_tax.clone()
                lbl_tax_only[~tax_only_mask] = IGNORE_INDEX

                unk_only = unk_prediction_metrics_per_rank(
                    logits_tax_full=logits_tax,
                    labels_taxa=lbl_tax_only,
                    attention_mask=att,
                    M_tensor=M_tensor,
                    rank_idx=rank_idx,
                    T_base=T_base,
                    unk_ids_by_rank=unk_ids_by_rank,
                    ignore_index=IGNORE_INDEX,
                    R=7,
                )
                for ch in ranks:
                    sup = unk_only[f"unk_support_{ch}"]
                    tp  = unk_only[f"unk_tp_{ch}"]
                    fp  = unk_only[f"unk_fp_{ch}"]
                    fn  = unk_only[f"unk_fn_{ch}"]

                    unk_sup_total_only[ch] += sup
                    unk_tp_total_only[ch]  += tp
                    unk_fp_total_only[ch]  += fp
                    unk_fn_total_only[ch]  += fn

        torch.set_grad_enabled(True)
        model.train()

    # -------------------
    # 3) AGGREGATE STATS
    # -------------------
    stats = dict(
        loss      = tot_loss      / max(1, seen_batches),
        loss_otu  = tot_loss_otu  / max(1, seen_batches),
        loss_tax  = tot_loss_tax  / max(1, seen_batches),
        loss_tree = tot_loss_tree / max(1, seen_batches),
        seen      = seen_batches,
    )

    # per-rank final metrics
    for ch in ranks:
        n_ch = rank_n[ch]
        if n_ch > 0:
            acc_ch = rank_acc_num[ch] / n_ch
        else:
            acc_ch = float("nan")

        if rank_f1_cnt[ch] > 0:
            f1_ch = rank_f1_sum[ch] / rank_f1_cnt[ch]
        else:
            f1_ch = float("nan")

        stats[f"tax_acc_{ch}"] = acc_ch
        stats[f"tax_f1_{ch}"]  = f1_ch
        stats[f"tax_n_{ch}"]   = n_ch

    # deepest-level final metrics (overall)
    if deep_n > 0:
        acc_deep = deep_acc_num / deep_n
    else:
        acc_deep = float("nan")
    if deep_f1_cnt > 0:
        f1_deep = deep_f1_sum / deep_f1_cnt
    else:
        f1_deep = float("nan")

    stats["tax_acc_deep"] = acc_deep
    stats["tax_f1_deep"]  = f1_deep
    stats["tax_n_deep"]   = deep_n

    # deepest-level metrics by masking type
    if deep_joint_n > 0:
        acc_deep_joint = deep_joint_acc_num / deep_joint_n
    else:
        acc_deep_joint = float("nan")
    if deep_joint_f1_cnt > 0:
        f1_deep_joint = deep_joint_f1_sum / deep_joint_f1_cnt
    else:
        f1_deep_joint = float("nan")

    stats["tax_acc_deep_joint"] = acc_deep_joint
    stats["tax_f1_deep_joint"]  = f1_deep_joint
    stats["tax_n_deep_joint"]   = deep_joint_n

    if deep_only_n > 0:
        acc_deep_only = deep_only_acc_num / deep_only_n
    else:
        acc_deep_only = float("nan")
    if deep_only_f1_cnt > 0:
        f1_deep_only = deep_only_f1_sum / deep_only_f1_cnt
    else:
        f1_deep_only = float("nan")

    stats["tax_acc_deep_only"] = acc_deep_only
    stats["tax_f1_deep_only"]  = f1_deep_only
    stats["tax_n_deep_only"]   = deep_only_n

    # ===== aggregate UNK metrics into stats =====
    for ch in ranks:
        # ---- ALL ----
        sup_all = unk_sup_total_all[ch]
        tp_all  = unk_tp_total_all[ch]
        fp_all  = unk_fp_total_all[ch]
        fn_all  = unk_fn_total_all[ch]

        if sup_all > 0:
            recall_all = tp_all / sup_all
        else:
            recall_all = float("nan")

        denom_all = tp_all + fp_all
        if denom_all > 0:
            precision_all = tp_all / denom_all
        else:
            precision_all = float("nan")

        stats[f"unk_recall_{ch}"]    = recall_all      # for backward compatibility
        stats[f"unk_precision_{ch}"] = precision_all
        stats[f"unk_support_{ch}"]   = sup_all
        stats[f"unk_fp_{ch}"]        = fp_all
        stats[f"unk_fn_{ch}"]        = fn_all

        # ---- JOINT ----
        sup_j = unk_sup_total_joint[ch]
        tp_j  = unk_tp_total_joint[ch]
        fp_j  = unk_fp_total_joint[ch]
        fn_j  = unk_fn_total_joint[ch]

        if sup_j > 0:
            recall_j = tp_j / sup_j
        else:
            recall_j = float("nan")

        denom_j = tp_j + fp_j
        if denom_j > 0:
            precision_j = tp_j / denom_j
        else:
            precision_j = float("nan")

        stats[f"unk_recall_joint_{ch}"]    = recall_j
        stats[f"unk_precision_joint_{ch}"] = precision_j
        stats[f"unk_support_joint_{ch}"]   = sup_j
        stats[f"unk_fp_joint_{ch}"]        = fp_j
        stats[f"unk_fn_joint_{ch}"]        = fn_j

        # ---- TAX-ONLY ----
        sup_o = unk_sup_total_only[ch]
        tp_o  = unk_tp_total_only[ch]
        fp_o  = unk_fp_total_only[ch]
        fn_o  = unk_fn_total_only[ch]

        if sup_o > 0:
            recall_o = tp_o / sup_o
        else:
            recall_o = float("nan")

        denom_o = tp_o + fp_o
        if denom_o > 0:
            precision_o = tp_o / denom_o
        else:
            precision_o = float("nan")

        stats[f"unk_recall_only_{ch}"]    = recall_o
        stats[f"unk_precision_only_{ch}"] = precision_o
        stats[f"unk_support_only_{ch}"]   = sup_o
        stats[f"unk_fp_only_{ch}"]        = fp_o
        stats[f"unk_fn_only_{ch}"]        = fn_o

    # ---- summary prints ----
    print(
        f"[E{epoch:02d}] {split.upper()} "
        f"loss={stats['loss']:.4f}  "
        f"otu={stats['loss_otu']:.4f}  tax={stats['loss_tax']:.4f}  "
        f"tree={stats['loss_tree']:.4f}  "
        f"acc_tax_deep={stats['tax_acc_deep']:.3f}"
    )

    print(f"[{split.upper()}] per-rank taxonomy metrics:")
    for ch in ranks:
        print(
            f"  {ch}: "
            f"acc={stats[f'tax_acc_{ch}']:.3f} "
            f"F1={stats[f'tax_f1_{ch}']:.3f} "
            f"n={stats[f'tax_n_{ch}']}"
        )

    # UNK prediction metrics (overall only, to keep stdout readable)
    print(f"[{split.upper()}] UNK prediction metrics per rank (ALL masked taxa positions):")
    for ch in ranks:
        sup_all = stats[f"unk_support_{ch}"]
        if sup_all > 0:
            print(
                f"  {ch}: "
                f"recall={stats[f'unk_recall_{ch}']:.3f} "
                f"precision={stats[f'unk_precision_{ch}']:.3f} "
                f"support={sup_all} "
                f"fp={stats[f'unk_fp_{ch}']} "
                f"fn={stats[f'unk_fn_{ch}']}"
            )

    if logger:
        logger.log(
            split=f"{split}_epoch",
            step=global_step,
            epoch=epoch,
            **stats,
        )

    return stats, global_step
