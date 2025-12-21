import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

#######################################
## masking version of the loss (fast) # 
#######################################

def get_tax_path_for_label(label_id: int,
                           M_tensor: torch.Tensor,
                           rank_idx: torch.Tensor,
                           R: int = 7):
    """
    Return a dict: rank r -> tax_id at that rank along the path
    (using ancestors of `label_id`).
    """
    anc_idx = torch.nonzero(M_tensor[:, label_id], as_tuple=False).view(-1)  # all ancestors (incl. self)
    path = {}
    for r in range(R):
        candidates = anc_idx[rank_idx[anc_idx] == r]
        if len(candidates) > 0:
            # in a valid tree you should have exactly one per rank;
            # if more, we just take the first.
            path[r] = int(candidates[0])
    return path



def factorized_tax_loss_batch_masked(
    logits_tax_base: torch.Tensor,    # [B, L, T_base]
    labels_taxa: torch.Tensor,        # [B, L]
    attention_mask: torch.Tensor,     # [B, L]
    path_by_label: torch.Tensor,      # [T_base, R]
    M_tensor: torch.Tensor,           # [T_base, T_base] descendant-closure
    rank_idx: torch.Tensor,           # [T_base] 0..6
    cand_k: torch.Tensor,             # [n_k] kingdom ids
    g2l_k: torch.Tensor,              # [T_base] global→local idx in cand_k, -1 if not kingdom
    ignore_index: int,
    R: int = 7,
) -> torch.Tensor:
    device = logits_tax_base.device
    B, L, T_base = logits_tax_base.shape

    # 1) Flatten supervised positions
    valid_pos = (labels_taxa != ignore_index) & attention_mask.bool()
    if not valid_pos.any():
        return logits_tax_base.new_zeros(())

    logits_flat = logits_tax_base[valid_pos]      # [N, T_base]
    labels_flat = labels_taxa[valid_pos]          # [N]
    paths_flat  = path_by_label[labels_flat]      # [N, R]
    N = logits_flat.size(0)

    total_loss = logits_tax_base.new_zeros(())

    # --------------------
    # Rank 0 (kingdom)
    # --------------------
    true_k = paths_flat[:, 0]              # [N]
    valid0 = (true_k >= 0)
    if valid0.any() and cand_k.numel() > 0:
        logits_k = logits_flat[valid0][:, cand_k]    # [N0, n_k]
        true_k_local = g2l_k[true_k[valid0]]         # [N0]
        valid_idx0 = (true_k_local >= 0)
        if valid_idx0.any():
            logits_k = logits_k[valid_idx0]          # [N0', n_k]
            true_k_local = true_k_local[valid_idx0]  # [N0']
            log_probs_k = F.log_softmax(logits_k, dim=-1)
            row_idx = torch.arange(true_k_local.size(0), device=device)
            loss_k = -log_probs_k[row_idx, true_k_local]
            total_loss = total_loss + loss_k.sum()

    # --------------------
    # Ranks 1..R-1
    # --------------------
    M_bool = M_tensor.bool()
    # [R, T_base] rank masks over vocab
    rank_mask_stack = torch.stack(
        [(rank_idx == r) for r in range(R)],
        dim=0
    )  # [R, T_base]
    for r in range(1, R):
        parents_r = paths_flat[:, r-1]      # [N]
        childs_r  = paths_flat[:, r]        # [N]
        valid_r = (parents_r >= 0) & (childs_r >= 0)
        if not valid_r.any():
            continue

        parents_r = parents_r[valid_r]      # [N_r]
        childs_r  = childs_r[valid_r]       # [N_r]
        logits_r  = logits_flat[valid_r]    # [N_r, T_base]

        # descendants of each parent: [N_r, T_base]
        M_parents = M_bool[parents_r]

        # restrict to rank r
        mask_rank = rank_mask_stack[r].unsqueeze(0)  # [1, T_base]
        child_mask = M_parents & mask_rank           # [N_r, T_base]

        # 1) row has at least one candidate child
        row_has_child = child_mask.any(dim=1)
        if not row_has_child.any():
            continue

        child_mask = child_mask[row_has_child]   # [N_r', T_base]
        logits_r   = logits_r[row_has_child]     # [N_r', T_base]
        childs_r   = childs_r[row_has_child]     # [N_r']

        # 2) ensure TRUE child is among candidates
        row_idx_all = torch.arange(childs_r.size(0), device=device)
        true_in_mask = child_mask[row_idx_all, childs_r]   # [N_r'] bool
        if not true_in_mask.any():
            continue

        child_mask = child_mask[true_in_mask]    # [N_r'', T_base]
        logits_r   = logits_r[true_in_mask]      # [N_r'', T_base]
        childs_r   = childs_r[true_in_mask]      # [N_r'']

        if childs_r.numel() == 0:
            continue

        # ---- HERE IS THE FIX  of the overlaod in the -1e9 ----
        very_neg = torch.finfo(logits_r.dtype).min
        masked_logits = logits_r.masked_fill(~child_mask, very_neg)
        log_probs_r = F.log_softmax(masked_logits, dim=-1)  # [N_r'', T_base]

        row_idx = torch.arange(childs_r.size(0), device=device)
        loss_r = -log_probs_r[row_idx, childs_r]
        total_loss = total_loss + loss_r.sum()

    # Normalize per supervised position (same as before)
    denom = float(valid_pos.sum().item())
    if denom == 0:
        return logits_tax_base.new_zeros(())
    return total_loss / denom



# NOTE: we reuse original factorized_tax_loss_batch_masked as-is.
# Only the hierarchy precompute + loss factory are UNK-specific.
def precompute_hierarchy_structures_with_unk(
    M_tensor: torch.Tensor,   # [T_base, T_base] descendant-closure
    rank_idx: torch.Tensor,   # [T_base] 0..6 (k..s)
    tax_vocab: list,          # length T_base
    T_base: int,
    R: int = 7,
):
    """
    UNK-specific version.

    Precompute:
      - path_by_label[y, r]:
          * real ancestor id at rank r if it exists
          * otherwise UNK_r (one per rank)
      - children_by_parent_rank[(parent, r)]:
          candidate child ids at rank r
      - cand_k: all kingdom ids

    Assumes:
      - Exactly one UNK token per rank (k..s) in tax_vocab.
      - M_tensor already contains UNKs in the closure.
    """
    device_cpu = torch.device("cpu")
    M_cpu = M_tensor.to(device_cpu)
    rank_cpu = rank_idx.to(device_cpu)

    # ---------------------------------
    # 0) Find UNK ids per rank (0..6)
    # ---------------------------------
    RANK_CHAR_TO_IDX = {
        "k": 0,
        "p": 1,
        "c": 2,
        "o": 3,
        "f": 4,
        "g": 5,
        "s": 6,
    }
    unk_ids = torch.full((R,), -1, dtype=torch.long)

    for g_id, name in enumerate(tax_vocab):
        if "UNK" not in name:
            continue
        c = name[0].lower()
        if c not in RANK_CHAR_TO_IDX:
            continue
        r = RANK_CHAR_TO_IDX[c]
        if unk_ids[r] != -1:
            raise ValueError(
                f"Multiple UNK tokens detected for rank {r}: "
                f"already {unk_ids[r]}, new {g_id}"
            )
        unk_ids[r] = int(g_id)

    if (unk_ids < 0).any():
        missing = [int(r) for r in torch.nonzero(unk_ids < 0, as_tuple=False).view(-1)]
        raise ValueError(
            f"Missing UNK tokens for ranks: {missing}. "
            f"Check tax_vocab and naming convention."
        )

    print("UNK ids per rank:", unk_ids.tolist())
    unk_ids_cpu = unk_ids.to(device_cpu)

    # ---------------------------------
    # 1) path_by_label: ancestors by rank (same as before)
    # ---------------------------------
    path_by_label = torch.full((T_base, R), -1, dtype=torch.long)

    for y in range(T_base):
        # all ancestors (including itself) of y
        anc_idx = torch.nonzero(M_cpu[:, y], as_tuple=False).view(-1)
        if anc_idx.numel() == 0:
            continue
        for r in range(R):
            mask = (rank_cpu[anc_idx] == r)
            cand = anc_idx[mask]
            if cand.numel() > 0:
                # In a well-formed tree there should be 1 per rank; take the first
                path_by_label[y, r] = int(cand[0])

    # ---------------------------------
    # 1b) fill deeper unknown ranks with UNK_r
    # ---------------------------------
    for y in range(T_base):
        row = path_by_label[y]  # [R]
        known_r = torch.nonzero(row >= 0, as_tuple=False).view(-1)
        if known_r.numel() == 0:
            # Should not happen for a valid taxonomy; leave row as-is.
            continue

        # deepest known rank r*
        r_star = int(known_r.max().item())
        # For all deeper ranks r > r*, fill with UNK_r
        for r in range(r_star + 1, R):
            row[r] = int(unk_ids_cpu[r])

    # ---------------------------------
    # 2) children_by_parent_rank (unchanged)
    # ---------------------------------
    children_by_parent_rank = {}

    for parent_id in range(T_base):
        row = M_cpu[parent_id]                       # descendants of parent_id
        desc_idx = torch.nonzero(row, as_tuple=False).view(-1)
        if desc_idx.numel() == 0:
            continue
        for r in range(1, R):                       # child ranks (p..s)
            mask = (rank_cpu[desc_idx] == r)
            cand = desc_idx[mask]
            if cand.numel() > 0:
                children_by_parent_rank[(parent_id, r)] = cand.clone()

    # ---------------------------------
    # 3) all kingdoms (unchanged)
    # ---------------------------------
    cand_k = torch.nonzero(rank_cpu == 0, as_tuple=False).view(-1)

    return path_by_label, children_by_parent_rank, cand_k

def make_factorized_tax_loss_fn_fast_masked_with_unk(
    M_tensor: torch.Tensor,
    rank_idx: torch.Tensor,
    tax_vocab: list,
    T_base: int,
    R: int = 7,
):
    """
    UNK-specific loss factory.

    Same interface as original factory, but uses
    precompute_hierarchy_structures_with_unk, which:
      - detects UNK ids,
      - fills unknown ranks in path_by_label with UNK_r.
    """
    M_cpu = M_tensor.to("cpu")
    rank_cpu = rank_idx.to("cpu")

    # Note: only difference from the old version is the function we call here:
    path_by_label_cpu, _, cand_k_cpu = precompute_hierarchy_structures_with_unk(
        M_tensor=M_cpu,
        rank_idx=rank_cpu,
        tax_vocab=tax_vocab,
        T_base=T_base,
        R=R,
    )

    cache = {
        "device": None,
        "path_by_label": None,
        "M": None,
        "rank_idx": None,
        "cand_k": None,
        "g2l_k": None,
    }

    def loss_fn(
        logits_tax: torch.Tensor,      # [B, L, T_ext]
        labels_taxa: torch.Tensor,     # [B, L]
        attention_mask: torch.Tensor,  # [B, L]
        *,
        ignore_index: int,
        **kwargs,
    ) -> torch.Tensor:
        device = logits_tax.device
        logits_base = logits_tax[..., :T_base]

        if cache["device"] is not device:
            cache["device"] = device

            path_dev = path_by_label_cpu.to(device)
            M_dev = M_cpu.to(device)
            rank_dev = rank_cpu.to(device)
            cand_k_dev = cand_k_cpu.to(device)

            g2l_k = torch.full(
                (T_base,),
                -1,
                dtype=torch.long,
                device=device,
            )
            if cand_k_dev.numel() > 0:
                g2l_k[cand_k_dev] = torch.arange(
                    cand_k_dev.numel(), device=device, dtype=torch.long
                )

            cache["path_by_label"] = path_dev
            cache["M"] = M_dev
            cache["rank_idx"] = rank_dev
            cache["cand_k"] = cand_k_dev
            cache["g2l_k"] = g2l_k

        return factorized_tax_loss_batch_masked(
            logits_tax_base=logits_base,
            labels_taxa=labels_taxa,
            attention_mask=attention_mask,
            path_by_label=cache["path_by_label"],
            M_tensor=cache["M"],
            rank_idx=cache["rank_idx"],
            cand_k=cache["cand_k"],
            g2l_k=cache["g2l_k"],
            ignore_index=ignore_index,
            R=R,
        )

    return loss_fn



############################################################
## Hierarchical Taxonomy Metric: per level predictions    ##
#############################################################

def hierarchical_predict_full_path(
    logits_vec: torch.Tensor,
    M_tensor: torch.Tensor,
    rank_idx: torch.Tensor,
    R: int = 7,
):
    """
    Returns a dict with predicted nodes for ALL ranks:
        pred[r] = tax_id or None if prediction failed before that rank.
    """
    device = logits_vec.device
    M_dev = M_tensor.to(device)
    rank_idx_dev = rank_idx.to(device)

    path_pred = {}

    # rank 0: kingdom
    cand_k = torch.nonzero(rank_idx_dev == 0, as_tuple=False).view(-1)
    if cand_k.numel() == 0:
        return {r: None for r in range(R)}
    logits_k = logits_vec[cand_k]
    pred_k = int(cand_k[int(logits_k.argmax())])
    path_pred[0] = pred_k

    # ranks 1..R-1
    for r in range(1, R):
        parent_id = path_pred.get(r - 1, None)
        if parent_id is None:
            for rr in range(r, R):
                path_pred[rr] = None
            break

        row = M_dev[parent_id]
        desc_idx = torch.nonzero(row, as_tuple=False).view(-1)
        cand_mask = (rank_idx_dev[desc_idx] == r)
        cand_ids  = desc_idx[cand_mask]

        if cand_ids.numel() == 0:
            for rr in range(r, R):
                path_pred[rr] = None
            break

        logits_r = logits_vec[cand_ids]
        pred_r = int(cand_ids[int(logits_r.argmax())])
        path_pred[r] = pred_r

    # ensure all ranks 0..R-1 are present
    for r in range(R):
        if r not in path_pred:
            path_pred[r] = None

    return path_pred


@torch.no_grad()
def hierarchical_accuracy_f1_per_rank(
    logits_tax_full,
    labels_taxa,
    attention_mask,
    M_tensor,
    rank_idx,
    T_base,
    ignore_index,
    R=7,
):
    device = logits_tax_full.device
    logits_base = logits_tax_full[..., :T_base]
    M_dev = M_tensor.to(device)
    rank_idx_dev = rank_idx.to(device)

    valid = (labels_taxa != ignore_index) & attention_mask.bool()
    b_idx, i_idx = valid.nonzero(as_tuple=True)

    ranks = ['k','p','c','o','f','g','s']

    # True coverage
    n_true = [0]*R

    # For accuracy
    correct = [0]*R

    # For F1: per-class true counts, pred counts, tp
    from collections import defaultdict
    true_counts = [defaultdict(int) for _ in range(R)]
    pred_counts = [defaultdict(int) for _ in range(R)]
    tp_counts   = [defaultdict(int) for _ in range(R)]

    for b,i in zip(b_idx.tolist(), i_idx.tolist()):
        y = int(labels_taxa[b,i])
        if not (0 <= y < T_base): 
            continue

        path_true = get_tax_path_for_label(y, M_dev, rank_idx_dev, R=R)
        path_pred = hierarchical_predict_full_path(
            logits_base[b,i], M_dev, rank_idx_dev, R=R
        )

        for r in range(R):
            if r in path_true:
                t_id = path_true[r]
                n_true[r] += 1

                p_id = path_pred[r]

                # false if p_id is None
                if p_id is not None:
                    pred_counts[r][p_id] += 1

                true_counts[r][t_id] += 1

                if p_id == t_id:
                    correct[r] += 1
                    tp_counts[r][t_id] += 1

    # Compute accuracy & macro F1
    metrics = {}
    for r,ch in enumerate(ranks):
        # accuracy
        acc = correct[r] / n_true[r] if n_true[r] > 0 else float("nan")

        # macro F1
        f1_list = []
        for cls_id, total_true in true_counts[r].items():
            tp = tp_counts[r].get(cls_id, 0)
            total_pred = pred_counts[r].get(cls_id, 0)
            fp = total_pred - tp
            fn = total_true - tp
            denom = 2*tp + fp + fn
            f1 = 2*tp/denom if denom>0 else 0.0
            f1_list.append(f1)
        f1_macro = float(sum(f1_list)/len(f1_list)) if f1_list else float("nan")

        metrics[f"acc_{ch}"] = acc
        metrics[f"f1_{ch}"]  = f1_macro
        metrics[f"n_{ch}"]   = n_true[r]

    return metrics


############################################################
## Hierarchical Taxonomy Metric: deppest prediction known ##
############################################################
from collections import defaultdict
@torch.no_grad()
def deepest_taxonomy_accuracy_f1(
    logits_tax_full: torch.Tensor,   # [B, L, T_ext]
    labels_taxa: torch.Tensor,       # [B, L] deepest true labels (0..T_base-1)
    attention_mask: torch.Tensor,    # [B, L]
    M_tensor: torch.Tensor,          # [T_base, T_base] (for hierarchical decoding)
    rank_idx: torch.Tensor,          # [T_base], 0..6 for k..s, -1 otherwise
    T_base: int,
    ignore_index: int,
    R: int = 7,
):
    """
    Deepest-level metric:

      For each valid (b,i):
        - true deepest label = labels_taxa[b,i]
        - its rank r* = rank_idx[true_label]
        - hierarchical prediction path = hierarchical_predict_full_path(...)
        - predicted label at r* = path_pred[r*] (or None if decoding failed)
        - count correct/incorrect at this deepest rank.

    Returns:
      {
        "acc_deep": accuracy over deepest labels,
        "f1_deep":  macro F1 over deepest labels,
        "n_deep":   number of evaluated positions
      }
    """
    device = logits_tax_full.device
    logits_base = logits_tax_full[..., :T_base]  # [B, L, T_base]
    rank_idx_dev = rank_idx.to(device)
    M_dev = M_tensor.to(device)

    # valid positions
    valid = (labels_taxa != ignore_index) & attention_mask.bool()
    b_idx, i_idx = valid.nonzero(as_tuple=True)

    total = 0
    correct = 0

    true_counts = defaultdict(int)  # class_id -> total_true
    pred_counts = defaultdict(int)  # class_id -> total_pred
    tp_counts   = defaultdict(int)  # class_id -> tp

    for b, i in zip(b_idx.tolist(), i_idx.tolist()):
        y = int(labels_taxa[b, i])
        if not (0 <= y < T_base):
            continue

        # rank of this deepest label (k..s)
        r_star = int(rank_idx_dev[y].item())
        if not (0 <= r_star < R):
            continue

        # hierarchical prediction path
        path_pred = hierarchical_predict_full_path(
            logits_base[b, i],
            M_dev,
            rank_idx_dev,
            R=R,
        )

        p_id = path_pred.get(r_star, None)  # predicted at deepest rank

        total += 1
        true_counts[y] += 1
        if p_id is not None:
            pred_counts[p_id] += 1

        if p_id == y:
            correct += 1
            tp_counts[y] += 1

    if total == 0:
        return {
            "acc_deep": float("nan"),
            "f1_deep":  float("nan"),
            "n_deep":   0,
        }

    acc_deep = correct / total

    # macro F1 over deepest labels
    f1_list = []
    for cls_id, tot_true in true_counts.items():
        tp = tp_counts.get(cls_id, 0)
        total_pred = pred_counts.get(cls_id, 0)
        fp = total_pred - tp
        fn = tot_true - tp
        denom = 2 * tp + fp + fn
        if denom > 0:
            f1_c = 2.0 * tp / denom
            f1_list.append(f1_c)

    f1_deep = float(sum(f1_list) / len(f1_list)) if f1_list else float("nan")

    return {
        "acc_deep": acc_deep,
        "f1_deep":  f1_deep,
        "n_deep":   int(total),
    }

#############################################################
## UNK prediction metrics per rank                          ##
#############################################################   
@torch.no_grad()
def unk_prediction_metrics_per_rank(
    logits_tax_full: torch.Tensor,   # [B, L, T_ext]
    labels_taxa: torch.Tensor,       # [B, L] deepest known labels (0..T_base-1)
    attention_mask: torch.Tensor,    # [B, L]
    M_tensor: torch.Tensor,          # [T_base, T_base]
    rank_idx: torch.Tensor,          # [T_base], 0..6 for k..s, -1 otherwise
    T_base: int,
    unk_ids_by_rank: dict,           # {r: global_id_of_UNK_r}
    ignore_index: int,
    R: int = 7,
):
    """
    Measure how well the model predicts UNK at each rank, under the semantic:

      - labels_taxa[b,i] = deepest known node y
      - r_star = rank_idx[y] is the deepest known rank
      - For each rank r:
            truth_is_unk(r) = (r > r_star)
            pred_is_unk(r)  = (path_pred[r] == UNK_r)

    Returns a dict with:
        unk_recall_{k..s}
        unk_precision_{k..s}
        unk_support_{k..s}   (number of true UNK positions)
        unk_fp_{k..s}        (predicted UNK when truth is real)
        unk_fn_{k..s}        (truth UNK, predicted non-UNK)
        unk_tp_{k..s}        (true positives for UNK)
    """
    device = logits_tax_full.device
    M_dev = M_tensor.to(device)
    rank_idx_dev = rank_idx.to(device)

    ranks = ["k","p","c","o","f","g","s"]

    # counters
    tp_unk  = {ch: 0 for ch in ranks}
    fp_unk  = {ch: 0 for ch in ranks}
    fn_unk  = {ch: 0 for ch in ranks}
    sup_unk = {ch: 0 for ch in ranks}  # support (truth_is_unk)

    # valid positions (where we know the deepest label)
    valid = (labels_taxa != ignore_index) & attention_mask.bool()
    b_idx, i_idx = valid.nonzero(as_tuple=True)

    for b, i in zip(b_idx.tolist(), i_idx.tolist()):
        y = int(labels_taxa[b, i])
        if not (0 <= y < T_base):
            continue

        r_star = int(rank_idx_dev[y].item())
        if not (0 <= r_star < R):
            continue

        # predicted path (over full taxonomy incl. UNKs)
        logits_vec = logits_tax_full[b, i]  # [T_ext]
        path_pred = hierarchical_predict_full_path(
            logits_vec,
            M_dev,
            rank_idx_dev,
            R=R,
        )

        for r, ch in enumerate(ranks):
            # if we didn't define an UNK at this rank, skip
            if r not in unk_ids_by_rank:
                continue
            unk_id_r = unk_ids_by_rank[r]

            truth_is_unk = (r > r_star)
            pred_id = path_pred[r]
            pred_is_unk = (pred_id == unk_id_r)

            if truth_is_unk:
                sup_unk[ch] += 1
                if pred_is_unk:
                    tp_unk[ch] += 1
                else:
                    fn_unk[ch] += 1
            else:
                # truth is a real taxon at this rank
                if pred_is_unk:
                    fp_unk[ch] += 1

    # aggregate into metrics
    metrics = {}
    for r, ch in enumerate(ranks):
        sup = sup_unk[ch]
        tp  = tp_unk[ch]
        fp  = fp_unk[ch]
        fn  = fn_unk[ch]

        if sup > 0:
            recall = tp / sup
        else:
            recall = float("nan")

        denom = tp + fp
        if denom > 0:
            precision = tp / denom
        else:
            precision = float("nan")

        metrics[f"unk_recall_{ch}"]    = recall
        metrics[f"unk_precision_{ch}"] = precision
        metrics[f"unk_support_{ch}"]   = sup
        metrics[f"unk_fp_{ch}"]        = fp
        metrics[f"unk_fn_{ch}"]        = fn
        metrics[f"unk_tp_{ch}"]        = tp

    return metrics

# @torch.no_grad()
# def unk_prediction_metrics_per_rank(
#     model,
#     dataloader,
#     device,
#     M_tensor,
#     rank_idx,
#     T_base,
#     unk_ids_by_rank,     # dict {r: global_id_of_UNK_r}
#     IGNORE_INDEX: int,
#     R: int = 7,
# ):
#     """
#     Measure how well the model predicts UNK at each rank, under the semantic:

#       - labels_taxa[b,i] = deepest known node y
#       - r_star = rank_idx[y] is the deepest known rank
#       - For each rank r:
#             truth_is_unk(r) = (r > r_star)
#             pred_is_unk(r)  = (path_pred[r] == UNK_r)

#     Returns a dict with:
#         unk_recall_{k..s}
#         unk_precision_{k..s}
#         unk_support_{k..s}   (number of true UNK positions)
#         unk_fp_{k..s}        (predicted UNK when truth is real)
#         unk_fn_{k..s}        (truth UNK, predicted non-UNK)
#     """
#     model.eval()
#     amp_on = (device == "cuda")

#     M_dev = M_tensor.to(device)
#     rank_idx_dev = rank_idx.to(device)

#     ranks = ["k","p","c","o","f","g","s"]

#     # counters
#     tp_unk = {ch: 0 for ch in ranks}
#     fp_unk = {ch: 0 for ch in ranks}
#     fn_unk = {ch: 0 for ch in ranks}
#     sup_unk = {ch: 0 for ch in ranks}  # support

#     for batch in dataloader:
#         # move to device
#         for k in ("input_otus", "input_taxa", "attention_mask", "labels_otu", "labels_taxa"):
#             batch[k] = batch[k].to(device, non_blocking=True)

#         att = batch["attention_mask"]      # [B,L]
#         lbl = batch["labels_taxa"]         # [B,L]

#         valid = (lbl != IGNORE_INDEX) & att.bool()
#         if not valid.any():
#             continue

#         with autocast(enabled=amp_on):
#             out = model(
#                 input_otus=batch["input_otus"],
#                 input_taxa=batch["input_taxa"],
#                 attention_mask=batch["attention_mask"],
#                 labels_otu=batch["labels_otu"],
#                 labels_taxa=batch["labels_taxa"],
#             )

#         logits_full = out["logits_tax"].detach()  # [B,L,T_ext]

#         b_idx, i_idx = valid.nonzero(as_tuple=True)

#         for b, i in zip(b_idx.tolist(), i_idx.tolist()):
#             y = int(lbl[b, i])
#             if not (0 <= y < T_base):
#                 continue

#             r_star = int(rank_idx_dev[y].item())
#             if not (0 <= r_star < R):
#                 continue

#             # predicted path (over full taxonomy incl. UNKs)
#             logits_vec = logits_full[b, i]  # [T_ext]
#             path_pred = hierarchical_predict_full_path(
#                 logits_vec,
#                 M_dev,
#                 rank_idx_dev,
#                 R=R,
#             )

#             for r, ch in enumerate(ranks):
#                 # if we didn't define an UNK at this rank, skip
#                 if r not in unk_ids_by_rank:
#                     continue
#                 unk_id_r = unk_ids_by_rank[r]

#                 truth_is_unk = (r > r_star)
#                 pred_id = path_pred[r]
#                 pred_is_unk = (pred_id == unk_id_r)

#                 if truth_is_unk:
#                     sup_unk[ch] += 1
#                     if pred_is_unk:
#                         tp_unk[ch] += 1
#                     else:
#                         fn_unk[ch] += 1
#                 else:
#                     # truth is a real taxon at this rank
#                     if pred_is_unk:
#                         fp_unk[ch] += 1

#     # aggregate into metrics
#     metrics = {}
#     for r, ch in enumerate(ranks):
#         sup = sup_unk[ch]
#         tp = tp_unk[ch]
#         fp = fp_unk[ch]
#         fn = fn_unk[ch]

#         if sup > 0:
#             recall = tp / sup
#         else:
#             recall = float("nan")

#         denom = tp + fp
#         if denom > 0:
#             precision = tp / denom
#         else:
#             precision = float("nan")

#         metrics[f"unk_recall_{ch}"]    = recall
#         metrics[f"unk_precision_{ch}"] = precision
#         metrics[f"unk_support_{ch}"]   = sup
#         metrics[f"unk_fp_{ch}"]        = fp
#         metrics[f"unk_fn_{ch}"]        = fn

#     return metrics