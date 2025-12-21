
import torch
import numpy as np
import os, json, time, random
from functools import lru_cache


"""
Helpers for training and evaluating the pretraining model.
"""

IGNORE_INDEX = -100  # for CrossEntropyLoss

# ---------------------------
# Utils
# ---------------------------
def set_seed(s: int):
    random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

# simple line-based JSON logger
class MetricsLogger:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "a", buffering=1)  # line-buffered
    def log(self, **kwargs):
        rec = {"ts": time.time(), **kwargs}
        self.f.write(json.dumps(rec) + "\n")
    def close(self):
        try: self.f.close()
        except Exception: pass

@torch.no_grad()
def masked_top1_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute top-1 acc on positions where labels != IGNORE_INDEX."""
    mask = labels != IGNORE_INDEX
    if mask.sum() == 0:
        return 0.0
    preds = logits.argmax(dim=-1)
    correct = (preds[mask] == labels[mask]).sum().item()
    total = int(mask.sum().item())
    return correct / max(total, 1)

def save_checkpoint(path: str, model, optimizer, scheduler, scaler, epoch, step, best_val):
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "step": step,
        "best_val": best_val,
    }
    torch.save(ckpt, path)

def add_counts(agg: dict, inc: dict):
    for k, (c, n) in inc.items():
        C, N = agg.get(k, (0, 0))
        agg[k] = (C + c, N + n)

def ratio(pair):
    c, n = pair
    return (c / n) if n > 0 else float("nan")

RANK_LETTERS = ("k","p","c","o","f","g","s")

def per_rank_logs(seg_counts, prefix="tax_only_"):
    """Return two dicts:
       acc:  {tax_only_k: 0.83, ...}
       nums: {tax_only_k_n: 24, ...}  (#positions used for that accuracy)
       (Optionally you can also return 'correct' counts if you need later.)
    """
    acc = {f"{prefix}{ch}": ratio(seg_counts.get(f"{prefix}{ch}", (0,0)))
           for ch in RANK_LETTERS}
    nums = {f"{prefix}{ch}_n": seg_counts.get(f"{prefix}{ch}", (0,0))[1]
            for ch in RANK_LETTERS}
    return acc, nums


# --- cached helpers to read taxonomy vocab and build rank-index mapping ---

@lru_cache(maxsize=4)
def _load_tax_id2name(vocab_path: str):
    """
    Returns a tuple id2name where id2name[id] -> token string.
    Supports either:
      - list: ["k:Bacteria", "p:Firmicutes", ...]
      - dict: {"k:Bacteria": 0, "p:Firmicutes": 1, ...}
    """
    with open(vocab_path, "r") as f:
        data = json.load(f)
    if isinstance(data, list):
        id2name = list(data)
    elif isinstance(data, dict):
        max_id = max(data.values())
        id2name = [None] * (max_id + 1)
        for name, idx in data.items():
            id2name[idx] = name
    else:
        raise ValueError("Unsupported taxonomy_vocab.json format.")
    return tuple(id2name)

@lru_cache(maxsize=8)
def _rank_idx_array(vocab_path: str, n_taxa_vocab: int):
    """
    Build numpy array rank_idx[id] ∈ {0..6, -1}, mapping each taxonomy token id
    to a rank bucket: k,p,c,o,f,g,s -> 0..6. Others -> -1.
    Length is n_taxa_vocab (so pad/mask slots at the end will be -1).
    """
    id2name = _load_tax_id2name(vocab_path)
    letters = ('k','p','c','o','f','g','s')
    l2i = {ch:i for i,ch in enumerate(letters)}

    arr = np.full((n_taxa_vocab,), -1, dtype=np.int64)
    base_len = min(len(id2name), n_taxa_vocab)
    for i in range(base_len):
        name = id2name[i]
        if isinstance(name, str) and len(name) > 0:
            rid = l2i.get(name[0].lower(), -1)
            arr[i] = rid
    return arr, letters

# --- UPDATED metric with per-rank split driven by taxonomy_vocab.json ---

@torch.no_grad()
def segmented_accuracy_top1(
    logits_otu, logits_tax, labels_otu, labels_taxa,
    input_otus, input_taxa,
    tax_vocab_path: str = None,   # <--- NEW (optional). If given, adds per-rank outputs.
):
    """
    Returns dict of (correct, total) counts for:
      - 'otu_masked', 'tax_masked'
      - 'otu_joint',  'tax_joint'
      - 'otu_only',   'tax_only'

    If tax_vocab_path is provided, also returns per-rank splits for TAX-only:
      - 'tax_only_k', 'tax_only_p', ..., 'tax_only_s'
      (split by the TRUE label's rank letter)
    """
    sel_otu = labels_otu != IGNORE_INDEX
    sel_tax = labels_taxa != IGNORE_INDEX

    joint     = sel_otu & sel_tax
    otu_only  = sel_otu & (~sel_tax)   # OTU selected, TAX visible
    tax_only  = sel_tax & (~sel_otu)   # TAX selected, OTU visible

    def count_top1(logits, targets):
        if targets.numel() == 0:
            return (0, 0)
        preds = logits.argmax(dim=-1)
        correct = (preds == targets).sum().item()
        total   = int(targets.numel())
        return (correct, total)

    out = {}
    # overall per-head masked
    out["otu_masked"] = count_top1(logits_otu[sel_otu], labels_otu[sel_otu])
    out["tax_masked"] = count_top1(logits_tax[sel_tax], labels_taxa[sel_tax])

    # joint (both selected)
    out["otu_joint"]  = count_top1(logits_otu[joint],   labels_otu[joint])
    out["tax_joint"]  = count_top1(logits_tax[joint],   labels_taxa[joint])

    # ONLY-selected (your KPIs)
    out["otu_only"]   = count_top1(logits_otu[otu_only], labels_otu[otu_only])
    out["tax_only"]   = count_top1(logits_tax[tax_only], labels_taxa[tax_only])

    # -------- PER-RANK split for TAX-only (driven by taxonomy_vocab.json) --------
    if tax_vocab_path is not None:
        V_tax = logits_tax.size(-1)
        rank_idx_np, letters = _rank_idx_array(tax_vocab_path, V_tax)
        # move to the same device as labels for masking/indexing
        rank_idx = torch.as_tensor(rank_idx_np, dtype=torch.long, device=labels_taxa.device)

        labels_tax_sel = labels_taxa[tax_only]        # [N]
        logits_tax_sel = logits_tax[tax_only]         # [N, V_tax]
        if labels_tax_sel.numel() == 0:
            for ch in letters:
                out[f"tax_only_{ch}"] = (0, 0)
        else:
            # group by TRUE label's rank letter
            ranks_sel = rank_idx[labels_tax_sel]      # [N] values in 0..6
            preds_sel = logits_tax_sel.argmax(dim=-1) # [N]
            for rid, ch in enumerate(letters):
                m = (ranks_sel == rid)
                n = int(m.sum().item())
                if n == 0:
                    out[f"tax_only_{ch}"] = (0, 0)
                else:
                    c = (preds_sel[m] == labels_tax_sel[m]).sum().item()
                    out[f"tax_only_{ch}"] = (c, n)

    return out

USE_EMBED_TAX = True

@torch.no_grad()
def per_rank_top1_accuracy(
    logits_tax_full: torch.Tensor,      # [B, L, T_ext]
    labels_taxa_ranks: torch.Tensor,    # [B, L, 7] (global ids incl rank-UNKs)
    attention_mask: torch.Tensor,       # [B, L]
    supervision_mask: torch.Tensor,     # [B, L] e.g., (labels_taxa != IGNORE_INDEX)
    classes_by_rank: list[torch.Tensor],# list[7] of 1D long tensors (global ids for each rank incl UNK_r)
    known_only: bool = True,            # exclude UNK targets from accuracy
):
    valid = attention_mask.bool() & supervision_mask.bool()
    accs = []
    for r in range(7):
        cls = classes_by_rank[r].to(logits_tax_full.device)        # global ids for this rank (incl UNK_r)
        logits_r = torch.index_select(logits_tax_full, -1, cls)    # [B, L, V_r]
        preds_r  = logits_r.argmax(dim=-1)                         # [B, L] in local space [0..V_r-1]

        # map global targets to local space by index in cls
        gid_to_local = {int(g.item()): i for i, g in enumerate(cls)}
        targets_g = labels_taxa_ranks[..., r]                      # [B, L] global ids (may be UNK_r)
        targets_l = targets_g.clone()
        # vectorized map via cpu dict not ideal; do simple gather:
        # build tensor map for fast index
        max_gid = int(cls.max().item())
        map_tensor = torch.full((max_gid+1,), -1, device=cls.device, dtype=torch.long)
        map_tensor[cls] = torch.arange(cls.size(0), device=cls.device)
        targets_l = map_tensor[targets_g]                          # [B, L] local ids

        m = valid.clone()
        if known_only:
            unk_local = (cls.size(0) - 1)  # last entry in cls is UNK_r
            m = m & (targets_l != unk_local)

        if m.any():
            correct = (preds_r[m] == targets_l[m]).sum().item()
            total   = int(m.sum().item())
            accs.append(correct / max(total, 1))
        else:
            accs.append(float('nan'))
    return accs  # list of 7 floats (k..s)
