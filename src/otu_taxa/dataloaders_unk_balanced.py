# src/dataloaders.py
import os, json, random
import torch
import numpy as np
from collections import defaultdict

from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable, Tuple
from torch.utils.data import Dataset

# NOTE:
# - Specials are NOT stored on disk.
# - We append specials at runtime:
#     OTU: pad = O, mask = O+1
#     TAX: pad = T, mask = T+1
# - Labels use -100 as ignore_index (standard in PyTorch CE loss)

IGNORE_INDEX = -100  # for labels only


# =========================
# Dataset: RAW ONLY 
# =========================
class OTUTaxaDataset(Dataset):
    """
    Loads:
      - otu_vocab.json
      - taxonomy_vocab.json
      - samples.jsonl with records: {"sample_id", "otus": [int], "taxa": [int|None]}

    Returns per-item (NO padding, NO masking, NO specials):
      {
        "sample_id": str,
        "otus": List[int],                 # raw indices into stored OTU vocab (0..O-1)
        "taxa": List[Optional[int]],       # raw indices into stored TAX vocab (0..T-1) or None
        "length": int
      }
    """
    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir
        with open(os.path.join(dataset_dir, "otu_vocab.json")) as f:
            self.otu_vocab = json.load(f)
        with open(os.path.join(dataset_dir, "taxonomy_vocab.json")) as f:
            self.tax_vocab = json.load(f)

        self.O = len(self.otu_vocab)  # real OTU vocab size (no specials)
        self.T = len(self.tax_vocab)  # real TAX vocab size (no specials)

        self.samples: List[Dict[str, Any]] = []
        with open(os.path.join(dataset_dir, "samples.jsonl"), "r") as f:
            for line in f:
                rec = json.loads(line)
                # keep taxa as list with possible None — collator will map to <unk_tax> at runtime
                self.samples.append({
                    "sample_id": rec["sample_id"],
                    "otus": rec["otus"],
                    "taxa": rec["taxa"],
                    "length": len(rec["otus"]),
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.samples[idx]
        # Return lists (not tensors); collator will tensorize + pad + mask
        return {
            "sample_id": rec["sample_id"],
            "otus": rec["otus"],
            "taxa": rec["taxa"],
            "length": rec["length"],
        }


# =========================
# Masking config + Collator(balanced over OTU and over taxonomy)
# =========================


@dataclass
class MaskingConfig:
    mlm_prob: float = 0.15
    prob_joint: float = 0.50
    prob_otu_only: float = 0.25
    prob_tax_only: float = 0.25
    keep_prob: float = 0.10
    random_prob: float = 0.10
    max_len: Optional[int] = None

    # NEW: choose one balancing strategy per batch
    balance_mode: str = "random"          # {"none", "otu", "family", "random"}
    balance_family_prob: float = 0.50   # only used if balance_mode="random"

def make_collator_balanced(dataset, cfg, tax2fam: Optional[List[int]] = None):
    """
    Collator with:
      1) Original stochastic masking (same as make_collator)
      2) Post-processing per batch: choose ONE balancing strategy:
         - OTU-balanced re-selection (uniform-ish quota over OTUs)
         - Family-balanced TAX-only re-selection (uniform-ish quota over families)

    tax2fam:
      list/array where taxid_to_family[t] = family_tax_id (int) or -1 if none.
      Required if cfg.balance_mode uses "family".
    """
    assert abs((cfg.prob_joint + cfg.prob_otu_only + cfg.prob_tax_only) - 1.0) < 1e-6, \
        "prob_joint + prob_otu_only + prob_tax_only must sum to 1"

    O, T = dataset.O, dataset.T
    pad_otu_id, mask_otu_id = O, O + 1
    pad_tax_id, mask_tax_id = T, T + 1

    otu_random_low, otu_random_high = 0, O - 1
    tax_random_low, tax_random_high = 0, T - 1

    if cfg.balance_mode in ("family", "random"):
        if tax2fam is None:
            raise ValueError("taxid_to_family is required for family balancing.")
        if len(tax2fam) < T:
            raise ValueError(f"taxid_to_family must have length >= T={T}, got {len(tax2fam)}")

    def _apply_masking_noise_to_otu(input_otus, b, i, true_otu):
        rr = random.random()
        if rr < cfg.keep_prob:
            # keep as is
            input_otus[b, i] = true_otu
        elif rr < cfg.keep_prob + cfg.random_prob:
            input_otus[b, i] = random.randint(otu_random_low, otu_random_high)
        else:
            input_otus[b, i] = mask_otu_id

    def _apply_masking_noise_to_tax(input_taxa, b, i, true_tax):
        rr = random.random()
        if rr < cfg.keep_prob:
            input_taxa[b, i] = true_tax
        elif rr < cfg.keep_prob + cfg.random_prob:
            input_taxa[b, i] = random.randint(tax_random_low, tax_random_high)
        else:
            input_taxa[b, i] = mask_tax_id

    def _otu_balancing(
        *,
        input_otus, labels_otu, attention_mask, true_otus
    ):
        """
        Balance OTU supervision positions in this batch.
        We only operate on positions where OTU is supervised: labels_otu != IGNORE_INDEX.
        We re-sample to make an approx-uniform quota across OTUs present in the batch.
        """
        masked = defaultdict(list)  # otu -> [(b,i)] currently supervised
        pool   = defaultdict(list)  # otu -> [(b,i)] all eligible positions (where OTU exists)
        K = 0

        B, L = input_otus.shape
        for b in range(B):
            for i in range(L):
                if attention_mask[b, i].item() == 0:
                    continue
                otu = int(true_otus[b, i].item())
                if otu == pad_otu_id:
                    continue

                pool[otu].append((b, i))
                if labels_otu[b, i].item() != IGNORE_INDEX:
                    masked[otu].append((b, i))
                    K += 1

        if K <= 0:
            return

        active = [o for o, lst in pool.items() if lst]
        U = len(active)
        if U <= 0:
            return

        cap = {o: len(pool[o]) for o in active}
        base = K // U
        quota = {o: min(base, cap[o]) for o in active}
        assigned = sum(quota.values())
        R = K - assigned

        order = sorted(active, key=lambda o: cap[o])
        j = 0
        while R > 0:
            o = order[j % U]
            if quota[o] < cap[o]:
                quota[o] += 1
                R -= 1
            j += 1
            if j > 10 * (U + K):
                break

        picks = set()
        for o in active:
            q = quota[o]
            if q <= 0:
                continue
            positions = pool[o]
            if q >= len(positions):
                picks.update(positions)
            else:
                picks.update(random.sample(positions, q))

        masked_set = {(b, i) for o in active for (b, i) in masked.get(o, [])}

        # un-supervise OTU at positions that were supervised but not picked
        to_unmask = masked_set - picks
        for (b, i) in to_unmask:
            labels_otu[b, i] = IGNORE_INDEX
            # restore true OTU token
            input_otus[b, i] = true_otus[b, i]

        # supervise OTU at positions that were picked but not previously supervised
        to_mask = picks - masked_set
        for (b, i) in to_mask:
            true_otu = int(true_otus[b, i].item())
            labels_otu[b, i] = true_otu
            _apply_masking_noise_to_otu(input_otus, b, i, true_otu)

    def _family_tax_only_balancing(
        *,
        input_taxa, labels_taxa, labels_otu, attention_mask, true_taxa
    ):
        """
        Balance TAX-only supervision positions grouped by FAMILY.
        Only considers TAX-only positions: labels_otu == IGNORE_INDEX.
        Family is computed from taxid_to_family[true_tax].
        Uniform-ish quota across families present in the batch.
        """
        masked = defaultdict(list)  # fam -> [(b,i)] currently supervised (tax-only)
        pool   = defaultdict(list)  # fam -> [(b,i)] all eligible tax-only positions
        K = 0

        B, L = input_taxa.shape
        for b in range(B):
            for i in range(L):
                if attention_mask[b, i].item() == 0:
                    continue
                # TAX-only means OTU is not supervised
                if labels_otu[b, i].item() != IGNORE_INDEX:
                    continue

                t = int(true_taxa[b, i].item())
                if t == pad_tax_id:
                    continue

                fam = int(tax2fam[t]) if (tax2fam is not None) else -1
                if fam < 0:
                    continue

                pool[fam].append((b, i))
                if labels_taxa[b, i].item() != IGNORE_INDEX:
                    masked[fam].append((b, i))
                    K += 1

        if K <= 0:
            return

        active = [f for f, lst in pool.items() if lst]
        U = len(active)
        if U <= 0:
            return

        cap = {f: len(pool[f]) for f in active}
        base = K // U
        quota = {f: min(base, cap[f]) for f in active}
        assigned = sum(quota.values())
        R = K - assigned

        order = sorted(active, key=lambda f: cap[f])
        j = 0
        while R > 0:
            f = order[j % U]
            if quota[f] < cap[f]:
                quota[f] += 1
                R -= 1
            j += 1
            if j > 10 * (U + K):
                break

        picks = set()
        for f in active:
            q = quota[f]
            if q <= 0:
                continue
            positions = pool[f]
            if q >= len(positions):
                picks.update(positions)
            else:
                picks.update(random.sample(positions, q))

        masked_set = {(b, i) for f in active for (b, i) in masked.get(f, [])}

        # un-supervise tax at positions that were supervised but not picked
        to_unmask = masked_set - picks
        for (b, i) in to_unmask:
            labels_taxa[b, i] = IGNORE_INDEX
            input_taxa[b, i] = true_taxa[b, i]  # restore true tax token

        # supervise tax at positions that were picked but not previously supervised
        to_mask = picks - masked_set
        for (b, i) in to_mask:
            true_tax = int(true_taxa[b, i].item())
            labels_taxa[b, i] = true_tax
            _apply_masking_noise_to_tax(input_taxa, b, i, true_tax)

    def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        lengths = [b["length"] for b in batch]
        L = cfg.max_len if cfg.max_len is not None else max(lengths)
        B = len(batch)

        input_otus = torch.full((B, L), pad_otu_id, dtype=torch.long)
        input_taxa = torch.full((B, L), pad_tax_id, dtype=torch.long)
        labels_otu = torch.full((B, L), IGNORE_INDEX, dtype=torch.long)
        labels_taxa = torch.full((B, L), IGNORE_INDEX, dtype=torch.long)
        attention_mask = torch.zeros((B, L), dtype=torch.long)

        sample_ids: List[str] = []
        for b, rec in enumerate(batch):
            sample_ids.append(rec["sample_id"])
            otus_raw = rec["otus"]
            taxa_raw = rec["taxa"]

            otus_p = otus_raw[:L] + [pad_otu_id] * max(0, L - len(otus_raw))
            taxa_p = taxa_raw[:L] + [pad_tax_id] * max(0, L - len(taxa_raw))

            input_otus[b] = torch.tensor(otus_p, dtype=torch.long)
            input_taxa[b] = torch.tensor(taxa_p, dtype=torch.long)
            attention_mask[b] = (input_otus[b] != pad_otu_id).long()

        # Keep true copies for balancing “apply/unmask” logic
        true_otus = input_otus.clone()
        true_taxa = input_taxa.clone()

        # ===== original stochastic masking =====
        for b in range(B):
            for i in range(L):
                if attention_mask[b, i].item() == 0:
                    continue
                if random.random() > cfg.mlm_prob:
                    continue

                r = random.random()
                if r < cfg.prob_joint:
                    do_otu, do_tax = True, True
                elif r < cfg.prob_joint + cfg.prob_otu_only:
                    do_otu, do_tax = True, False
                else:
                    do_otu, do_tax = False, True

                if do_otu:
                    true_otu = int(true_otus[b, i].item())
                    labels_otu[b, i] = true_otu
                    _apply_masking_noise_to_otu(input_otus, b, i, true_otu)

                if do_tax:
                    true_tax = int(true_taxa[b, i].item())
                    if true_tax != pad_tax_id:
                        labels_taxa[b, i] = true_tax
                        _apply_masking_noise_to_tax(input_taxa, b, i, true_tax)

        # ===== choose ONE balancing strategy per batch =====
        mode = cfg.balance_mode.lower()
        if mode == "random":
            mode = "family" if (random.random() < cfg.balance_family_prob) else "otu"

        if mode == "otu":
            _otu_balancing(
                input_otus=input_otus,
                labels_otu=labels_otu,
                attention_mask=attention_mask,
                true_otus=true_otus,
            )
        elif mode == "family":
            _family_tax_only_balancing(
                input_taxa=input_taxa,
                labels_taxa=labels_taxa,
                labels_otu=labels_otu,
                attention_mask=attention_mask,
                true_taxa=true_taxa,
            )
        # else: "none" => do nothing

        return {
            "sample_id": sample_ids,
            "input_otus": input_otus,
            "input_taxa": input_taxa,
            "labels_otu": labels_otu,
            "labels_taxa": labels_taxa,
            "attention_mask": attention_mask,
            "lengths": torch.tensor(lengths),
            "special_ids": {
                "otu": {"pad": pad_otu_id, "mask": mask_otu_id},
                "tax": {"pad": pad_tax_id, "mask": mask_tax_id},
                "vocab_sizes": {"otu": O + 2, "tax": T + 2},
            },
        }

    return collate

## helpers

def build_tax2rank_from_vocab(tax_vocab_list):
    tax2rank = {}
    for tid, name in enumerate(tax_vocab_list):
        if isinstance(name, str) and ":" in name:
            tax2rank[tid] = name.split(":", 1)[0]
        else:
            tax2rank[tid] = None
    return tax2rank

def build_tax2ancestor_at_rank(
    M_np,
    tax_vocab_list,
    target_rank="f",
    missing_value=-1,
):
    """
    Build a dense mapping from each tax_id to its ancestor at target_rank.

    Returns:
        taxid_to_ancestor: List[int]
            taxid_to_ancestor[t] = ancestor_id at target_rank,
            or missing_value if none exists.
    """
    # tax_id -> rank letter (k,p,c,o,f,g,s)
    tax2rank = {}
    for tid, name in enumerate(tax_vocab_list):
        if isinstance(name, str) and ":" in name:
            tax2rank[tid] = name.split(":", 1)[0]
        else:
            tax2rank[tid] = None

    T = M_np.shape[0]
    assert M_np.shape[1] == T

    taxid_to_ancestor = [missing_value] * T

    for t in range(T):
        # ancestors of t are rows a where M[a, t] == 1
        ancestors = np.flatnonzero(M_np[:, t])
        for a in ancestors:
            if tax2rank.get(int(a)) == target_rank:
                taxid_to_ancestor[t] = int(a)
                break

    return taxid_to_ancestor
