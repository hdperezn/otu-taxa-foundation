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

    # balancing
    balance_mode: str = "random"          # {"none", "otu", "rank", "random"}
    balance_rank_prob: float = 0.3       # only if balance_mode="random"

    # rank balancing behavior
    rank_balance_policy: str = "all"   # {"single", "all"}  single-rank per batch OR all ranks per batch
    rank_choice: str = "random"           # {"random","k","p","c","o","f","g","s"} used if policy="single"
    ranks: tuple = ("k","p","c","o","f","g","s")

def make_collator_balanced_rank(
    dataset,
    cfg: MaskingConfig,
    tax2anc_by_rank: Optional[Dict[str, List[int]]] = None,  # rank -> [T] ancestor id
):
    assert abs((cfg.prob_joint + cfg.prob_otu_only + cfg.prob_tax_only) - 1.0) < 1e-6

    O, T = dataset.O, dataset.T
    pad_otu_id, mask_otu_id = O, O + 1
    pad_tax_id, mask_tax_id = T, T + 1

    otu_random_low, otu_random_high = 0, O - 1
    tax_random_low, tax_random_high = 0, T - 1

    mode0 = cfg.balance_mode.lower()
    if mode0 in ("rank", "random"):
        if tax2anc_by_rank is None:
            raise ValueError("tax2anc_by_rank is required for rank balancing.")
        for r in cfg.ranks:
            if r not in tax2anc_by_rank:
                raise ValueError(f"tax2anc_by_rank missing rank '{r}'")
            if len(tax2anc_by_rank[r]) < T:
                raise ValueError(f"tax2anc_by_rank['{r}'] must have length >= T={T}")

    def _apply_noise_otu(input_otus, b, i):
        rr = random.random()
        if rr < cfg.keep_prob:
            pass
        elif rr < cfg.keep_prob + cfg.random_prob:
            input_otus[b, i] = random.randint(otu_random_low, otu_random_high)
        else:
            input_otus[b, i] = mask_otu_id

    def _apply_noise_tax(input_taxa, b, i):
        rr = random.random()
        if rr < cfg.keep_prob:
            pass
        elif rr < cfg.keep_prob + cfg.random_prob:
            input_taxa[b, i] = random.randint(tax_random_low, tax_random_high)
        else:
            input_taxa[b, i] = mask_tax_id

    def _quota_resample(pool: Dict[int, List[tuple]], K: int) -> set:
        active = [k for k, lst in pool.items() if lst]
        U = len(active)
        if K <= 0 or U <= 0:
            return set()

        cap = {k: len(pool[k]) for k in active}
        base = K // U
        quota = {k: min(base, cap[k]) for k in active}
        assigned = sum(quota.values())
        R = K - assigned

        order = sorted(active, key=lambda k: cap[k])
        j = 0
        while R > 0:
            k = order[j % U]
            if quota[k] < cap[k]:
                quota[k] += 1
                R -= 1
            j += 1
            if j > 10 * (U + K):
                break

        picks = set()
        for k in active:
            q = quota[k]
            if q <= 0:
                continue
            positions = pool[k]
            if q >= len(positions):
                picks.update(positions)
            else:
                picks.update(random.sample(positions, q))
        return picks

    def _balance_otu_positions(*, input_otus, labels_otu, attention_mask, true_otus):
        pool = defaultdict(list)
        masked = defaultdict(list)
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

        picks = _quota_resample(pool, K)
        masked_set = {(b, i) for _, lst in masked.items() for (b, i) in lst}

        to_unmask = masked_set - picks
        for (b, i) in to_unmask:
            labels_otu[b, i] = IGNORE_INDEX
            input_otus[b, i] = true_otus[b, i]

        to_mask = picks - masked_set
        for (b, i) in to_mask:
            true_otu = int(true_otus[b, i].item())
            labels_otu[b, i] = true_otu
            input_otus[b, i] = true_otu
            _apply_noise_otu(input_otus, b, i)

    def _balance_tax_only_by_rank(*, rank: str, input_taxa, labels_taxa, labels_otu, attention_mask, true_taxa):
        """
        Balance TAX-only supervised positions across groups defined by ancestor at `rank`.
        - TAX-only positions: labels_otu == IGNORE_INDEX
        - Group id: anc = tax2anc_by_rank[rank][true_tax]
        - Preserve K: number of TAX-only positions currently supervised (labels_taxa != IGNORE_INDEX)
        """
        anc_map = tax2anc_by_rank[rank]
        pool = defaultdict(list)    # anc -> [(b,i)] eligible positions
        masked = defaultdict(list)  # anc -> [(b,i)] supervised positions
        K = 0

        B, L = input_taxa.shape
        for b in range(B):
            for i in range(L):
                if attention_mask[b, i].item() == 0:
                    continue
                if labels_otu[b, i].item() != IGNORE_INDEX:   # TAX-only filter
                    continue

                t = int(true_taxa[b, i].item())
                if t == pad_tax_id:
                    continue

                anc = int(anc_map[t])
                if anc < 0:
                    continue

                pool[anc].append((b, i))
                if labels_taxa[b, i].item() != IGNORE_INDEX:
                    masked[anc].append((b, i))
                    K += 1

        if K <= 0:
            return

        picks = _quota_resample(pool, K)
        masked_set = {(b, i) for _, lst in masked.items() for (b, i) in lst}

        # un-supervise not picked
        to_unmask = masked_set - picks
        for (b, i) in to_unmask:
            labels_taxa[b, i] = IGNORE_INDEX
            input_taxa[b, i] = true_taxa[b, i]

        # supervise newly picked
        to_mask = picks - masked_set
        for (b, i) in to_mask:
            true_tax = int(true_taxa[b, i].item())
            labels_taxa[b, i] = true_tax
            input_taxa[b, i] = true_tax
            _apply_noise_tax(input_taxa, b, i)

    def _choose_rank():
        if cfg.rank_choice != "random":
            return cfg.rank_choice
        return random.choice(list(cfg.ranks))

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

        true_otus = input_otus.clone()
        true_taxa = input_taxa.clone()

        # ===== ORIGINAL stochastic masking (unchanged) =====
        for b in range(B):
            for i in range(L):
                if attention_mask[b, i] == 0:
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
                    labels_otu[b, i] = int(true_otus[b, i].item())
                    _apply_noise_otu(input_otus, b, i)

                if do_tax:
                    true_tax = int(true_taxa[b, i].item())
                    if true_tax != pad_tax_id:
                        labels_taxa[b, i] = true_tax
                        _apply_noise_tax(input_taxa, b, i)

        # ===== choose ONE balancing strategy per batch =====
        mode = cfg.balance_mode.lower()
        if mode == "random":
            mode = "rank" if (random.random() < cfg.balance_rank_prob) else "otu"

        if mode == "otu":
            _balance_otu_positions(
                input_otus=input_otus,
                labels_otu=labels_otu,
                attention_mask=attention_mask,
                true_otus=true_otus,
            )
        elif mode == "rank":
            if cfg.rank_balance_policy == "single":
                rr = _choose_rank()
                _balance_tax_only_by_rank(
                    rank=rr,
                    input_taxa=input_taxa,
                    labels_taxa=labels_taxa,
                    labels_otu=labels_otu,
                    attention_mask=attention_mask,
                    true_taxa=true_taxa,
                )
            elif cfg.rank_balance_policy == "all":
                for rr in cfg.ranks:
                    _balance_tax_only_by_rank(
                        rank=rr,
                        input_taxa=input_taxa,
                        labels_taxa=labels_taxa,
                        labels_otu=labels_otu,
                        attention_mask=attention_mask,
                        true_taxa=true_taxa,
                    )
            else:
                raise ValueError(f"Unknown rank_balance_policy={cfg.rank_balance_policy!r}")

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

def build_tax2ancestor_at_ranks(M_np, tax_vocab_list, ranks=("k","p","c","o","f","g","s"), missing_value=-1):
    tax2rank = {}
    for tid, name in enumerate(tax_vocab_list):
        if isinstance(name, str) and ":" in name:
            tax2rank[tid] = name.split(":", 1)[0]
        else:
            tax2rank[tid] = None

    T = M_np.shape[0]
    out = {r: [missing_value] * T for r in ranks}

    # For each taxon t, find its ancestors (including itself if M[t,t]=1)
    for t in range(T):
        ancestors = np.flatnonzero(M_np[:, t])
        # We want, for each rank r, the unique ancestor at that rank (if exists)
        for a in ancestors:
            ra = tax2rank.get(int(a))
            if ra in out and out[ra][t] == missing_value:
                out[ra][t] = int(a)

    return out



## ---------- taxonomy collator: mask TAX only at affected OTU positions ----------

import torch

# ---------- taxonomy collator: mask TAX only at affected OTU positions ----------
def make_tax_only_mask_collator(dataset, affected_ids, *, T_base: int, max_len=None):
    """
    dataset.O is fine.
    dataset.T is T_real (not used for PAD/MASK anymore).
    T_base is the UNK-extended taxonomy size (real + 7 UNKs), no specials.
    """
    O = dataset.O
    pad_otu_id, mask_otu_id = O, O + 1

    pad_tax_id, mask_tax_id = T_base, T_base + 1  # IMPORTANT (matches model)

    affected_ids = set(map(int, affected_ids))

    def collate(batch):
        lengths = [len(b["otus"]) for b in batch]
        L = max_len if max_len is not None else max(lengths)
        B = len(batch)

        input_otus = torch.full((B, L), pad_otu_id, dtype=torch.long)
        input_taxa = torch.full((B, L), pad_tax_id, dtype=torch.long)
        attention_mask = torch.zeros((B, L), dtype=torch.bool)

        masked_positions = []
        sample_ids = []
        true_taxa = []

        for i, rec in enumerate(batch):
            otus = torch.tensor(rec["otus"], dtype=torch.long)
            taxa = torch.tensor(rec["taxa"], dtype=torch.long)

            L_i = min(L, len(otus))
            input_otus[i, :L_i] = otus[:L_i]
            input_taxa[i, :L_i] = taxa[:L_i]
            attention_mask[i, :L_i] = True

            # affected OTU positions
            aff_mask = torch.tensor(
                [int(int(x) in affected_ids) for x in otus[:L_i]],
                dtype=torch.bool
            )

            # MASK taxonomy at affected positions
            input_taxa[i, :L_i][aff_mask] = mask_tax_id

            pos_idx = torch.nonzero(aff_mask, as_tuple=False).view(-1).tolist()
            masked_positions.append(pos_idx)

            sample_ids.append(rec["sample_id"])
            true_taxa.append(rec["taxa"])  # keep original behavior (list/seq)

        return {
            "input_otus": input_otus,
            "input_taxa": input_taxa,
            "attention_mask": attention_mask,
            "masked_positions": masked_positions,
            "sample_ids": sample_ids,
            "true_taxa": true_taxa,
        }

    return collate
