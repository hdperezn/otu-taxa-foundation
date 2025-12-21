# src/otu_taxa_transformer.py
from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any, Literal, List
import inspect
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


IGNORE_INDEX = -100  # for CrossEntropyLoss


@dataclass
class ModelConfig:
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 4
    d_ff: int = 1024
    dropout: float = 0.1
    activation: Literal["relu", "gelu"] = "gelu"
    tie_otu_weights: bool = True
    otu_loss_weight: float = 1.0
    tax_loss_weight: float = 1.0
    emb_dropout: float = 0.1
    layernorm_emb: bool = True
    # tree regularization params
    lambda_tree: float = 0.0
    lca_csv: Optional[str] = None   # path to LCA csv file

    # This MUST be set in the UNK version when using tree regularization.
    T_real: Optional[int] = None

# =========================================== #
#  regularizing model by tree structure loss  #
# =========================================== #

def _loss_fn_name(fn):
    try:
        return fn.__name__
    except Exception:
        return str(fn)

def default_combine_loss_fn(loss_otu: torch.Tensor, loss_tax: torch.Tensor, cfg) -> torch.Tensor:
    w_otu = getattr(cfg, "otu_loss_weight", 1.0)
    w_tax = getattr(cfg, "tax_loss_weight", 1.0)
    return w_otu * loss_otu + w_tax * loss_tax

def default_otu_loss_fn(
    logits_otu: torch.Tensor,           # [B, L, O]
    labels_otu: torch.Tensor,           # [B, L]
    *,
    ignore_index: int,
    attention_mask: Optional[torch.Tensor] = None,  # not required, but allowed
    **kwargs,
) -> torch.Tensor:
    # zero-safe: if no supervised positions, return 0
    valid = (labels_otu != ignore_index)
    if attention_mask is not None:
        valid = valid & attention_mask.bool()
    if not valid.any():
        return logits_otu.new_zeros(())
    # gather only valid positions
    B, L, O = logits_otu.shape
    logits_flat = logits_otu.view(-1, O)
    labels_flat = labels_otu.view(-1)
    valid = valid.view(-1)
    return F.cross_entropy(logits_flat[valid], labels_flat[valid], reduction="mean")

def default_tax_loss_fn(
    logits_tax: torch.Tensor,           # [B, L, T]
    labels_taxa: torch.Tensor,          # [B, L]
    attention_mask: torch.Tensor,       # [B, L]
    *,
    ignore_index: int,
    **kwargs,
) -> torch.Tensor:
    # zero-safe: if no supervised positions, return 0
    valid = (labels_taxa != ignore_index) & attention_mask.bool()
    if not valid.any():
        return logits_tax.new_zeros(())
    B, L, T = logits_tax.shape
    logits_flat = logits_tax.view(-1, T)
    labels_flat = labels_taxa.view(-1)
    valid = valid.view(-1)
    return F.cross_entropy(logits_flat[valid], labels_flat[valid], reduction="mean")

def _pairwise_cosine_distance01(E: torch.Tensor) -> torch.Tensor:
    """
    E: [N, d] float32
    returns D in [0,1]: D = (1 - cos_sim)/2 or similar normalization.
    Implement consistent with your lca_distance scale.
    """
    # cosine similarity
    E = F.normalize(E, p=2, dim=-1)
    S = E @ E.t()                      # [-1,1]
    D = 0.5 * (1 - S)                  # [0,1]
    return D



# ==============================
#  New model with tree regularizer + UNK handling
# ==============================
class OTUTaxaTransformerEmbedTaxTreeUnkTaxa(nn.Module):
    """
    Same interface as your base model, but with optional global tree regularizer:
      - reads cfg.lambda_tree and cfg.lca_csv
      - keeps full LCA matrix on CPU (self.lca_cpu)
      - computes global [T x T] cosine-distance from taxonomy embeddings (excluding specials)
      - adds lambda_tree * loss_tree to total loss when enabled
    When tree loss is disabled (no lca_csv or lambda_tree<=0), behavior matches your old class.
    """

    def __init__(
        self,
        n_otus: int,
        n_taxa: int,
        pad_otu_id: int,
        pad_tax_id: int,
        config: Optional["ModelConfig"] = None,
        # injectable losses (same as your base model)
        otu_loss_fn: Optional[Callable[..., torch.Tensor]] = None,
        tax_loss_fn: Optional[Callable[..., torch.Tensor]] = None,
        combine_loss_fn: Optional[Callable[..., torch.Tensor]] = None,
        debug_losses: bool = True,
    ):
        super().__init__()
        self.cfg = config or ModelConfig()
        self.n_otus = n_otus 
        self.n_taxa = n_taxa # number of taxa including pad & mask
        self.pad_otu_id = pad_otu_id
        self.pad_tax_id = pad_tax_id
        self.T_real: Optional[int] = self.cfg.T_real # number of real taxa (no pad/mask/unk)


        # --- Embeddings
        self.otu_emb = nn.Embedding(n_otus, self.cfg.d_model, padding_idx=pad_otu_id)
        self.tax_emb = nn.Embedding(n_taxa, self.cfg.d_model, padding_idx=pad_tax_id)

        self.emb_ln = nn.LayerNorm(self.cfg.d_model) if getattr(self.cfg, "layernorm_emb", True) else nn.Identity()
        self.emb_drop = nn.Dropout(getattr(self.cfg, "emb_dropout", 0.1))

        # --- Encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.cfg.d_model,
            nhead=self.cfg.n_heads,
            dim_feedforward=self.cfg.d_ff,
            dropout=self.cfg.dropout,
            activation=self.cfg.activation,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=self.cfg.n_layers)

        # --- Heads
        self.otu_head = nn.Linear(self.cfg.d_model, n_otus, bias=False)
        if self.cfg.tie_otu_weights:
            self.otu_head.weight = self.otu_emb.weight

        # --- Loss wiring
        self.otu_loss_fn = otu_loss_fn or default_otu_loss_fn
        self.tax_loss_fn = tax_loss_fn or default_tax_loss_fn
        self.combine_loss_fn = combine_loss_fn or default_combine_loss_fn
        self.debug_losses = debug_losses
        self._printed_otu_loss_fn = False
        self._printed_tax_loss_fn = False
        self._printed_combine_loss_fn = False

        # --- LCA matrix (CPU only) ---
        self.lca_cpu: Optional[torch.Tensor] = None
        if getattr(self.cfg, "lca_csv", None):
            import pandas as pd
            df_D = pd.read_csv(self.cfg.lca_csv, index_col=0)
            self.lca_cpu = torch.tensor(df_D.values, dtype=torch.float32, device="cpu")

        # Caches (as non-persistent buffers so .to(device) doesn't try to move CPU LCA)
        self.register_buffer("_lca_dev_cache", None, persistent=False)
        self.register_buffer("_triu_mask", None, persistent=False)

        # If LCA is provided, assert shape matches T_real (NO UNK, NO PAD/MASK)
        if self.lca_cpu is not None:
            if self.T_real is None:
                raise ValueError(
                    "In the UNK-aware model, cfg.T_real must be set when lca_csv is provided "
                    "(T_real = number of original taxonomy nodes, without UNKs or specials)."
                )

            if self.lca_cpu.shape != (self.T_real, self.T_real):
                raise ValueError(
                    f"LCA matrix shape {tuple(self.lca_cpu.shape)} != (T_real, T_real) "
                    f"with T_real={self.T_real}. Rebuild LCA or set T_real correctly."
                )

            # additional sanity: T_real must be <= (n_taxa - 2) [exclude pad/mask]
            if self.T_real > (self.n_taxa - 2):
                raise ValueError(
                    f"T_real={self.T_real} is larger than n_taxa-2={self.n_taxa-2}. "
                    "Remember: n_taxa = T_base + 2, and T_base >= T_real."
                )

        # --- tree debug flags ---
        self._printed_tree_once = False
        self._tree_debug_calls = 0

    def _loss_fn_name(self, fn):
        try:
            return fn.__name__
        except Exception:
            return str(fn)

    def _debug_call_loss(self, which: str, fn, *args, **kwargs):
        flag = f"_printed_{which}"
        if self.debug_losses and not getattr(self, flag, False):
            print(f"[OTUTaxaTransformerEmbedTaxTree] using {which}={self._loss_fn_name(fn)} "
                  f"(id={id(fn)}) kwargs={sorted(list(kwargs.keys()))}", flush=True)
            setattr(self, flag, True)
        return fn(*args, **kwargs)

    def _tree_loss_global(self) -> torch.Tensor:
        """
        Global tree regularization (UNK-aware):
          - Use ONLY the first T_real taxonomy embeddings (original taxa).
          - Compute pairwise cosine distance D_emb [T_real, T_real].
          - Compare with LCA matrix [T_real, T_real] on same device.
          - Return standardized MSE over strict upper triangle.

        UNK tokens and specials (pad/mask) are NOT regularized.
        """
        if (self.lca_cpu is None) or (self.lca_cpu.numel() == 0):
            return self.tax_emb.weight.new_zeros(())

        if self.T_real is None:
            raise RuntimeError(
                "T_real is None inside _tree_loss_global. "
                "In the UNK-aware model, cfg.T_real must be set."
            )

        device = self.tax_emb.weight.device
        T_real = self.T_real

        # taxonomy embeddings for REAL taxa only
        E = self.tax_emb.weight[:T_real, :].float()   # [T_real, d], requires grad
        D_emb = _pairwise_cosine_distance01(E)       # [T_real, T_real]

        # LCA to device (cache)
        if (self._lca_dev_cache is None) or (self._lca_dev_cache.device != device):
            self._lca_dev_cache = self.lca_cpu.to(device, non_blocking=True)
        D_tree = self._lca_dev_cache                 # [T_real, T_real] on device

        # Upper-triangle mask (cache)
        if (self._triu_mask is None) or (self._triu_mask.device != device) or (self._triu_mask.shape[0] != T_real):
            self._triu_mask = torch.triu(
                torch.ones(T_real, T_real, dtype=torch.bool, device=device),
                diagonal=1,
            )

        A = D_emb[self._triu_mask]
        B = D_tree[self._triu_mask]

        # standardize each to unit variance
        A = (A - A.mean()) / (A.std() + 1e-6)
        B = (B - B.mean()) / (B.std() + 1e-6)

        diff = A - B
        return (diff * diff).mean()



    def forward(
        self,
        input_otus: torch.LongTensor,
        input_taxa: torch.LongTensor,
        attention_mask: torch.LongTensor,
        labels_otu: Optional[torch.LongTensor] = None,
        labels_taxa: Optional[torch.LongTensor] = None,
        **loss_kwargs,
    ) -> Dict[str, Any]:
        # --- embeddings + encoder ---
        x = self.otu_emb(input_otus) + self.tax_emb(input_taxa)
        x = self.emb_ln(x)
        x = self.emb_drop(x)

        key_padding_mask = (attention_mask == 0)
        h = self.encoder(x, src_key_padding_mask=key_padding_mask)

        logits_otu = self.otu_head(h)                  # [B, L, O]
        logits_tax = F.linear(h, self.tax_emb.weight)  # [B, L, T]

        out: Dict[str, Any] = {"hidden": h, "logits_otu": logits_otu, "logits_tax": logits_tax}

        if labels_otu is not None and labels_taxa is not None:
            # zero-safe head losses
            loss_otu = self._debug_call_loss(
                "otu_loss_fn", self.otu_loss_fn,
                logits_otu, labels_otu,
                ignore_index=IGNORE_INDEX,
                attention_mask=attention_mask,
                **loss_kwargs,
            )
            loss_tax = self._debug_call_loss(
                "tax_loss_fn", self.tax_loss_fn,
                logits_tax, labels_taxa, attention_mask,
                ignore_index=IGNORE_INDEX,
                **loss_kwargs,
            )
            loss = self._debug_call_loss(
                "combine_loss_fn", self.combine_loss_fn,
                loss_otu, loss_tax, self.cfg
            )

            # --- tree regularizer (global) ---
            if (getattr(self.cfg, "lambda_tree", 0.0) > 0.0) and (self.lca_cpu is not None):
                if not self._printed_tree_once:
                    dev = self.tax_emb.weight.device
                    print(
                        f"[TREE-REG] enabled: lambda_tree={float(self.cfg.lambda_tree)}  "
                        f"LCA_shape={tuple(self.lca_cpu.shape)}  "
                        f"T_real={self.T_real}  n_taxa={self.n_taxa}  device={dev}",
                        flush=True,
                    )
                    self._printed_tree_once = True


                loss_tree = self._tree_loss_global()
                self._tree_debug_calls += 1
                if self._tree_debug_calls <= 3:
                    lt = float(loss_tree.detach().item())
                    print(f"[TREE-REG] call#{self._tree_debug_calls}  loss_tree={lt:.8f}", flush=True)

                loss = loss + float(self.cfg.lambda_tree) * loss_tree
                out["loss_tree"] = loss_tree

            out.update({"loss": loss, "loss_otu": loss_otu, "loss_tax": loss_tax})

        return out

    @torch.no_grad()
    def pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor, mode: str = "mean"):
        if mode == "cls":
            return hidden[:, 0]
        mask = attention_mask.unsqueeze(-1)
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        return summed / counts
    


