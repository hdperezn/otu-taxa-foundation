
import os, json, re
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
from typing import List, Tuple, Optional, Dict, Set

def load_sintax_table(path: str) -> pd.DataFrame:
    """
    Load SINTAX output into a 2-column dataframe:
      otu_id | taxonomy

    Tries the (0,3) column layout first; falls back to parsing raw strings.
    """
    try:
        df = pd.read_csv(path, sep="\t", header=None, engine="python",
                         usecols=[0, 3], names=["otu_id", "taxonomy"], dtype=str)
    except Exception:
        df_raw = pd.read_csv(path, sep="\t", header=None, engine="python",
                             usecols=[0, 1], names=["otu_id", "raw_sintax"], dtype=str)

        def drop_conf(s: str) -> str:
            if pd.isna(s): 
                return ""
            parts = []
            for p in s.strip().rstrip(";").split(","):
                if ":" not in p:
                    continue
                parts.append(p.split("(", 1)[0].strip())
            return ",".join(parts)

        df_raw["taxonomy"] = df_raw["raw_sintax"].apply(drop_conf)
        df = df_raw[["otu_id", "taxonomy"]]

    df["otu_id"] = df["otu_id"].astype(str)
    df["taxonomy"] = df["taxonomy"].fillna("").astype(str)
    return df


# ============================================================
# Taxonomy parsing: enforce contiguous + valid prefix
# ============================================================

RANKS: List[str] = ["k", "p", "c", "o", "f", "g", "s"]
RANK_TO_IDX: Dict[str, int] = {r: i for i, r in enumerate(RANKS)}
_CONF_TAIL_RE = re.compile(r"\s*\([^)]*\)\s*$")

def strip_confidence(name: str) -> str:
    if not isinstance(name, str):
        return ""
    return _CONF_TAIL_RE.sub("", name).strip().rstrip(";").strip()

def split_tax_path(tax_str: str) -> List[str]:
    if not isinstance(tax_str, str) or not tax_str:
        return []
    s = tax_str.strip().rstrip(";").replace(";", ",")
    return [p.strip() for p in s.split(",") if p.strip()]

def parse_token(tok: str) -> Tuple[Optional[str], str]:
    if not isinstance(tok, str) or ":" not in tok:
        return None, ""
    r, name = tok.split(":", 1)
    r = (r or "").strip().lower()
    name = strip_confidence(name)
    if r not in RANKS:
        return None, name
    return r, name

def is_unidentified_name(name: str) -> bool:
    n = (name or "").strip().strip("'\"").lower()
    return n in {"unidentified", "unknown", "__unknown"}

def is_valid_token(tok: str) -> bool:
    r, name = parse_token(tok)
    if r is None:
        return False
    if name == "":
        return False
    if is_unidentified_name(name):
        return False
    return True

def pick_chain(tokens: List[str]) -> List[Optional[str]]:
    chain: List[Optional[str]] = []
    start = 0
    for r in RANKS:
        pref = r + ":"
        found = None
        for i in range(start, len(tokens)):
            t = tokens[i]
            if isinstance(t, str) and t.startswith(pref):
                found = (i, t)
                break
        if found is None:
            chain.append(None)
        else:
            chain.append(found[1])
            start = found[0] + 1
    return chain

def last_contiguous_valid_token(tokens: List[str]) -> Optional[str]:
    chain = pick_chain(tokens)
    last_valid = None
    for t in chain:
        if t is None or not is_valid_token(t):
            break
        last_valid = t
    return last_valid

def contiguous_chain(tokens: List[str]) -> List[str]:
    chain = pick_chain(tokens)
    out: List[str] = []
    for t in chain:
        if t is None or not is_valid_token(t):
            break
        out.append(t)
    return out

def token_depth(tok: str) -> Optional[int]:
    r, _ = parse_token(tok)
    return None if r is None else RANK_TO_IDX[r]

def ensure_child(mapping: dict, key: str) -> dict:
    if key not in mapping:
        mapping[key] = {}
    return mapping[key]

def load_affected_otu_ids_txt(path: str, otu_name2id: dict) -> Set[int]:
    """
    Load affected OTUs from a text file.
    Supports:
      - integer OTU ids (as strings)
      - OTU names like '90_1015;96_21955;97_71802'
    """
    ids = set()
    missing = []

    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue

            # Case 1: already an integer OTU id
            if s.isdigit():
                ids.add(int(s))
                continue

            # Case 2: OTU name → map to id
            if otu_name2id is not None and s in otu_name2id:
                ids.add(int(otu_name2id[s]))
            else:
                missing.append(s)

    if missing:
        print(
            f"[WARN] {len(missing)} affected OTUs could not be mapped to dataset OTU ids "
            f"(showing first 10): {missing[:10]}"
        )

    return ids
