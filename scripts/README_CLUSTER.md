cat > scripts/README_CLUSTER.md << 'EOF'
# Cluster usage notes (IBEX – KAUST)

This folder contains scripts intended to be executed on the IBEX cluster.  
This document describes how to connect to the cluster and the expected
directory structure for data and code.

---

## 1. Connecting to IBEX

From your local machine, connect using:

ssh -X pereznhd@glogin.ibex.kaust.edu.sa

Once connected, move to the project workspace:

cd /ibex/scratch/projects/c2014/hernan_perez/otu_taxa_project

---

## 2. Project layout on IBEX

The project is organized as follows:

```text
otu_taxa_project/
├── Microbeatlas_preprocess_training/
│   └── level_97/
│       └── silva-138.2/
│           └── incomplete_silva_sintax/
│               └── dataset_full_top999/
│                   ├── dataset/        # formatted samples
│                   ├── taxonomy/       # taxonomy files and vocab
│                   ├── artifacts/      # LCA matrices, ancestors, mappings
│                   └── logs/           # preprocessing logs
│
├── otu-taxa-foundation/                 # cloned GitHub repository (code only)
│   ├── src/
│   ├── scripts/
│   ├── configs/
│   └── notebooks/
│
└── runs/                                # training outputs (metrics, checkpoints)
```

### Important rules

- Data is never stored inside the git repository
- Training outputs under runs/ are never committed
- Code changes are committed from the local machine, not from IBEX

---

## 3. Code management on IBEX

The repository is cloned from GitHub:

cd otu-taxa-foundation
git pull


---

## 4. Data paths

Training scripts expect the same directory hierarchy used locally.  
Only the root path changes between local and cluster environments.

Example cluster root:

/ibex/scratch/projects/c2014/hernan_perez/otu_taxa_project

All scripts should reference data relative to this root (via configuration
files or environment variables), not hardcoded absolute paths.

---

## 5. Running jobs

- Use Slurm job scripts located in this folder
- Always start with a small sanity run before launching long jobs

### how to send a job



---

## 6. Purpose of this file

This file exists to:
- remind how to connect to IBEX
- document the on-disk data layout
- avoid accidental git or data management mistakes

If paths or structure change, update this file accordingly.
EOF
