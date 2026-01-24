# OTU–Taxa Foundation Model

This repository contains the codebase for developing a **foundation model over
microbiome OTUs and hierarchical taxonomy**, using large-scale datasets derived
from the Microbe Atlas.

The goal of this project is to learn joint representations of:
- OTU abundance patterns across samples, and
- hierarchical taxonomic structure (kingdom → species),

in a way that supports taxonomy-aware pretraining and downstream tasks.

---

## Data

The Microbe Atlas dataset (https://microbeatlas.org/about) was converted into a
model-ready format consisting of:
- OTU vocabularies,
- taxonomy vocabularies and tree artifacts,
- and per-sample OTU/taxonomy index representations.

The code used to construct these datasets and artifacts is documented in:

The code used to construct these datasets and artifacts is documented in
[notebooks/preprocess](notebooks/preprocess/).


The generated data files are large and are therefore stored outside this
repository.  
This repository contains the code needed to reproduce their construction and to
load them during model training. 

---

## Model overview

![OTU–Taxa foundation model architecture](figures/foundation_OTU_and_TAXA-Model_scheme.jpg)

**Figure 1.** Overview of the OTU–Taxa foundation model. The model jointly encodes
OTU abundance information and hierarchical taxonomy tokens using a shared
Transformer backbone, enabling taxonomy-aware pretraining and downstream tasks.

---

## Repository structure

This repository is organized to reflect the full lifecycle of the OTU–Taxa
foundation model, from data preprocessing and pretraining to evaluation and
analysis. The main components are:

### `configs/`
Configuration files for experiments and model variants.  
These files define model hyperparameters, training settings, and dataset
references, enabling reproducible runs across different experimental setups.

### `figures/`
Static figures used in the paper and documentation, including model diagrams,
evaluation plots, and schematic illustrations.

### `notebooks/`
Jupyter notebooks used for data construction, model training, and analysis.
This directory is organized into thematic subfolders:

- `preprocess/`  
  Notebooks for converting the Microbe Atlas data into model-ready formats.
  This includes OTU filtering, abundance-based tokenization, taxonomy
  construction, and generation of tree artifacts.

- `pretrain_basic_models/`  
  Notebooks for pretraining the OTU–Taxa model under different conditions,
  including standard pretraining, genus-corrupted taxonomy experiments, and
  K-shot split construction.

- `test/`  
  Notebooks used for evaluation and analysis. Subfolders correspond to
  specific evaluation regimes:
  - `genus_corrupted/`: experiments with incomplete taxonomy references.
  - `test_infrequent_otus/`: evaluation on low-frequency OTUs.
  - `test_k_shot/`: zero-shot and few-shot adaptation experiments.
  - `test_pred_taxa_model/`: taxonomy prediction analyses, including affected
    OTUs and random OTU evaluations.

These notebooks are primarily exploratory and are intended to document the
analysis performed for the paper.

### `scripts/`
Standalone scripts for running training, evaluation, or preprocessing steps
from the command line. These scripts provide a lightweight alternative to
notebooks for batch execution and cluster-based runs.

### `src/`
Core Python source code for the project. This directory contains the
implementation of:
- the OTU–Taxa Transformer model,
- hierarchical taxonomy heads and losses,
- data loaders and collators,
- training and evaluation utilities.

All reusable components are implemented here and imported by notebooks and
scripts.

### `runs_microbeatlas/`
Logs and outputs from pretraining runs on the full Microbe Atlas dataset.
This typically includes checkpoints, training curves, and evaluation summaries.
These files are not tracked by Git due to their size.

### `runs_microbeatlas_genus_corruption/`
Outputs from experiments using corrupted taxonomy references, where a subset
of species-level annotations is removed during training.

### `runs_microbeatlas_k_shot/`
Outputs from K-shot adaptation experiments, including zero-shot and few-shot
taxonomy prediction results for held-out OTUs.

### `tests/`
Unit and integration tests for selected components of the codebase. These
tests are intended to validate core functionality (e.g., losses, decoding
logic) as the code evolves.

---

## Relationship to the paper

This repository accompanies the paper:

**OTU–TAXA: Contextual and Hierarchy-Aware Taxonomy Prediction for Amplicon
Microbiome Data**

The notebooks and scripts reproduce the data processing, model pretraining,
and evaluation experiments described in the manuscript. Figure references,
evaluation regimes (random OTUs, affected OTUs, zero-shot and few-shot OTUs),
and hierarchical metrics implemented here directly correspond to the methods
and results sections of the paper.

---

## Data availability

Due to size constraints, the processed Microbe Atlas datasets and derived
artifacts are not included in this repository. The code provided here allows
full reconstruction of these artifacts given access to the original Microbe
Atlas data and the SILVA reference taxonomy.

Paths to external datasets are configured explicitly in the preprocessing and
training notebooks and scripts.

---

## Status

This repository reflects an active research codebase.  
Some notebooks contain exploratory analysis or intermediate results used during
paper development. The `src/` directory should be considered the stable core
implementation.
