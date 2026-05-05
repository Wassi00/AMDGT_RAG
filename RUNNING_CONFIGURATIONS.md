# AMDGT Running Configurations

## Project Overview

AMDGT (Attention Aware Multi-Modal Learning Using Dual Graph Transformer) is a model for drug-disease association prediction. The project supports multiple running configurations with different datasets, retrieval modes, and hyperparameters.

## Requirements

- Python 3.9.13
- CUDA Toolkit 11.3.1
- PyTorch 1.10.0
- DGL 0.9.0
- NetworkX 2.8.4
- NumPy 1.23.1
- scikit-learn 0.24.2

## Basic Usage

Execute the training script with optional parameters:

```bash
python train_DDA.py [options]
```

## Main Running Configurations

### 1. Dataset Selection

Available datasets:

- `B-dataset` - Default benchmark dataset
- `C-dataset` - Default (used if not specified)
- `F-dataset` - Cold-start drug dataset

Each dataset includes:

- Drug embeddings (mol2vec, fingerprints, GIP similarity)
- Disease embeddings and similarity measurements (PS, GIP)
- Protein embeddings (ESM-2)
- Pre-existing association information (drug-disease, drug-protein, protein-disease)
- 10-fold cross-validation splits

Example:

```bash
python train_DDA.py --dataset B-dataset
python train_DDA.py --dataset F-dataset
```

### 2. Retrieval Modes

Three retrieval modes control how the model uses neighbor information:

#### Mode: `full`

Complete retrieval-augmented generation with attention mechanism. Uses retrieved neighbors for reasoning.

```bash
python train_DDA.py --retrieval_mode full --top_k 10
```

#### Mode: `baseline`

Baseline model without retrieval mechanism. No neighbor information used.

```bash
python train_DDA.py --retrieval_mode baseline
```

#### Mode: `retrieval`

Retrieval-only without full augmentation. Limited reasoning from retrieved neighbors.

```bash
python train_DDA.py --retrieval_mode retrieval --top_k 5
```

### 3. Top-K Retrieval Parameter

Controls number of retrieved neighbors per sample. Common values: 3, 5, 10

```bash
python train_DDA.py --retrieval_mode full --top_k 10
python train_DDA.py --retrieval_mode full --top_k 5
python train_DDA.py --retrieval_mode full --top_k 3
```

## Common Configuration Examples

### Configuration 1: Standard Full Retrieval on C-dataset

```bash
python train_DDA.py \
  --dataset C-dataset \
  --retrieval_mode full \
  --top_k 10 \
  --epochs 1000 \
  --lr 1e-4
```

### Configuration 2: Cold-start Evaluation on F-dataset

```bash
python train_DDA.py \
  --dataset F-dataset \
  --retrieval_mode full \
  --top_k 10 \
  --epochs 1000
```

### Configuration 3: Baseline Model (No Retrieval)

```bash
python train_DDA.py \
  --dataset C-dataset \
  --retrieval_mode baseline \
  --epochs 1000
```

### Configuration 4: Retrieval Ablation Study

```bash
python train_DDA.py \
  --dataset C-dataset \
  --retrieval_mode retrieval \
  --top_k 5 \
  --epochs 1000
```
