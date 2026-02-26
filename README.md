# KPCA-DA-New

Kernel PCA-based Domain Adaptation (KPCA-DA) for gene expression data alignment.

## Overview

This library implements domain adaptation by aligning source and target domains in a kernel PCA space using Maximum Mean Discrepancy (MMD) loss with regularization. It is designed for cross-study integration of gene expression datasets.

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Functional API

```python
import numpy as np
from scripts.kpcada import load_expression_matrix, fit_transform

# Load source and target datasets
X_source, genes_source = load_expression_matrix("path/to/source.csv")
X_target, genes_target = load_expression_matrix("path/to/target.csv")

# Run domain adaptation
results = fit_transform(
    X_source,
    X_target,
    genes_source=genes_source,
    genes_target=genes_target,
    n_components=2,
    kernel="rbf",
    kpca_gamma=0.5,
    mmd_gamma=0.5,
    lambda_reg=0.1,
    lr=0.0001,
    epochs=100,
    verbose=True,
)

print(f"Common genes found: {results['common_genes_count']}")
print(f"Adapted target shape: {results['Z_t'].shape}")
```

### Class-based API

```python
from scripts.kpcada import KPCA_DA

model = KPCA_DA(
    n_components=2,
    kernel="rbf",
    kpca_gamma=0.5,
    mmd_gamma=0.5,
    lambda_reg=0.1,
    lr=0.0001,
    epochs=100,
    verbose=True,
)

Z_s, Z_t = model.fit_transform(X_source, X_target, genes_source, genes_target)
print(f"Transformed source shape: {Z_s.shape}")
print(f"Transformed target shape: {Z_t.shape}")
```

## Input Format

CSV files should be formatted with genes as columns and samples as rows (the script transposes on load). The last column is used as labels and is excluded from the feature matrix.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_components` | `2` | Number of KPCA components |
| `kernel` | `"rbf"` | Kernel type for KernelPCA |
| `kpca_gamma` | `None` | Gamma for KernelPCA kernel |
| `mmd_gamma` | `0.5` | Gamma for MMD RBF kernel |
| `lambda_reg` | `0.1` | Regularization strength |
| `lr` | `0.0001` | Learning rate |
| `epochs` | `100` | Number of optimization epochs |
| `verbose` | `False` | Print training progress |