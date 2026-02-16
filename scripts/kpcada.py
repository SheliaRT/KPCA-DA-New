import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
from typing import Any, Dict, List, Optional, Tuple, Union


DEFAULT_SOURCE_PATH = r"C:\Zmyfiles\codes\research\KPCA-DA\KPCA-DA\examples\data\TvsN\TCGA_BRCA_NvT.csv"
DEFAULT_TARGET_PATH = r"C:\Zmyfiles\codes\research\KPCA-DA\KPCA-DA\examples\data\TvsN\GSE45498_NvT.csv"


def load_expression_matrix(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load and reshape the expression matrix from disk, returning data and gene names."""
    expr = pd.read_csv(csv_path, encoding="utf-8", engine="python")
    expr = expr.transpose()
    expr.columns = expr.iloc[0]
    matrix = expr.iloc[1:, 0:-1].astype(float).values
    genes = expr.iloc[:, :-1].columns.to_numpy()
    return matrix.T, genes


def standardize_datasets(X_source: np.ndarray, X_target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Standardize source and target datasets independently."""
    scaler = StandardScaler()
    X_source_scaled = scaler.fit_transform(X_source)
    X_target_scaled = scaler.fit_transform(X_target)
    return X_source_scaled, X_target_scaled


def find_common_gene_indices(genes_source: np.ndarray, genes_target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Find common genes between source and target datasets and return their indices."""
    common_genes = []
    for gene_s in genes_source:
        for gene_t in genes_target:
            if gene_s == gene_t:
                common_genes.append(gene_s)
    
    condition_source = np.isin(genes_source, common_genes)
    index_source = np.where(condition_source)[0]
    
    condition_target = np.isin(genes_target, common_genes)
    index_target = np.where(condition_target)[0]
    
    return index_source, index_target


def rbf_kernel(x: torch.Tensor, y: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    pairwise_distances = torch.cdist(x, y) ** 2
    return torch.exp(-gamma * pairwise_distances)


def mmd_loss(Zs_common: torch.Tensor, Zt_common: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """Modified MMD loss using only K_tt and K_st terms."""
    K_tt = rbf_kernel(Zt_common, Zt_common, gamma)
    K_st = rbf_kernel(Zs_common, Zt_common, gamma)
    
    # Modified MMD: abs(K_tt.mean() - 2 * K_st.mean())
    mmd = torch.abs(K_tt.mean() - 2 * K_st.mean())
    return mmd


def regularization_term(Z_t: torch.Tensor, X_t: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    K_t = rbf_kernel(X_t.T, X_t.T, gamma)
    A = Z_t.T @ K_t @ Z_t
    return torch.trace(A)


def fit_transform(
    X_source: np.ndarray,
    X_target: np.ndarray,
    genes_source: Optional[np.ndarray] = None,
    genes_target: Optional[np.ndarray] = None,
    *,
    n_components: int = 2,
    kernel: str = "rbf",
    kpca_gamma: Optional[float] = None,
    mmd_gamma: float = 0.5,
    lambda_reg: float = 0.1,
    lr: float = 0.0001,
    epochs: int = 100,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Align source and target domains using kernel PCA and MMD-based adaptation.
    
    Uses gene-based matching to find common features between datasets.
    """
    # Find common gene indices
    if genes_source is None or genes_target is None:
        raise ValueError("genes_source and genes_target must be provided for gene-based matching")
    
    common_source_idx, common_target_idx = find_common_gene_indices(genes_source, genes_target)

    if common_source_idx.size == 0 or common_target_idx.size == 0:
        raise ValueError("No common genes found between the source and target datasets.")

    # Note: Standardization is optional - commented out in modified version
    # X_source_scaled, X_target_scaled = standardize_datasets(X_source, X_target)
    # Using data as-is (already scaled/normalized externally if needed)
    X_source_scaled = X_source
    X_target_scaled = X_target

    # Apply KPCA
    kpca = KernelPCA(n_components=n_components, kernel=kernel, gamma=kpca_gamma)
    Z_s_np = kpca.fit_transform(X_source_scaled)
    Z_t_np = kpca.fit_transform(X_target_scaled)

    # Convert to tensors
    Z_s = torch.tensor(Z_s_np, dtype=torch.float32)
    Z_t = torch.tensor(Z_t_np, dtype=torch.float32, requires_grad=True)
    X_t_tensor = torch.tensor(X_target_scaled.T, dtype=torch.float32)

    optimizer = optim.SGD([Z_t], lr=lr)
    loss_history: List[float] = []

    for epoch in range(epochs):
        # Detach and re-enable gradients (as in modified version)
        Z_t = Z_t.detach().requires_grad_(True)
        optimizer = optim.SGD([Z_t], lr=lr)  # Reinitialize optimizer with new Z_t
        optimizer.zero_grad()

        # Extract common points
        Zs_common = Z_s[common_source_idx]
        Zt_common = Z_t[common_target_idx]

        # Compute losses
        mmd = mmd_loss(Zs_common, Zt_common, gamma=mmd_gamma)
        reg = regularization_term(Z_t, X_t_tensor, gamma=mmd_gamma)
        loss = mmd + lambda_reg * reg

        # Backpropagate
        loss.backward(retain_graph=True)
        optimizer.step()

        loss_history.append(loss.item())

        if verbose and epoch % max(1, epochs // 10) == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}, MMD: {mmd.item():.4f}, Reg: {reg.item():.4f}")

    return {
        "Z_s": Z_s.detach().numpy(),
        "Z_t": Z_t.detach().numpy(),
        "common_source_idx": common_source_idx,
        "common_target_idx": common_target_idx,
        "loss_history": loss_history,
        "kpca_model": kpca,
        "common_genes_count": len(common_source_idx),
    }


class KPCA_DA:
    """Kernel PCA-based Domain Adaptation using MMD and regularization.
    
    This class implements domain adaptation by aligning source and target domains
    in a kernel PCA space using Maximum Mean Discrepancy (MMD) loss with regularization.
    """
    
    def __init__(
        self,
        n_components: int = 2,
        kernel: str = "rbf",
        kpca_gamma: Optional[float] = None,
        mmd_gamma: float = 0.5,
        lambda_reg: float = 0.1,
        lr: float = 0.0001,
        epochs: int = 100,
        verbose: bool = False,
    ):
        """Initialize KPCA_DA model.
        
        Parameters
        ----------
        n_components : int, default=2
            Number of components for KernelPCA
        kernel : str, default='rbf'
            Kernel type for KernelPCA
        kpca_gamma : float or None, default=None
            Gamma parameter for KernelPCA kernel
        mmd_gamma : float, default=0.5
            Gamma parameter for MMD kernel
        lambda_reg : float, default=0.1
            Regularization strength
        lr : float, default=0.0001
            Learning rate for optimization
        epochs : int, default=100
            Number of optimization epochs
        verbose : bool, default=False
            Whether to print progress
        """
        self.n_components = n_components
        self.kernel = kernel
        self.kpca_gamma = kpca_gamma
        self.mmd_gamma = mmd_gamma
        self.lambda_reg = lambda_reg
        self.lr = lr
        self.epochs = epochs
        self.verbose = verbose
        
        self.kpca_model = None
        self.Z_s_ = None
        self.Z_t_ = None
        self.common_source_idx_ = None
        self.common_target_idx_ = None
        self.loss_history_ = None
    
    def fit_transform(
        self,
        X_source: np.ndarray,
        X_target: np.ndarray,
        genes_source: Optional[np.ndarray] = None,
        genes_target: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit the model and transform both source and target data.
        
        Parameters
        ----------
        X_source : np.ndarray
            Source domain data (genes x samples)
        X_target : np.ndarray
            Target domain data (genes x samples)
        genes_source : np.ndarray or None
            Gene names for source data
        genes_target : np.ndarray or None
            Gene names for target data
            
        Returns
        -------
        Z_s : np.ndarray
            Transformed source data
        Z_t : np.ndarray
            Adapted target data
        """
        results = fit_transform(
            X_source=X_source,
            X_target=X_target,
            genes_source=genes_source,
            genes_target=genes_target,
            n_components=self.n_components,
            kernel=self.kernel,
            kpca_gamma=self.kpca_gamma,
            mmd_gamma=self.mmd_gamma,
            lambda_reg=self.lambda_reg,
            lr=self.lr,
            epochs=self.epochs,
            verbose=self.verbose,
        )
        
        # Store results
        self.Z_s_ = results["Z_s"]
        self.Z_t_ = results["Z_t"]
        self.common_source_idx_ = results["common_source_idx"]
        self.common_target_idx_ = results["common_target_idx"]
        self.loss_history_ = results["loss_history"]
        self.kpca_model = results["kpca_model"]
        
        return self.Z_s_, self.Z_t_
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform new data using fitted KPCA model.
        
        Parameters
        ----------
        X : np.ndarray
            Data to transform
            
        Returns
        -------
        Z : np.ndarray
            Transformed data
        """
        if self.kpca_model is None:
            raise ValueError("Model has not been fitted yet. Call fit_transform first.")
        
        return self.kpca_model.transform(X)


def main() -> None:
    # Load data with gene names
    X_source, genes_source = load_expression_matrix(DEFAULT_SOURCE_PATH)
    X_target, genes_target = load_expression_matrix(DEFAULT_TARGET_PATH)

    # Using functional API
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

    print(f"\nCommon genes found: {results['common_genes_count']}")
    print("Optimized Z_t shape:", results["Z_t"].shape)
    
    # Using class-based API
    print("\n" + "="*50)
    print("Using KPCA_DA class API:")
    print("="*50)
    
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
    print(f"\nTransformed source shape: {Z_s.shape}")
    print(f"Transformed target shape: {Z_t.shape}")


if __name__ == "__main__":
    main()