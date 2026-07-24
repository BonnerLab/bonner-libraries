import pickle
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Self

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

EPSILON = torch.finfo(torch.float32).eps


def _nndsvd_init(
    x: torch.Tensor,
    n_components: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """NNDSVD initialization for NMF (GPU-accelerated).

    Non-negative Double Singular Value Decomposition.
    Better initialization than random for faster convergence.

    Based on: Boutsidis & Gallopoulos, "SVD based initialization: A head start for NMF"

    Args:
        x: Input data of shape (n_samples, n_features), non-negative
        n_components: Number of components
        device: Device to use

    Returns:
        (W, H) initial matrices
    """
    # Compute truncated SVD
    u, s, vt = torch.linalg.svd(x, full_matrices=False)

    # Take only n_components
    u = u[:, :n_components]
    s = s[:n_components]
    vt = vt[:n_components, :]

    # Initialize W and H
    w = torch.zeros(x.shape[0], n_components, device=device)
    h = torch.zeros(n_components, x.shape[1], device=device)

    # First component: use abs of first singular vectors scaled by sqrt(s)
    w[:, 0] = torch.sqrt(s[0]) * torch.abs(u[:, 0])
    h[0, :] = torch.sqrt(s[0]) * torch.abs(vt[0, :])

    # Remaining components
    for j in range(1, n_components):
        uj = u[:, j]
        vj = vt[j, :]
        sj = s[j]

        # Split into positive and negative parts
        uj_pos = torch.clamp(uj, min=0)
        uj_neg = torch.clamp(-uj, min=0)
        vj_pos = torch.clamp(vj, min=0)
        vj_neg = torch.clamp(-vj, min=0)

        # Norms
        uj_pos_norm = torch.norm(uj_pos)
        uj_neg_norm = torch.norm(uj_neg)
        vj_pos_norm = torch.norm(vj_pos)
        vj_neg_norm = torch.norm(vj_neg)

        # Choose the combination that gives larger contribution
        mp = uj_pos_norm * vj_pos_norm
        mn = uj_neg_norm * vj_neg_norm

        if mp > mn:
            u_init = uj_pos / (uj_pos_norm + EPSILON)
            v_init = vj_pos / (vj_pos_norm + EPSILON)
            sigma = mp
        else:
            u_init = uj_neg / (uj_neg_norm + EPSILON)
            v_init = vj_neg / (vj_neg_norm + EPSILON)
            sigma = mn

        w[:, j] = torch.sqrt(sj * sigma) * u_init
        h[j, :] = torch.sqrt(sj * sigma) * v_init

    # Replace zeros with small values
    w = torch.clamp(w, min=EPSILON)
    h = torch.clamp(h, min=EPSILON)

    return w, h


def _hals_update_w(
    w: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """One HALS (block coordinate descent) sweep over the columns of W with H fixed.

    Solves min_{W>=0} ||X - W @ H||_F for W given the precomputed
    ``a = X @ H.T`` (n_samples, n_components) and ``b = H @ H.T`` (n_components,
    n_components). Each column is updated in closed form and projected onto the
    non-negative orthant; columns are updated Gauss-Seidel-style (later columns
    see the already-updated earlier ones), which is what gives HALS its
    fast convergence.
    """
    n_components = w.shape[1]
    for j in range(n_components):
        # Closed-form 1-D minimizer of the residual along component j, projected >= 0.
        wj = w[:, j] + (a[:, j] - w @ b[:, j]) / (b[j, j] + EPSILON)
        w[:, j] = torch.clamp(wj, min=EPSILON)
    return w


def _hals_update_h(
    h: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
) -> torch.Tensor:
    """One HALS sweep over the rows of H with W fixed.

    Solves min_{H>=0} ||X - W @ H||_F for H given the precomputed
    ``c = W.T @ X`` (n_components, n_features) and ``d = W.T @ W``
    (n_components, n_components). Row-wise Gauss-Seidel analog of
    :func:`_hals_update_w`.
    """
    n_components = h.shape[0]
    for j in range(n_components):
        hj = h[j, :] + (c[j, :] - d[j, :] @ h) / (d[j, j] + EPSILON)
        h[j, :] = torch.clamp(hj, min=EPSILON)
    return h


class NMF:
    """Non-negative Matrix Factorization with fit/transform interface.

    Factorizes X ≈ W @ H where W and H are non-negative.
    Uses HALS (Hierarchical Alternating Least Squares — block coordinate
    descent with per-component closed-form non-negative updates) for both fit
    and transform, which converges an order of magnitude faster than
    multiplicative updates on genuinely low-rank data while remaining fully
    GPU-vectorizable.

    Args:
        n_components: Number of components. If None, uses min(n_samples, n_features).
        init: Initialization method. 'random' (default) or 'nndsvd'.
        max_iter: Maximum number of iterations for fitting.
        tol: Tolerance for convergence (relative change in reconstruction error).
        transform_max_iter: Maximum iterations for transform step.
        seed: Optional seed for the random W/H initialization in `fit()` and
            for `transform()`'s `w_init=None` branch. When non-None, uses a
            per-call `torch.Generator` so global RNG state is untouched. Each
            `transform()` call reseeds independently, so output is fully
            determined by (seed, input shape, components_) regardless of prior
            transform history. Warm-started transforms (`w_init=` provided) are
            unaffected.
    """

    def __init__(
        self: Self,
        *,
        n_components: int | None = None,
        init: str = "random",
        max_iter: int = 200,
        tol: float = 1e-4,
        transform_max_iter: int = 200,
        seed: int | None = None,
    ) -> None:
        self.n_components = n_components
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.transform_max_iter = transform_max_iter
        self.seed = seed

        self.n_samples: int
        self.components_: torch.Tensor  # H matrix (n_components, n_features)
        self.device: torch.device

    def to(self: Self, device: torch.device | str) -> None:
        self.components_ = self.components_.to(device)
        self.device = torch.device(device)

    def fit(self: Self, x: torch.Tensor, /) -> None:
        """Fit NMF model to data X.

        Learns W and H such that X ≈ W @ H.
        Stores H as components_ for later transform operations.

        Args:
            x: Input data of shape (n_samples, n_features). Must be non-negative.
        """
        if x.ndim == 1:
            x = x.unsqueeze(dim=-1)

        self.n_samples, n_features = x.shape[-2], x.shape[-1]
        max_n_components = min(self.n_samples, n_features)

        if self.n_components is None:
            self.n_components = max_n_components
        elif self.n_components > max_n_components:
            error = f"n_components must be <= {max_n_components}"
            raise ValueError(error)

        self.device = x.device

        # Ensure non-negativity (shift if needed)
        x_min = x.min()
        if x_min < 0:
            x = x - x_min + EPSILON

        with torch.no_grad():
            # Initialize W and H
            if self.init == "nndsvd":
                w, h = _nndsvd_init(x, self.n_components, self.device)
            else:  # random
                gen = torch.Generator(device=self.device).manual_seed(self.seed) if self.seed is not None else None
                w = torch.abs(torch.randn(self.n_samples, self.n_components, device=self.device, generator=gen)) * 0.01 + EPSILON
                h = torch.abs(torch.randn(self.n_components, n_features, device=self.device, generator=gen)) * 0.01 + EPSILON

            # HALS (block coordinate descent) iterations
            prev_error = float('inf')
            for i in range(self.max_iter):
                # Update W with H fixed (one HALS sweep over W's columns).
                h_ht = h @ h.T
                x_ht = x @ h.T
                w = _hals_update_w(w, x_ht, h_ht)

                # Update H with W fixed (one HALS sweep over H's rows).
                w_tw = w.T @ w
                w_tx = w.T @ x
                h = _hals_update_h(h, w_tx, w_tw)

                # Check convergence periodically (every 10 iterations)
                if i % 10 == 0:
                    reconstruction = w @ h
                    error = torch.norm(x - reconstruction).item()
                    if abs(prev_error - error) / (prev_error + EPSILON) < self.tol:
                        break
                    prev_error = error

            # Sort components by explained variance (descending)
            # Variance explained by each component ≈ ||w_i||^2 * ||h_i||^2
            component_importance = (w ** 2).sum(dim=0) * (h ** 2).sum(dim=1)
            sort_idx = torch.argsort(component_importance, descending=True)
            h = h[sort_idx, :]

            self.components_ = h

    def transform(
        self: Self,
        z: torch.Tensor,
        /,
        *,
        components: Sequence[int] | int | None = None,
        w_init: torch.Tensor | None = None,
        use_tol: bool = False,
    ) -> torch.Tensor:
        """Transform data Z using the fitted components.

        Finds W_new such that Z ≈ W_new @ H (with H fixed from fit).
        Solves the convex non-negative least-squares problem with FISTA
        (accelerated projected gradient — one matmul per step, fully vectorized
        over all components), so it stays fast even at large ``n_components``
        where a per-column HALS sweep would be dominated by kernel-launch
        overhead. The solution is independent of the W init (the problem is
        convex); ``w_init`` only warm-starts to reduce iterations.

        After each call, the final W matrix is stored as self.last_transform_w_
        for use as a warm-start in the next call (useful when transforming
        sequentially-related data, e.g., consecutive training checkpoints).

        Args:
            z: Data to transform of shape (n_samples_new, n_features).
            components: Which components to return. If None, returns all.
            w_init: Optional warm-start initial W of shape (n_samples_new, n_components).
                If provided, starts optimization from w_init instead of random init,
                which can dramatically reduce iterations for similar inputs.
                Must match z's sample count. If None, uses random initialization.
            use_tol: If True, apply tolerance-based early stopping (same tol as fit).
                Reduces iterations when w_init is close to the optimum. Default False
                preserves exact same convergence behavior as the original implementation.

        Returns:
            Transformed data W_new of shape (n_samples_new, n_components).
            Final W is also stored in self.last_transform_w_ for warm-starting.
        """
        if components is None:
            components = self.n_components
        if isinstance(components, int):
            components = list(range(components))

        z = z.to(self.device)

        # Ensure non-negativity
        z_min = z.min()
        if z_min < 0:
            z = z - z_min + EPSILON

        n_samples_new = z.shape[0]
        h = self.components_

        with torch.no_grad():
            # Initialize W_new: warm-start if provided, else random
            if w_init is not None:
                w_new = w_init.clone().to(self.device).clamp_(min=EPSILON)
            else:
                # Reseed each call so transform output is fully determined by
                # (seed, shape, components_) — independent of prior transform history.
                gen = torch.Generator(device=self.device).manual_seed(self.seed) if self.seed is not None else None
                w_new = torch.abs(torch.randn(n_samples_new, self.n_components, device=self.device, generator=gen)) * 0.01 + EPSILON

            # Precompute the fixed-H NNLS normal-equation blocks
            h_ht = h @ h.T          # B = H H^T (n_components, n_components)
            z_ht = z @ h.T          # A = Z H^T (n_samples_new, n_components)

            # HALS block coordinate descent on W with H fixed — the same closed-form,
            # Gauss-Seidel per-column update fit() uses. Converges in a handful of sweeps even
            # when H H^T is ill-conditioned (notably n_components == n_features), where the
            # previous FISTA transform (step ∝ 1/‖B‖₂) crawled and left the reconstruction worse
            # than zeros at the default 200 iters (rel-recon 2.23 vs the ~0.02 optimum at K=512).
            # Convergence-checked like fit() so it stops early (~5 sweeps here); ``use_tol`` is
            # retained for API compatibility but the relative-error break always applies.
            w_new = w_new.clamp_(min=EPSILON)
            prev_error = float('inf')
            for i in range(self.transform_max_iter):
                w_new = _hals_update_w(w_new, z_ht, h_ht)
                if i % 5 == 0:
                    error = torch.norm(z - w_new @ h).item()
                    if abs(prev_error - error) / (prev_error + EPSILON) < self.tol:
                        break
                    prev_error = error

        # Store final W for warm-starting the next transform call
        self.last_transform_w_ = w_new
        return w_new[..., components]

    def inverse_transform(
        self: Self,
        w: torch.Tensor,
        /,
        *,
        components: Sequence[int] | int | None = None,
    ) -> torch.Tensor:
        """Reconstruct data from transformed representation.

        Args:
            w: Transformed data of shape (n_samples, n_components).
            components: Which components were used in transform.

        Returns:
            Reconstructed data of shape (n_samples, n_features).
        """
        if components is None:
            components = self.n_components
        if isinstance(components, int):
            components = list(range(components))

        w = w.to(self.device)
        return w @ self.components_[components, :]


class AsgMuNmf(Dataset):
    """Asymmetric gradient multiplicative update nonnegative-matrix factorization."""

    def __init__(self: Self, *, data: np.ndarray, n_components: int) -> None:
        self.data = data
        self.n_samples = data.shape[0]
        self.n_dimensions = data.shape[1]
        self.n_components = n_components

    def __len__(self: Self) -> int:
        return self.n_samples

    def __getitem__(self: Self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return idx

    def collate_samples(self: Self, indices):
        indices = np.array(indices, dtype=np.int64)
        rows = torch.from_numpy(self.data[indices, :].toarray())
        indices = torch.from_numpy(indices).long()
        return rows, indices

    def fit_transform(
        self: Self,
        u: torch.Tensor | None = None,
        v: torch.Tensor | None = None,
        n_epochs: int = 200,
        batch_size: int = 10000,
        num_workers: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)

        if u is None:
            u = torch.normal(0, 0.01, size=(self.n_samples, self.n_components)).abs()
        if v is None:
            v = torch.normal(0, 0.01, size=(self.n_components, self.n_dimensions)).abs()

        with Path("nmf_initialization.pkl").open("wb") as f:
            pickle.dump((u.cpu().numpy(), v.cpu().numpy()), f)

        u, v = u.to(device), v.to(device)
        _run_asg_mu_nmf(
            self,
            u=u,
            v=v,
            device=device,
            n_epochs=n_epochs,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=self.collate_samples,
        )
        u, v = u.cpu().numpy(), v.cpu().numpy()

        return u, v


def _run_asg_mu_nmf(
    x: Dataset,
    /,
    *,
    u: torch.Tensor,
    v: torch.Tensor,
    device: torch.device | str,
    n_epochs: int,
    batch_size: int,
    num_workers: int,
    collate_fn: Callable,
) -> None:
    with torch.no_grad():
        for epoch in range(n_epochs):
            batches = DataLoader(
                x,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=True,
                pin_memory=True,
                collate_fn=collate_fn,
            )

            for batch in batches:
                x_batch, indices = batch
                x_batch = x_batch.to(device)
                indices = indices.to(device)
                u_batch = u[indices, :]
                u[indices, :], v = _update_u_and_v(x_batch, u_batch, v)

            with Path(f"nmf_epoch_{epoch}.pkl").open("wb") as f:
                pickle.dump((u.cpu().numpy(), v.cpu().numpy()), f)

            if epoch > 0:
                Path(f"nmf_epoch_{epoch - 1}.pkl").unlink()


def _update_u_and_v(
    x: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    u_t = u.transpose(0, 1)
    v_t = v.transpose(0, 1)
    uv = torch.mm(u, v)
    u *= torch.mm(x, v_t) / (torch.mm(uv, v_t) + EPSILON)
    v *= torch.mm(u_t, x) / (torch.mm(u_t, uv) + EPSILON)
    return u, v
