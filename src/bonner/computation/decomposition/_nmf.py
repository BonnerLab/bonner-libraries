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
    """Initialize an NMF factorization from the SVD, following Boutsidis and Gallopoulos.

    Each singular triplet is split into its positive and negative parts, and whichever pair
    carries more norm becomes a component — which gives a deterministic, non-negative starting
    point that converges in fewer iterations than a random one.

    Args:
    ----
        x: non-negative data (n_samples, n_features)
        n_components: number of components
        device: device to allocate the factors on

    Returns:
    -------
        the initial ``W`` (n_samples, n_components) and ``H`` (n_components, n_features), both
        floored at machine epsilon so no entry starts at exactly zero

    """
    u, s, vt = torch.linalg.svd(x, full_matrices=False)

    u = u[:, :n_components]
    s = s[:n_components]
    vt = vt[:n_components, :]

    w = torch.zeros(x.shape[0], n_components, device=device)
    h = torch.zeros(n_components, x.shape[1], device=device)

    # The leading singular vectors are sign-definite up to a global flip, so their magnitudes are
    # already a valid non-negative component; the split below is only needed from the second on.
    w[:, 0] = torch.sqrt(s[0]) * torch.abs(u[:, 0])
    h[0, :] = torch.sqrt(s[0]) * torch.abs(vt[0, :])

    for j in range(1, n_components):
        uj = u[:, j]
        vj = vt[j, :]
        sj = s[j]

        uj_pos = torch.clamp(uj, min=0)
        uj_neg = torch.clamp(-uj, min=0)
        vj_pos = torch.clamp(vj, min=0)
        vj_neg = torch.clamp(-vj, min=0)

        uj_pos_norm = torch.norm(uj_pos)
        uj_neg_norm = torch.norm(uj_neg)
        vj_pos_norm = torch.norm(vj_pos)
        vj_neg_norm = torch.norm(vj_neg)

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
    """Non-negative matrix factorization, factorizing ``X`` into ``W @ H`` with both factors ≥ 0.

    Both ``fit`` and ``transform`` use HALS — hierarchical alternating least squares, a block
    coordinate descent whose per-component update has a closed form — which converges in fewer
    sweeps than multiplicative updates while staying vectorized on the GPU.

    ``fit`` stores ``H`` as ``components_``, ordered by descending explained variance, and
    ``transform`` then solves for the ``W`` of new data against that fixed ``H``.

    Basic usage:

    ```
    nmf = NMF(n_components=10, seed=0)
    nmf.fit(x_train)
    scores = nmf.transform(x_test)
    ```

    Args:
    ----
        n_components: number of components; ``None`` uses ``min(n_samples, n_features)``, and
            ``fit`` writes the resolved value back to this attribute
        init: ``"nndsvd"`` for the deterministic SVD-based initialization, anything else for
            random
        max_iter: iteration ceiling for ``fit``
        tol: relative change in reconstruction error at which the iteration stops
        transform_max_iter: iteration ceiling for ``transform``
        seed: seed for the random initialization used by ``fit``, and by ``transform`` when no
            warm start is given. A per-call generator is used, so the global RNG is untouched and
            each ``transform`` reseeds independently — its output depends only on the seed, the
            input shape and ``components_``, never on how many transforms preceded it. A
            warm-started ``transform`` does not consult it.

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
        """Move the fitted components to a device.

        Args:
        ----
            device: destination device

        """
        self.components_ = self.components_.to(device)
        self.device = torch.device(device)

    def fit(self: Self, x: torch.Tensor, /) -> None:
        """Fit the factorization, storing ``H`` as ``components_``.

        A negative input is shifted to be non-negative rather than rejected, which changes what is
        being factorized — pass data that is already non-negative if that matters.

        Components are returned ordered by descending explained variance, so a caller taking the
        leading few gets the dominant ones.

        Args:
        ----
            x: data to factorize (n_samples, n_features)

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

        x_min = x.min()
        if x_min < 0:
            x = x - x_min + EPSILON

        with torch.no_grad():
            if self.init == "nndsvd":
                w, h = _nndsvd_init(x, self.n_components, self.device)
            else:  # random
                gen = torch.Generator(device=self.device).manual_seed(self.seed) if self.seed is not None else None
                w = torch.abs(torch.randn(self.n_samples, self.n_components, device=self.device, generator=gen)) * 0.01 + EPSILON
                h = torch.abs(torch.randn(self.n_components, n_features, device=self.device, generator=gen)) * 0.01 + EPSILON

            prev_error = float('inf')
            for i in range(self.max_iter):
                h_ht = h @ h.T
                x_ht = x @ h.T
                w = _hals_update_w(w, x_ht, h_ht)

                w_tw = w.T @ w
                w_tx = w.T @ x
                h = _hals_update_h(h, w_tx, w_tw)

                # Reconstructing to measure the error costs a full matmul, so it is checked on a
                # stride rather than every sweep; the stride bounds how far past ``tol`` this runs.
                if i % 10 == 0:
                    reconstruction = w @ h
                    error = torch.norm(x - reconstruction).item()
                    if abs(prev_error - error) / (prev_error + EPSILON) < self.tol:
                        break
                    prev_error = error

            # Rank by ||w_i||^2 * ||h_i||^2, the squared Frobenius norm of the rank-1 term each
            # component contributes, which stands in for explained variance.
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
        """Solve for the ``W`` of new data against the fitted ``H``.

        This is a convex non-negative least-squares problem, so the solution does not depend on
        where the iteration starts; ``w_init`` only reduces how many sweeps it takes to get there.

        The final ``W`` is left in ``last_transform_w_``, ready to warm-start the next call — which
        pays off when consecutive inputs are related, as successive training checkpoints are.

        A negative input is shifted to be non-negative, as in ``fit``.

        Args:
        ----
            z: data to transform (n_samples_new, n_features)
            components: which components to return; an integer is read as the leading that many,
                and ``None`` returns all
            w_init: warm start (n_samples_new, n_components), whose sample count must match ``z``;
                ``None`` starts from a random initialization
            use_tol: inert. Early stopping on ``tol`` always applies, whatever this is set to

        Returns:
        -------
            the transformed data (n_samples_new, n_selected_components)

        """
        if components is None:
            components = self.n_components
        if isinstance(components, int):
            components = list(range(components))

        z = z.to(self.device)

        z_min = z.min()
        if z_min < 0:
            z = z - z_min + EPSILON

        n_samples_new = z.shape[0]
        h = self.components_

        with torch.no_grad():
            if w_init is not None:
                w_new = w_init.clone().to(self.device).clamp_(min=EPSILON)
            else:
                # Reseeding per call is what makes the output a function of the seed, the shape
                # and ``components_`` alone; a generator advanced across calls would make it
                # depend on how many transforms came before.
                gen = torch.Generator(device=self.device).manual_seed(self.seed) if self.seed is not None else None
                w_new = torch.abs(torch.randn(n_samples_new, self.n_components, device=self.device, generator=gen)) * 0.01 + EPSILON

            h_ht = h @ h.T          # B = H H^T (n_components, n_components)
            z_ht = z @ h.T          # A = Z H^T (n_samples_new, n_components)

            # HALS, not a gradient method. Its per-column closed form is insensitive to the
            # conditioning of ``H H^T``, which degrades as n_components approaches n_features; a
            # projected-gradient step scaled by the inverse spectral norm of that matrix stalls in
            # exactly that regime, returning a reconstruction worse than the zero matrix while
            # reporting no error.
            w_new = w_new.clamp_(min=EPSILON)
            prev_error = float('inf')
            for i in range(self.transform_max_iter):
                w_new = _hals_update_w(w_new, z_ht, h_ht)
                if i % 5 == 0:
                    error = torch.norm(z - w_new @ h).item()
                    if abs(prev_error - error) / (prev_error + EPSILON) < self.tol:
                        break
                    prev_error = error

        self.last_transform_w_ = w_new
        return w_new[..., components]

    def inverse_transform(
        self: Self,
        w: torch.Tensor,
        /,
        *,
        components: Sequence[int] | int | None = None,
    ) -> torch.Tensor:
        """Reconstruct data from its transformed representation.

        Args:
        ----
            w: transformed data (n_samples, n_selected_components)
            components: the components ``w`` was transformed onto; an integer is read as the
                leading that many, and ``None`` uses all of them

        Returns:
        -------
            reconstructed data (n_samples, n_features)

        """
        if components is None:
            components = self.n_components
        if isinstance(components, int):
            components = list(range(components))

        w = w.to(self.device)
        return w @ self.components_[components, :]


class AsgMuNmf(Dataset):
    """Asymmetric gradient multiplicative update nonnegative-matrix factorization.

    A ``Dataset`` rather than an estimator: it factorizes out-of-core, streaming minibatches of
    rows through a ``DataLoader`` and updating only the rows of ``U`` a batch touches, so the data
    never has to be resident all at once.

    ``data`` must be a scipy sparse matrix — rows are densified per batch — and factorizing writes
    checkpoint pickles into the current working directory.
    """

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
        """Densify the requested rows, returning them alongside the indices they came from.

        Pass this as a ``DataLoader``'s ``collate_fn``. The indices are returned because the
        update writes back into the rows of ``U`` that the batch addressed.

        Args:
        ----
            indices: row indices in the batch

        Returns:
        -------
            the dense rows and their indices

        """
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
        """Factorize the data, returning the two factors.

        Writes into the current working directory: ``nmf_initialization.pkl`` before starting, and
        one ``nmf_epoch_{n}.pkl`` per epoch, each deleting its predecessor so only the newest
        survives. A run therefore leaves the initialization and the last epoch behind, and two
        runs in one directory overwrite each other.

        Args:
        ----
            u: initial (n_samples, n_components) factor; ``None`` draws a small random one
            v: initial (n_components, n_dimensions) factor; ``None`` draws a small random one
            n_epochs: passes over the data
            batch_size: rows per minibatch
            num_workers: ``DataLoader`` worker processes

        Returns:
        -------
            the fitted ``U`` and ``V`` as arrays on the host

        """
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
