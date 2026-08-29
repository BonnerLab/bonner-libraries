"""K-Means clustering for finding interpretable directions in activation space.

Based on: Klindt et al. (2023) "Identifying Interpretable Visual Features in
Artificial and Biological Neural Systems" (arXiv:2310.11431)

The centroids from K-Means define non-axis-aligned directions in activation space
that can be more interpretable than individual neurons.
"""

from collections.abc import Sequence
from typing import Literal, Self

import torch

EPSILON = torch.finfo(torch.float32).eps


class KMeans:
    """K-Means clustering with fit/transform interface.

    Finds K cluster centroids that serve as interpretable directions in activation space.
    Supports both Euclidean and cosine distance metrics.

    Based on the approach from Klindt et al. (2023) for finding interpretable
    features in neural networks.

    Args:
        n_clusters: Number of clusters/directions to find. If None, defaults to
            min(n_samples, n_features) like PCA's n_components.
        metric: Distance metric - 'euclidean' or 'cosine'. Paper uses 'cosine'.
        max_iter: Maximum number of iterations for fitting.
        tol: Tolerance for convergence (relative change in inertia).
        n_init: Number of initializations to try, keeping the best.
        init: Initialization method - 'kmeans++' or 'random'.
        random_state: Random seed for reproducibility. Applied by seeding the global torch
            generator, so a fit with this set perturbs any other sampling in the process.
    """

    def __init__(
        self: Self,
        *,
        n_clusters: int | None = None,
        metric: Literal["euclidean", "cosine"] = "cosine",
        max_iter: int = 300,
        tol: float = 1e-4,
        n_init: int = 10,
        init: Literal["kmeans++", "random"] = "kmeans++",
        random_state: int | None = None,
    ) -> None:
        self.n_clusters = n_clusters
        self.metric = metric
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.init = init
        self.random_state = random_state

        self.n_samples: int
        self.n_features: int
        self.centroids: torch.Tensor  # (n_clusters, n_features)
        self.labels_: torch.Tensor  # (n_samples,) cluster assignments from fit
        self.inertia_: float  # Sum of distances to closest centroid
        self.device: torch.device

    def to(self: Self, device: torch.device | str) -> Self:
        """Move model to specified device."""
        self.centroids = self.centroids.to(device)
        if hasattr(self, "labels_"):
            self.labels_ = self.labels_.to(device)
        self.device = torch.device(device)
        return self

    def _normalize(self: Self, x: torch.Tensor) -> torch.Tensor:
        """L2 normalize for cosine distance."""
        return x / (torch.norm(x, dim=-1, keepdim=True) + EPSILON)

    def _compute_distances(
        self: Self,
        x: torch.Tensor,
        centroids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute distances from samples to centroids.

        Args:
            x: Data of shape (..., n_samples, n_features)
            centroids: Centroids of shape (n_clusters, n_features)

        Returns:
            Distances of shape (..., n_samples, n_clusters)
        """
        if self.metric == "cosine":
            x_norm = self._normalize(x)
            centroids_norm = self._normalize(centroids)
            similarity = x_norm @ centroids_norm.T
            return 1.0 - similarity
        else:
            # Expanded as ||x||^2 + ||c||^2 - 2 x.c so the pairwise distances come from one
            # matmul instead of an (n_samples, n_clusters, n_features) difference tensor.
            x_sq = (x**2).sum(dim=-1, keepdim=True)
            c_sq = (centroids**2).sum(dim=-1).unsqueeze(0)
            cross = x @ centroids.T
            distances_sq = x_sq + c_sq - 2 * cross
            return torch.sqrt(torch.clamp(distances_sq, min=0))

    def _compute_similarity(
        self: Self,
        x: torch.Tensor,
        centroids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute similarity from samples to centroids.

        Args:
            x: Data of shape (..., n_samples, n_features)
            centroids: Centroids of shape (n_clusters, n_features)

        Returns:
            Similarity of shape (..., n_samples, n_clusters)
        """
        if self.metric == "cosine":
            x_norm = self._normalize(x)
            centroids_norm = self._normalize(centroids)
            return x_norm @ centroids_norm.T
        else:
            distances = self._compute_distances(x, centroids)
            return -distances

    def _kmeans_plusplus_init(
        self: Self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """K-Means++ initialization for better convergence.

        Args:
            x: Data of shape (n_samples, n_features)

        Returns:
            Initial centroids of shape (n_clusters, n_features)
        """
        n_samples = x.shape[0]
        centroids = torch.empty(
            self.n_clusters, self.n_features, device=self.device, dtype=x.dtype
        )

        idx = torch.randint(n_samples, (1,), device=self.device)
        centroids[0] = x[idx]

        for k in range(1, self.n_clusters):
            distances = self._compute_distances(x, centroids[:k])
            min_distances, _ = distances.min(dim=-1)

            if self.metric == "euclidean":
                weights = min_distances**2
            else:
                weights = min_distances**2

            weights = weights / (weights.sum() + EPSILON)

            idx = torch.multinomial(weights, 1)
            centroids[k] = x[idx]

        return centroids

    def _random_init(self: Self, x: torch.Tensor) -> torch.Tensor:
        """Random initialization by sampling data points."""
        n_samples = x.shape[0]
        indices = torch.randperm(n_samples, device=self.device)[: self.n_clusters]
        return x[indices].clone()

    def _single_fit(
        self: Self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Run a single K-Means fit.

        Returns:
            (centroids, labels, inertia)
        """
        if self.init == "kmeans++":
            centroids = self._kmeans_plusplus_init(x)
        else:
            centroids = self._random_init(x)

        prev_inertia = float("inf")

        for _ in range(self.max_iter):
            distances = self._compute_distances(x, centroids)
            labels = distances.argmin(dim=-1)
            inertia = distances.min(dim=-1).values.sum().item()

            if abs(prev_inertia - inertia) / (prev_inertia + EPSILON) < self.tol:
                break
            prev_inertia = inertia

            new_centroids = torch.zeros_like(centroids)
            for k in range(self.n_clusters):
                mask = labels == k
                if mask.sum() > 0:
                    new_centroids[k] = x[mask].mean(dim=0)
                else:
                    # An emptied cluster is reseeded from a random sample rather than dropped,
                    # so the returned centroid count always equals n_clusters.
                    idx = torch.randint(x.shape[0], (1,), device=self.device)
                    new_centroids[k] = x[idx]

            centroids = new_centroids

        return centroids, labels, inertia

    def fit(self: Self, x: torch.Tensor, /) -> Self:
        """Fit K-Means to data X.

        Finds K cluster centroids that serve as interpretable directions.

        Args:
            x: Input data of shape (n_samples, n_features).

        Returns:
            self
        """
        if x.ndim == 1:
            x = x.unsqueeze(dim=-1)

        self.n_samples, self.n_features = x.shape[-2], x.shape[-1]
        self.device = x.device

        max_n_clusters = min(self.n_samples, self.n_features)
        if self.n_clusters is None:
            self.n_clusters = max_n_clusters
        elif self.n_clusters > max_n_clusters:
            error = f"n_clusters ({self.n_clusters}) must be <= {max_n_clusters}"
            raise ValueError(error)

        best_inertia = float("inf")
        best_centroids = None
        best_labels = None

        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        with torch.no_grad():
            for _ in range(self.n_init):
                centroids, labels, inertia = self._single_fit(x)

                if inertia < best_inertia:
                    best_inertia = inertia
                    best_centroids = centroids.clone()
                    best_labels = labels.clone()

        self.centroids = best_centroids
        self.labels_ = best_labels
        self.inertia_ = best_inertia

        return self

    def transform(
        self: Self,
        z: torch.Tensor,
        /,
        *,
        components: Sequence[int] | int | None = None,
    ) -> torch.Tensor:
        """Transform data Z by projecting onto fitted centroids (directions).

        Returns the projection (dot product) of each sample onto each centroid.
        This is analogous to PCA's transform returning projections onto eigenvectors.

        For cosine metric: returns cosine similarity (normalized dot product)
        For euclidean metric: returns dot product with centroids

        Args:
            z: Data to transform of shape (n_samples_new, n_features).
            components: Which clusters/directions to return. If None, returns all.

        Returns:
            Projections of shape (n_samples_new, n_clusters).
        """
        if components is None:
            components = self.n_clusters
        if isinstance(components, int):
            components = list(range(components))

        z = z.to(self.device)

        if self.metric == "cosine":
            z_norm = self._normalize(z)
            centroids_norm = self._normalize(self.centroids)
            projections = z_norm @ centroids_norm.T
        else:
            projections = z @ self.centroids.T

        return projections[..., components]

    def transform_distances(
        self: Self,
        z: torch.Tensor,
        /,
        *,
        components: Sequence[int] | int | None = None,
    ) -> torch.Tensor:
        """Get distances from samples to centroids.

        Args:
            z: Data of shape (n_samples, n_features).
            components: Which clusters to return.

        Returns:
            Distances of shape (n_samples, n_clusters).
        """
        if components is None:
            components = self.n_clusters
        if isinstance(components, int):
            components = list(range(components))

        z = z.to(self.device)
        distances = self._compute_distances(z, self.centroids)
        return distances[..., components]

    def predict(self: Self, z: torch.Tensor, /) -> torch.Tensor:
        """Predict cluster labels for data Z.

        Args:
            z: Data of shape (n_samples, n_features).

        Returns:
            Cluster labels of shape (n_samples,).
        """
        z = z.to(self.device)
        distances = self._compute_distances(z, self.centroids)
        return distances.argmin(dim=-1)

    def inverse_transform(
        self: Self,
        labels: torch.Tensor,
        /,
        *,
        components: Sequence[int] | int | None = None,
    ) -> torch.Tensor:
        """Get centroids for given cluster labels.

        Args:
            labels: Cluster labels of shape (n_samples,).
            components: Which components were used (for compatibility).

        Returns:
            Centroids of shape (n_samples, n_features).
        """
        labels = labels.to(self.device)
        return self.centroids[labels]

    def get_meis(
        self: Self,
        x: torch.Tensor,
        /,
        *,
        n_meis: int = 5,
        cluster_idx: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get Maximally Exciting Images (MEIs) for each cluster.

        MEIs are the samples with highest similarity to each centroid.
        This is the key method for the interpretability analysis from Klindt et al.

        Args:
            x: Data of shape (n_samples, n_features).
            n_meis: Number of MEIs to return per cluster.
            cluster_idx: If specified, only return MEIs for this cluster.

        Returns:
            (mei_indices, mei_similarities):
                - mei_indices: Shape (n_clusters, n_meis) or (n_meis,) if cluster_idx
                - mei_similarities: Same shape, the similarity values
        """
        x = x.to(self.device)
        similarity = self._compute_similarity(x, self.centroids)

        if cluster_idx is not None:
            sim_k = similarity[..., cluster_idx]
            top_values, top_indices = torch.topk(sim_k, k=n_meis, dim=-1)
            return top_indices, top_values

        n_clusters = self.centroids.shape[0]
        mei_indices = torch.empty(
            n_clusters, n_meis, dtype=torch.long, device=self.device
        )
        mei_similarities = torch.empty(n_clusters, n_meis, device=self.device)

        for k in range(n_clusters):
            sim_k = similarity[..., k]
            top_values, top_indices = torch.topk(sim_k, k=n_meis, dim=-1)
            mei_indices[k] = top_indices
            mei_similarities[k] = top_values

        return mei_indices, mei_similarities


class MiniBatchKMeans(KMeans):
    """Mini-batch K-Means for large datasets.

    Uses mini-batches for scalability while maintaining similar results.
    Particularly useful for large activation datasets.

    Args:
        n_clusters: Number of clusters/directions. If None, defaults to
            min(n_samples, n_features).
        metric: Distance metric - 'euclidean' or 'cosine'.
        max_iter: Maximum number of iterations.
        batch_size: Size of mini-batches.
        tol: Tolerance for convergence.
        n_init: Number of random initializations.
        init: Initialization method.
        random_state: Random seed for reproducibility.
    """

    def __init__(
        self: Self,
        *,
        n_clusters: int | None = None,
        metric: Literal["euclidean", "cosine"] = "cosine",
        max_iter: int = 100,
        batch_size: int = 1024,
        tol: float = 1e-4,
        n_init: int = 3,
        init: Literal["kmeans++", "random"] = "kmeans++",
        random_state: int | None = None,
    ) -> None:
        super().__init__(
            n_clusters=n_clusters,
            metric=metric,
            max_iter=max_iter,
            tol=tol,
            n_init=n_init,
            init=init,
            random_state=random_state,
        )
        self.batch_size = batch_size

    def _single_fit(
        self: Self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Run a single mini-batch K-Means fit."""
        n_samples = x.shape[0]

        if self.init == "kmeans++":
            centroids = self._kmeans_plusplus_init(x)
        else:
            centroids = self._random_init(x)

        # Per-centroid counts accumulate across batches, so the step size below shrinks as a
        # centroid absorbs more samples — the streaming mean, not a tuned schedule.
        counts = torch.ones(self.n_clusters, device=self.device)
        prev_inertia = float("inf")

        for iteration in range(self.max_iter):
            batch_indices = torch.randperm(n_samples, device=self.device)[
                : self.batch_size
            ]
            x_batch = x[batch_indices]

            distances = self._compute_distances(x_batch, centroids)
            labels = distances.argmin(dim=-1)

            for k in range(self.n_clusters):
                mask = labels == k
                if mask.sum() > 0:
                    counts[k] += mask.sum()
                    eta = 1.0 / counts[k]
                    centroids[k] = (1 - eta) * centroids[k] + eta * x_batch[mask].mean(
                        dim=0
                    )

            # Inertia is measured over the whole dataset, not the batch, so this costs a full
            # distance computation and is taken on a stride rather than every iteration.
            if iteration % 10 == 0:
                all_distances = self._compute_distances(x, centroids)
                inertia = all_distances.min(dim=-1).values.sum().item()
                if abs(prev_inertia - inertia) / (prev_inertia + EPSILON) < self.tol:
                    break
                prev_inertia = inertia

        distances = self._compute_distances(x, centroids)
        labels = distances.argmin(dim=-1)
        inertia = distances.min(dim=-1).values.sum().item()

        return centroids, labels, inertia
