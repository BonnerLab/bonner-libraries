import itertools
from typing import Any

import torch

from bonner.computation.metrics._corrcoef import pearson_r


class CCAPytorch:
    """Based on pyrcca's _CCABase class https://github.com/gallantlab/pyrcca/blob/main/rcca/rcca.py#L13.

    Based on sklearn's CCA class https://github.com/scikit-learn/scikit-learn/blob/baf828ca1/sklearn/cross_decomposition/_pls.py#L801
    """

    def __init__(
        self,
        alpha: float = 0,
        n_components: int = 2,
        scale: bool = True,
        tol: float = 1e-15,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
        kwargs_kernel: dict[str, Any] = None,
    ) -> None:
        """Initialize the CCA class.

        Args:
            alpha: strength of regularization, defaults to 0
            n_components: defaults to 2
            scale: whether to standardize the data to unit variance, defaults to True
            tol: tolerance, defaults to 1e-15
            device: torch.device, defaults to None
            kwargs_kernel: kernel arguments, defaults to None
        """
        self.alpha = alpha
        self.n_components = n_components
        self.scale = scale
        self.tol = tol
        self.dtype = dtype
        self.kwargs_kernel = kwargs_kernel
        self.device = device

        self._means = []
        self._stds = []

    def _center_scale(self, data: list[torch.Tensor]) -> list[torch.Tensor]:
        self._means = [d.mean(dim=0).to(self.device).to(self.dtype) for d in data]
        if self.scale:
            self._stds = []
            for d in data:
                std = d.std(dim=0).to(self.device)
                std[std == 0] = 1
                self._stds.append(std)
        else:
            self._stds = [
                torch.ones(d.shape[1], device=self.device, dtype=self.dtype)
                for d in data
            ]

        data = [(d - mean) / std for d, mean, std in zip(data, self._means, self._stds)]
        return data

    def fit(self, data: list[torch.Tensor]) -> None:
        """Fit the CCA model.

        Args:
            data: list of tensors of shape (n_samples, n_features); n_samples must be the same for all tensors
        """
        data = [d.to(self.device).to(self.dtype) for d in data]
        data_preprocessed = self._center_scale(data)

        self._canonical_coefficients = _compute_canonical_coefficients(
            data=data_preprocessed,
            alpha=self.alpha,
            n_components=self.n_components,
            kwargs_kernel=self.kwargs_kernel,
            device=self.device,
        )
        if self.kwargs_kernel:
            self._weights = [
                d.t() @ c
                for d, c in zip(data_preprocessed, self._canonical_coefficients)
            ]
        else:
            self._weights = self._canonical_coefficients
        self._canonical_correlations = self.compute_canonical_correlations(data)

    def transform(self, data: list[torch.Tensor]) -> list[torch.Tensor]:
        data = [
            (d.to(self.device).to(self.dtype) - mean) / std
            for d, mean, std in zip(data, self._means, self._stds)
        ]
        return [d @ w for d, w in zip(data, self._weights)]

    def compute_canonical_correlations(self, data: list[torch.Tensor]) -> torch.Tensor:
        canonical_correlations = torch.zeros(
            self.n_components,
            len(data),
            len(data),
            device=self.device,
            dtype=self.dtype,
        )
        canonical_components = self.transform(data)
        for d1, d2 in itertools.combinations(range(len(data)), 2):
            canonical_correlations[:, d1, d2] = pearson_r(
                x=canonical_components[d1].t(),
                y=canonical_components[d2].t(),
            )
        if len(data) == 2:
            canonical_correlations = canonical_correlations[
                torch.nonzero(canonical_correlations, as_tuple=True)
            ]
        return canonical_correlations


#     def validate(self, data: list[torch.Tensor]) -> list[torch.Tensor]:
#         data = self.preprocess(data)
#         self.predictions = _compute_predictions(data, self._ws, rtol=self.tol)
#         self.correlations = _compute_correlations(data, self.predictions)
#         return self.correlations

#     def compute_explained_variance(
#         self, data: list[torch.Tensor]
#     ) -> list[torch.Tensor]:
#         # FIXME potential bug: https://github.com/gallantlab/pyrcca/issues/20
#         data = [d.to(self.device).float() for d in data]
#         return _compute_explained_variance(data=data, ws=self._ws, rtol=self.tol)


# def _compute_explained_variance(
#     data: list[torch.Tensor], ws: list[torch.Tensor], rtol: float = 1e-15
# ) -> list[torch.Tensor]:
#     with torch.no_grad():
#         n_cc = ws[0].shape[1]
#         n_features = [d.shape[1] for d in data]
#         evs = [torch.zeros((n_cc, f)) for f in n_features]

#         for i_cc in range(n_cc):
#             predictions = _compute_predictions(
#                 data, [w[:, i_cc : i_cc + 1] for w in ws], rtol=rtol
#             )
#             residuals = [(d - p).abs() for d, p in zip(data, predictions)]
#             for i_data, (d, r) in enumerate(zip(data, residuals)):
#                 d_v = d.var(dim=0)
#                 ev = abs(d_v - r.var(dim=0)) / d_v
#                 ev[torch.isnan(ev)] = 0
#                 evs[i_data][i_cc, :] = ev
#         return evs


# def _compute_predictions(
#     data: list[torch.Tensor], ws: list[torch.Tensor], rtol: float = 1e-15
# ) -> list[torch.Tensor]:
#     with torch.no_grad():
#         inverse_ws = [torch.linalg.pinv(w, rtol=rtol) for w in ws]
#         c_components = torch.stack(
#             [d @ w for d, w in zip(data, ws)], dim=0
#         )
#         predictions = []

#         for d in range(len(data)):
#             idx = torch.ones(len(data))
#             idx[d] = 0
#             projection = c_components[idx > 0].mean(dim=0)
#             prediction = projection @ inverse_ws[d]
#             prediction = (prediction - prediction.mean(dim=0)) / prediction.std(dim=0)
#             predictions.append(prediction)
#         return predictions


# def _compute_correlations(
#     data: list[torch.Tensor], predictions: list[torch.Tensor]
# ) -> list[torch.Tensor]:
#     with torch.no_grad():
#         return [
#             pairwise_corrcoef(
#                 x=d.t(),
#                 y=prediction.t(),
#             )
#             for d, prediction in zip(data, predictions)
#         ]


def _compute_canonical_coefficients(
    data: list[torch.Tensor],
    alpha: float = 0.0,
    n_components: int = None,
    kwargs_kernel: dict[str, Any] = None,
    device: torch.device | str | None = None,
) -> list[torch.Tensor]:
    """Compute the canonical coefficients.

    Args:
        data: collection of matrices of shape (n_samples, n_features), where the data are centered
        alpha: regularization parameter, defaults to 0.0
        n_cc: number of canonical correlations to compute, defaults to None
        kwargs_kernel: arguments to be passed to _kernelize_data, defaults to None

    Returns:
        TODO not sure
    """
    with torch.no_grad():
        if kwargs_kernel is not None:
            data = [_kernelize_data(d, device=device, **kwargs_kernel) for d in data]
            # TODO check whether kernel CCA actually works
            raise NotImplementedError("kernel CCA has not been tested yet")
        else:
            data = [d.t() for d in data]

        n_datasets = len(data)
        n_features = [d.shape[0] for d in data]
        if n_components is None:
            n_components = min(n_features)
        dtype = data[0].dtype

        covariances = [[d1 @ d2.t() for d2 in data] for d1 in data]

        n = sum(n_features)
        lh = torch.zeros((n, n), device=device, dtype=dtype)
        rh = torch.zeros((n, n), device=device, dtype=dtype)

        for d1, d2 in itertools.product(range(n_datasets), repeat=2):
            s1 = slice(sum(n_features[:d1]), sum(n_features[: d1 + 1]))
            s2 = slice(sum(n_features[:d2]), sum(n_features[: d2 + 1]))
            rh[s1, s1] = covariances[d1][d1] + alpha * torch.eye(
                n_features[d1],
                device=device,
                dtype=dtype,
            )
            if d1 != d2:
                lh[s2, s1] = covariances[d2][d1]

        lh = (lh + lh.t()) / 2
        rh = (rh + rh.t()) / 2

        eigenvalues, v = torch.lobpcg(A=lh, k=n_components, B=rh, largest=True)
        v = v[:, torch.argsort(eigenvalues).flip(dims=(0,))]
        return [
            v[sum(n_features[:d]) : sum(n_features[: d + 1]), :n_components]
            for d in range(n_datasets)
        ]


def _kernelize_data(
    data: torch.Tensor,
    normalize: bool = True,
    kernel: str = "linear",
    sigma: float = 1,
    degree: int = 2,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Kernelizes the input data.

    If `kernel` is "linear", the kernel is a linear inner product
    If `kernel` is "gaussian", the kernel is a Gaussian kernel, with standard deviation `sigma`
    If `kernel` is "polynomial", the kernel is a polynomial kernel with degree `degree`

    Args:
        data: matrix of shape (n_samples, n_features), where the data are centered
        normalize: TODO not sure what this means, defaults to True
        kernel: kernel to be used, defaults to "linear"
        sigma: standard deviation of the Gaussian kernel used, defaults to 1
        degree: degree of the polynomial kernel used, defaults to 2

    Raises:
        NotImplementedError: the Gaussian kernel is not implemented yet since torch.squareform() is not implemented

    Returns:
        kernelized data
    """
    with torch.no_grad():
        data = data.to(device)
        match kernel:
            case "linear":
                data_kernelized = data @ data.t()
            case "gaussian":
                pairwise_distances = torch.nn.functional.pdist(data)
                data_kernelized = torch.exp(
                    -(pairwise_distances**2) / (2 * sigma**2)
                )
                raise NotImplementedError("squareform has to be applied")
            case "polynomial":
                data_kernelized = (data @ data.t()) ** degree

        data_kernelized = (data_kernelized + data_kernelized.t()) / 2
        if normalize:
            data_kernelized = (
                data_kernelized / torch.linalg.eigvalsh(data_kernelized).max()
            )
        return data_kernelized
