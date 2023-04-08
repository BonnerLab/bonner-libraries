import torch

from bonner.computation.regression._utilities import Regression

def _get_first_singular_vectors_power_method(
    x: torch.Tensor,
    y: torch.Tensor,
    max_iter: int,
    tol: float
 ) -> tuple[torch.Tensor, torch.Tensor, int]:
    eps = torch.finfo(x.dtype).eps
    y_score = next(col for col in y.T if torch.any(torch.abs(col) > eps))
    x_weights_old = 100
    for i in range(max_iter):
        x_weights = (x.T @ y_score) / (y_score @ y_score)
        x_weights /= torch.sqrt(x_weights @ x_weights) + eps
        x_score = x @ x_weights
        y_weights = (y.T @ x_score) / (x_score.T @ x_score)
        y_score = (y @ y_weights) / ((y_weights @ y_weights) + eps)
        x_weights_diff = x_weights - x_weights_old
        if (x_weights_diff @ x_weights_diff) < tol or y.shape[1] == 1:
            break
        x_weights_old = x_weights
    n_iter = i + 1
    return x_weights, y_weights, n_iter


def _svd_flip_1d(u: torch.Tensor, v: torch.Tensor) -> None:
    biggest_abs_val_idx = torch.argmax(torch.abs(u))
    sign = torch.sign(u[biggest_abs_val_idx])
    # in-place operation
    u *= sign
    v *= sign


class PLSRegression(Regression):
    def __init__(
        self,
        n_components: int = 25,
        scale: bool = True,
        max_iter: int = 500,
        tol: float = 1e-06,
    ) -> None:
        self.n_components = n_components
        self.scale = scale
        self.max_iter = max_iter
        self.tol = tol
        
    def fit(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> None:
        x = torch.clone(x)
        y = torch.clone(y).to(x.device)

        x = x.unsqueeze(dim=-1) if x.ndim == 1 else x
        y = y.unsqueeze(dim=-1) if y.ndim == 1 else y

        # many sets of predictors, only 1 set of targets
        if x.ndim == 3 and y.ndim == 2:
            y = y.unsqueeze(0)
            
        n_samples, n_features = x.shape[-2], x.shape[-1]
        
        if y.shape[-2] != n_samples:
            raise ValueError(
                f"number of samples in x and y must be equal (x={n_samples},"
                f" y={y.shape[-2]})"
            )
            
        self.x_mean = x.mean(dim=-2, keepdim=True)
        y -= self.y_mean
        self.y_mean = y.mean(dim=-2, keepdim=True)
        x -= self.x_mean
        
        if self.scale:
            self.x_std = x.std(dim=-2, keepdim=True)
            self.y_std = y.std(dim=-2, keepdim=True)
            self.x_std[self.x_std == 0.0] = 1.0
            self.y_std[self.y_std == 0.0] = 1.0
            x /= self.x_std
            y /= self.y_std
        else:
            self.x_std, self.y_std = torch.ones(n_features), torch.ones(y.shape[-1])
            
        
            
            
        
        