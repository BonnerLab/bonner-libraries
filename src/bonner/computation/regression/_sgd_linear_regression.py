import torch
from torch import nn, optim
import numpy as np

from bonner.computation.regression._utilities import Regression


class SGDLinearRegression():
    def __init__(
        self, 
        lr: float = 1e-4, 
        fit_intercept: bool = True, 
        l1_strength: float = 0.0, 
        l2_strength: float = 0.0, 
        max_epoch: int = 1000,
        tol: float = 1e-2,
        num_epoch_tol: int = 20,
        batch_size: int = 1000,
        seed: int = 11,
    ):
        self._lr = lr
        self._fit_intercept = fit_intercept
        self._l1_strength = l1_strength
        self._l2_strength = l2_strength
        
        self._max_epoch = max_epoch
        self._tol = tol
        self._num_epoch_tol = num_epoch_tol
        self._batch_size = batch_size
        self._seed = seed
        self._device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self._loss_func = nn.MSELoss(reduction='sum')
        self._linear = None
        self._optimizer = None
    
    def fit(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> None:
        rng = np.random.default_rng(self._seed)
        assert x.shape[-2] == y.shape[-2]
        idx = np.arange(x.shape[-2])
        losses = []
        for _ in range(self._max_epoch):
            rng.shuffle(idx)
            epoch_loss = []
            for j in range(0, len(idx), self._batch_size):
                epoch_loss.append(
                    self._fit_partial(
                        x[idx[j : j + self._batch_size]],
                        y[idx[j : j + self._batch_size]],
                    )
                )
            epoch_loss = np.mean(epoch_loss)
            if len(losses) >= self._num_epoch_tol:
                if epoch_loss / np.mean(losses[-self._num_epoch_tol:]) > 1 - self._tol:
                    break
            losses.append(epoch_loss)
        
    def _fit_partial(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor,
    ) -> float:
        x = torch.clone(x).to(self._device)
        y = torch.clone(y).to(self._device)
        self._initialize_from(x, y)
        
        loss = self._loss_func(self._linear(x), y)

        # L1 regularizer
        if self._l1_strength > 0:
            l1_reg = self._linear.weight.abs().sum()
            loss += self._l1_strength * l1_reg

        # L2 regularizer
        if self._l2_strength > 0:
            l2_reg = self._linear.weight.pow(2).sum()
            loss += self._l2_strength * l2_reg

        loss /= x.size(0)

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        return loss.item()

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        assert self._linear is not None
        x = x.to(self._device)
        with torch.no_grad():
            preds = self._linear(x)
        return preds

    def _initialize_from(self, x: torch.Tensor, y: torch.Tensor):
        if self._linear is None:
            self._linear = nn.Linear(
                x.shape[1], 
                y.shape[1],
                bias=self._fit_intercept, 
                device=self._device,
            )
            self._optimizer = optim.Adam(
                self._linear.parameters(),
                lr=self._lr,
            )