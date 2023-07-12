import logging

logging.basicConfig(level=logging.INFO)

import torch
from torch import nn, optim
import numpy as np

from bonner.computation.regression._utilities import Regression

MIN_LR = 1e-6
LR_STEP = 5


class SGDLinearRegression(Regression):
    def __init__(
        self, 
        lr: float = 1e-2, 
        adaptive: bool = True,
        fit_intercept: bool = True, 
        l1_strength: float = 0.0, 
        l2_strength: float = 0.0, 
        max_epoch: int = 1000,
        tol: float = 1e-3,
        num_epoch_tol: int = 10,
        batch_size: int = 1000,
        seed: int = 11,
    ):
        self._adaptive = adaptive
        self._lr0 = lr
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
        self._initialized = False
    
    def fit(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> None:
        self._initialized = False
        
        rng = np.random.default_rng(self._seed)
        assert x.size(-2) == y.size(-2)
        n_sample = x.size(-2)
        idx = np.arange(n_sample)
        best_loss, n_tol = None, 0
        for e in range(self._max_epoch):
            rng.shuffle(idx)
            epoch_loss = []
            for j in range(0, n_sample, self._batch_size):
                j_end = np.min((j + self._batch_size, n_sample))
                epoch_loss.append(self._fit_partial(
                    x[idx[j : j_end]],
                    y[idx[j : j_end]],
                ))
            epoch_loss = np.mean(epoch_loss)
            
            if not best_loss:
                best_loss = epoch_loss
            elif epoch_loss / best_loss > 1 - self._tol:
                n_tol += 1
            else:
                ntol = 0
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    
            if n_tol >= self._num_epoch_tol:
                if self._adaptive and self._lr > MIN_LR:
                    self._lr /= LR_STEP
                    for g in self._optimizer.param_groups:
                        g["lr"] = self._lr
                    n_tol = 0
                else:
                    # logging.info(f"Early stopped at epoch {e}: loss = {epoch_loss:.2e}")
                    break     
                
            if e == self._max_epoch - 1:
                logging.info(f"No convergence: loss = {epoch_loss:.2e}")  
        
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
        # return preds * self._y_std + self._y_mean

    def _initialize_from(self, x: torch.Tensor, y: torch.Tensor):
        if not self._initialized:
            self._lr = self._lr0
            self._linear = nn.Linear(
                x.shape[1], 
                y.shape[1],
                bias=self._fit_intercept, 
                device=self._device,
            )
            self._optimizer = optim.Adam(
                self._linear.parameters(),
                lr=self._lr0,
            )
            self._initialized = True