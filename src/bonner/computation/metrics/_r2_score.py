import torch


def r2_score(y: torch.Tensor, y_predicted: torch.Tensor) -> torch.Tensor:
    y = y.unsqueeze(1) if y.ndim == 1 else y
    y_predicted = y_predicted.unsqueeze(1) if y_predicted.ndim == 1 else y_predicted

    sse = ((y - y_predicted) ** 2).sum(dim=-2)
    ss = ((y - y.mean(dim=-2, keepdim=True)) ** 2).sum(dim=-2)
    return 1 - sse / ss
