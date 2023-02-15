import torch
from bonner.computation.decomposition import PCA


def cv_pca(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    pca = PCA()
    pca.fit(x.T)

    u = pca.components_.T
    sv = pca.singular_values_

    xproj = x.T @ (u / sv)
    cproj0 = x @ xproj
    cproj1 = y @ xproj
    ss = (cproj0 * cproj1).sum(dim=0)
    return ss
