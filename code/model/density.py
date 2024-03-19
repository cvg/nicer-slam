import torch
import torch.nn as nn


class Density(nn.Module):
    def __init__(self, params_init={}):
        super().__init__()
        for p in params_init:
            param = nn.Parameter(torch.tensor(params_init[p]))
            setattr(self, p, param)

    def forward(self, sdf, beta=None, x=None):
        return self.density_func(sdf, beta=beta, x=x)


class LaplaceDensity(Density):  # alpha * Laplace(loc=0, scale=beta).cdf(-sdf)
    def __init__(self, params_init={}, beta_min=0.0001):
        super().__init__(params_init=params_init)
        self.beta_min = torch.tensor(beta_min).cuda()

    def density_func(self, sdf, beta=None, x=None):
        if beta is None:
            beta = self.get_beta()
        alpha = 1 / beta
        return alpha * (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() / beta))

    def get_beta(self, x=None):
        beta = self.beta.abs() + self.beta_min
        return beta


# the beta is assigned according to the voxel counter, e.g. how many times a voxel is accessed during the mapping
class GridPredefineDensity(nn.Module):
    def __init__(self):
        super().__init__()

    def density_func(self, sdf, beta=None, x=None):
        if beta is None:
            beta = self.get_beta(x)
        alpha = 1 / beta
        return alpha * (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() / beta))

    def func(self, x):
        count = torch.zeros((x.shape[0])).cuda()
        mask = torch.zeros((x.shape[0])).bool().cuda()
        for dim in range(3):
            dim_mask = torch.abs(x[:, dim]) > 0.99
            mask |= dim_mask
        x = x[~mask]
        x = (x + 1) / 2
        x = (x * self.voxel_res).long()
        tmpcount = self.voxels[x[:, 0], x[:, 1], x[:, 2]]
        count[~mask] = tmpcount
        count[mask] = 0

        a = 0.01207724805
        b = 0.0116544676
        c = 0.0023639156
        d = 5.37538
        return a * torch.exp(-b * 0.0001 * count * d) + c

    def get_beta(self, x):
        beta = self.func(x).cuda().unsqueeze(-1)
        return beta

    def forward(self, sdf, x=None, beta=None):
        return self.density_func(sdf, x=x, beta=beta)


class AbsDensity(Density):  # like NeRF++
    def density_func(self, sdf, beta=None):
        return torch.abs(sdf)


class SimpleDensity(Density):  # like NeRF
    def __init__(self, params_init={}, noise_std=1.0):
        super().__init__(params_init=params_init)
        self.noise_std = noise_std

    def density_func(self, sdf, beta=None):
        if self.training and self.noise_std > 0.0:
            noise = torch.randn(sdf.shape).cuda() * self.noise_std
            sdf = sdf + noise
        return torch.relu(sdf)
