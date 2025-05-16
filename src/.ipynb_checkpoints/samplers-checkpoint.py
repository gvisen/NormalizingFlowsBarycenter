import numpy as np
import torch
from sklearn.datasets import make_swiss_roll
from tqdm.notebook import tqdm

from .utils import AffineTransform, sqrtm

class Sampler:
    def __init__(self, device='cuda'):
        self.device = device
        self.mean = None
        self.cov = None
        self.var = None
    
    def sample(self, num_samples=1):
        raise NotImplementerError

    def to(self, device):
        self.device = device

class Uniform(Sampler):
    """Uniform in [-sqrt(3), sqrt(3)].
    """

    def __init__(self, d, device='cuda'):
        super().__init__(device=device)
        self.distr = torch.distributions.uniform.Uniform(-pow(3, 0.5), pow(3, 0.5))
        self.d = d
        self.mean = torch.zeros(self.d).to(self.device)
        self.cov = torch.eye(self.d).to(self.device)
        self.var = self.d
        
    def sample(self, num_samples=1):
        z = self.distr.sample(sample_shape=(num_samples, self.d)).to(self.device)
        return z

class Gaussian(Sampler):
    """Standard Gaussian.
    """

    def __init__(self, d, mean=None, cov=None, device='cuda'):
        super().__init__(device=device)
        self.d = d
        self.dim = d
        if mean is None:
            self.mean = torch.zeros(d).to(self.device)
        else:
            self.mean = mean

        if cov is None:
            self.cov = torch.eye(d).to(self.device)
            self.var = self.d
        else:
            self.cov = cov
            self.var = torch.trace(cov).item()

        self.distr = torch.distributions.multivariate_normal.MultivariateNormal(self.mean, covariance_matrix=self.cov)
        

    def sample(self, num_samples=1):
        z = self.distr.sample(sample_shape=(num_samples,)).to(self.device)
        return z

class SwissRoll(Sampler):
    def __init__(self, device='cuda'):
        super().__init__(device=device)
        self.dim = 2
        self.d = 2

        batch = self.sample(1024)        
        self.mean = torch.mean(batch, dim=0)
        self.cov = torch.cov(batch.T)
        
    def sample(self, num_samples=1):
        sample = make_swiss_roll(
            n_samples=num_samples,
            noise=0.8)[0].astype(np.float32)[:, [0, 2]] / 7.5
        return torch.tensor(sample, device=self.device)

    def to(self, device):
        self.device = device


class LocationScatterSampler(Sampler):

    def __init__(self, base_sampler, weight, bias, device='cuda'):
        super().__init__(device=device)
        self.dim = base_sampler.d
        self.base_sampler = base_sampler
        self.transform = AffineTransform(weight.to(self.device), bias.to(self.device))
        self.mean = self.transform.weight @ base_sampler.mean + self.transform.bias
        self.cov = self.transform.weight @ self.base_sampler.cov @ self.transform.weight.T
        self.var = torch.sum(torch.diag(self.cov))

    def sample(self, num_samples=1):
        z = self.base_sampler.sample(num_samples=num_samples).to(self.device)
        z = self.transform(z)
        return z

    def to(self, device):
        super().to(device)
        self.to(device)

def get_linear_transport(mean1, cov1, mean2, cov2):
    sqrt_cov1 = sqrtm(cov1)
    sqrt_cov1_inv = torch.linalg.inv(sqrt_cov1)
    weight = sqrt_cov1_inv @ sqrtm(sqrt_cov1 @ cov2 @ sqrt_cov1) @ sqrt_cov1_inv
    bias = mean2 - weight @ mean1
    return weight, bias

def get_barycenter_cov(covs, alphas, max_iter=1000, tol=1e-10):

    def get_H(S):
        sqrt_S = sqrtm(S)
        sqrt_S_inv = torch.linalg.inv(sqrt_S)

        inner_sum = 0.
        for alpha, cov in zip(alphas, covs):
            inner_sum += alpha * sqrtm(sqrt_S @ cov @ sqrt_S)
        
        H = sqrt_S_inv @ inner_sum @ sqrt_S_inv
        return H

    def get_error(S_old, S_new):
        H_old = get_H(S_old)
        H_new = get_H(S_new)

        return (torch.trace(S_old) - 2*torch.trace(S_old @ H_old) - (torch.trace(S_new) - 2*torch.trace(S_new @ H_new))).item()

    S_old = torch.eye(covs[0].shape[0], device=covs[0].device)

    errs = []
    for _ in tqdm(range(max_iter)):
        H_old = get_H(S_old)
        S_new = H_old @ S_old @ H_old
        err = get_error(S_old, S_new)
        errs.append(err)
        if err < tol:
            break
        S_old = S_new
            
    return S_new

class LocationScatterBenchmark:

    def __init__(self, base_sampler, samplers, alphas, device='cuda'):
        self.device = device
        
        self.samplers = samplers
        self.num = len(self.samplers)

        # Compute barycenter mean
        self.mean = torch.zeros_like(samplers[0].mean)
        for alpha, sampler in zip(alphas, samplers):
            self.mean += alpha*sampler.mean
        
        # Compute barycenter covariance
        covs = [sampler.cov for sampler in samplers]
        self.cov = get_barycenter_cov(covs, alphas, max_iter=1000)
        self.var = torch.sum(torch.diag(self.cov))

        # Create barycenter sampler
        weight, bias = get_linear_transport(base_sampler.mean, base_sampler.cov, self.mean, self.cov)
        self.bar_sampler = LocationScatterSampler(base_sampler, weight, bias, device=self.device)

        # Create transport maps from samplers to barycenter
        self.bar_maps = []
        for sampler in samplers:
            weight, bias = get_linear_transport(sampler.mean, sampler.cov, self.mean, self.cov)
            self.bar_maps.append(AffineTransform(weight, bias))

        # Create transport maps from barycenter to samplers
        self.bar_maps_inv = []
        for sampler in samplers:
            weight, bias = get_linear_transport(self.mean, self.cov, sampler.mean, sampler.cov)
            self.bar_maps_inv.append(AffineTransform(weight, bias))