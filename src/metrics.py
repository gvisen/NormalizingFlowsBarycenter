import torch

from .utils import sqrtm

def get_L2_UVP(benchmark, model, alphas, batch_size=1024, device=None):
    
    assert (benchmark.bar_maps is not None) and (benchmark.bar_sampler is not None)
    
    L2_UVP = torch.zeros(benchmark.num, device=device)
    
    for i in range(benchmark.num):
        
        with torch.no_grad():
            
            x = benchmark.samplers[i].sample(batch_size)
            
            s = model.s_base.encode(i*torch.ones((x.shape[0], 1), dtype=torch.int64, device=x.device))
            bar_pred = model.bar_maps(x, s=s)
            
            bar_true = benchmark.bar_maps[i](x)
            
            L2_UVP[i] = 100 * (((bar_pred - bar_true) ** 2).sum(dim=1).mean() / benchmark.bar_sampler.var).item()

    return torch.inner(alphas, L2_UVP).item()

def get_BW_metric(mean1, mean2, cov1, cov2):
    """Wasserstein metric btw two Gaussian distributions.
    """
    sqrt_cov1 = sqrtm(cov1)
    sqrt_cov1_inv = torch.linalg.inv(sqrt_cov1)    
    mean_term = torch.sum((mean1 - mean2)**2)
    cov_term = torch.trace(cov1 + cov2 - 2*sqrtm(sqrt_cov1 @ cov2 @ sqrt_cov1)) 
    return  mean_term + cov_term


def get_BW2_UVP(benchmark, model, num_samples=100000):
    """Bures-Wasserstein UVP metric between benchmark barycenter and model barycenter.
    """
    
    assert (benchmark.bar_sampler is not None)

    with torch.no_grad():

        # Compute empirical mean and covariance of model barycenter
        batch = model.bar_sample(num_samples=num_samples, temperature=None)
        mean = torch.mean(batch, dim=0)
        cov = torch.cov(batch.T)

        # Compute Bures-Wasserstein metric
        BW_metric = get_BW_metric(benchmark.bar_sampler.mean, mean, benchmark.bar_sampler.cov, cov) 

        # Compute BW2-UVP metric
        BW2_UVP = 100*(BW_metric / benchmark.bar_sampler.var)
    return BW2_UVP.item()