import torch
import torch.nn as nn
from normflows import NormalizingFlow
from .bases import SXBase, CategoricalBase, GaussianBase

class SXMultiscaleFlow(nn.Module):
    """ SX Normalizing Flow model with multiscale architecture.

    """

    def __init__(self, z_bases, s_base, flows, merges):
        """Constructor

        Args:

          z_bases: List of Z base distributions (one per multi-scale level)
          s_base : Base distribution for random variable S
          flows: List of flows for each level
          merges: List of merge/split operations (forward pass must do merge)
        """
        super().__init__()
        
        self.z_bases = nn.ModuleList(z_bases)
        self.s_base = s_base
        if not isinstance(self.s_base, CategoricalBase):
            raise ValueError('S base of SX multi-scale flow must be Categorical.')
        self.flows = torch.nn.ModuleList([nn.ModuleList(flow) for flow in flows])
        self.merges = torch.nn.ModuleList(merges)
        

    def forward_kld(self, x, s=None):
        """Estimates forward KL divergence, see [arXiv 1912.02762](https://arxiv.org/abs/1912.02762)

        Args:
          x: Batch sampled from target distribution
          y: Batch of classes to condition on, if applicable

        Returns:
          Estimate of forward KL divergence averaged over batch
        """
        return -torch.mean(self.log_prob(x, s))

    def forward(self, z, s):
        """Forward of the model from latent space.

        """
        x, _ = self.forward_and_log_det(z, s)
        return x

    def forward_and_log_det(self, z, s):
        """Get observed variable x from list of latent variables z and s.

        Args:
            z: List of latent variables

        Returns:
            Observed variable x, log determinant of Jacobian
        """
        log_det = torch.zeros(len(z[0]), dtype=z[0].dtype, device=z[0].device)
        for i in range(len(self.z_bases)):
            if i == 0:
                z_ = z[0]
            else:
                z_, log_det_ = self.merges[i - 1]([z_, z[i]])
                log_det += log_det_
            for flow in self.flows[i]:
                z_, log_det_ = flow(z_, s=s)
                log_det += log_det_
        return z_, log_det

    def inverse_and_log_det(self, x, s):
        """Get latent variable z from observed variable x, s

        Args:
            x: Observed variable

        Returns:
            List of latent variables z, log determinant of Jacobian
        """
        log_det = torch.zeros(len(x), dtype=x.dtype, device=x.device)
        z = [None] * len(self.z_bases)
        for i in range(len(self.z_bases) - 1, -1, -1):
            for flow in reversed(self.flows[i]):
                x, log_det_ = flow.inverse(x, s=s)
                log_det += log_det_
            if i == 0:
                z[i] = x
            else:
                [x, z[i]], log_det_ = self.merges[i - 1].inverse(x)
                log_det += log_det_
        return z, log_det

    def sample_latent(self, num_samples=1, temperature=None):
        """Sample latent variables z and s.

        Args:
          num_samples: Number of samples to draw
          temperature: Temperature parameter for temp annealed sampling

        Returns:
          z : list of torch.tensors 
          s : torch.tensor
        """
        if temperature is not None:
            self.set_temperature(temperature)

       
        s = self.s_base.sample(num_samples)
        z = []
        for z_base in self.z_bases:
            z_, _ = z_base(num_samples)
            z.append(z_)
            
        if temperature is not None:
            self.reset_temperature()

        return z, s

    
    def sample(self, num_samples=1, s=None, temperature=None, return_sz=False):
        """Samples from flow-based approximate distribution

        Args:
          num_samples: Number of samples to draw
          s : torch.tensor, size (num_samples, self.S_dim)
              Value of S.
          temperature: Temperature parameter for temp annealed sampling

        Returns:
          Samples, log probability
        """
        if temperature is not None:
            self.set_temperature(temperature)

        if s is None:
            s, log_q = self.s_base(num_samples)
        else:
            log_q = self.s_base.log_prob(s)
        
        for i, z_base in enumerate(self.z_bases):
            
            if i == 0:
                z_, log_q_ = z_base(num_samples)
                log_q += log_q_
                z = z_
            else:
                z_, log_q_ = z_base(num_samples)
                log_q += log_q_
                z, log_det = self.merges[i - 1]([z, z_])
                log_q -= log_det
                
            for flow in self.flows[i]:
                z, log_det = flow(z, s=s)
                log_q -= log_det
                
        if temperature is not None:
            self.reset_temperature()
        x = z
        return x, log_q

    def log_prob(self, x, s):
        """Get log probability for batch

        Args:
          x: Batch

        Returns:
          log probability
        """
        log_q = self.s_base.log_prob(s)
        z = x
        for i in range(len(self.z_bases) - 1, -1, -1):
            for j in range(len(self.flows[i]) - 1, -1, -1):
                z, log_det = self.flows[i][j].inverse(z, s)
                log_q += log_det
            
            if i > 0:
                [z, z_], log_det = self.merges[i - 1].inverse(z)
                log_q += log_det
            else:
                z_ = z
                
            log_q += self.z_bases[i].log_prob(z_)
                
        return log_q

    def save(self, path):
        """Save state dict of model

        Args:
          path: Path including filename where to save model
        """
        torch.save(self.state_dict(), path)

    def load(self, path):
        """Load model from state dict

        Args:
          path: Path including filename where to load model from
        """
        self.load_state_dict(torch.load(path))

    def set_temperature(self, temperature):
        """Set temperature for temperature a annealed sampling

        Args:
          temperature: Temperature parameter
        """
        for z_base in self.z_bases:
            if hasattr(z_base, "temperature"):
                z_base.temperature = temperature
            else:
                raise NotImplementedError(
                    "One base function does not "
                    "support temperature annealed sampling"
                )

    def reset_temperature(self):
        """
        Set temperature values of base distributions back to None
        """
        self.set_temperature(None)



class SXNormalizingFlow(SXMultiscaleFlow):
    """Custom Normalizing Flow model for training on (S, X) datasets.

    Arguments:
    ----------
    base : SZBase object
        Base for latent variables (S, Z).
    flows : list of Flow objects
        Flows.

    Attributes:
    -----------
    base : SZBase object
        Base for latent variables (S, Z).

    Methods:
    --------
    inverse_conditional(X, s=None)
        Compute model inverse on X sample conditionally on S=s.
    sample_conditional(num_samples=1, s=None)
        Sample X conditionally on S=s.
    bar_map(Z)
        Transport latent sample Z to barycenter.
    bar_sample(num_samples=1)
        Sample from the barycenter.
    bar_map_conditional(self, X, s=None)
        Transport sample X conditional on S=s into barycenter.        
    """

    def __init__(self, z_bases, s_base, flows, merges):
        super().__init__(z_bases, s_base, flows, merges)
        
    def bar(self, z):
        """Transport latent sample z to barycenter.

        The barycenter is the map: z -> sum_s P(S=s) f(s, z).

        Arguments:
        ---------
        z : list of torch.tensors

        Returns:
        --------
        x : torch.tensor, size (z[0].shape[0], dim of output)
        """
  
        s_vals = torch.arange(self.s_base.p.numel()).to(z[0].device).reshape(-1, 1)
        s = torch.tile(self.s_base.encode(s_vals), (z[0].shape[0], 1))
        
        z_aug = []
        for z_ in z:
            if z_.ndim == 2:
                z_aug += [torch.stack(tuple(z_ for _ in s_vals), dim=1).view(z_.shape[0]*s_vals.numel(), z_.shape[1])]
            elif z_.ndim == 4:
                z_aug += [torch.stack(tuple(z_ for _ in s_vals), dim=1).view(z_.shape[0]*s_vals.numel(), z_.shape[1], z_.shape[2], z_.shape[3])]
                
        
        x = self.forward(z_aug, s)

        if z[0].ndim == 2:
            # Dimension of output
            d = sum([z_base.d for z_base in self.z_bases])
            # Compute expectation E[f(S,Z)|Z] = sum_s f(s, Z)
            bar = self.s_base.p @ x.reshape(z[0].shape[0], s_vals.numel(), d)
        elif z[0].ndim == 4:
            #bar = self.s_base.p @ x.reshape(z[0].shape[0], s_vals.numel(), x.shape[1], x.shape[2], x.shape[3])
            bar = self.s_base.p @ x.reshape(z[0].shape[0], s_vals.numel(), -1)
            bar = bar.reshape(z[0].shape[0], x.shape[1], x.shape[2], x.shape[3])
        
        return bar

    def bar_sample(self, num_samples=1, temperature=None):
        """Sample from the barycenter.

        Arguments:
        ----------
        num_samples : int (default: 1)
            Number of samples

        Returns:
        --------
        X : torch.tensor, size (num_samples, self.base.Z_base.dim)
            Barycenter sample.
        """

        z, _ = self.sample_latent(num_samples=num_samples, temperature=temperature)
        bar = self.bar(z)

        return bar
    
    
    def bar_maps(self, x, s=None):
        """Transport conditional sample of x given s into barycenter.

        Arguments:
        ----------
        x : torch.tensor, size (num_samples, self.base.Z_base.dim)
            Conditional sample of X given S.
        s : int
            Value of S on which the sample X is conditioned.
        """

        z, _ = self.inverse_and_log_det(x, s)
        bar = self.bar(z)
        
        return bar