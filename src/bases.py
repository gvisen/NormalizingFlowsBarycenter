import torch
from normflows.distributions import BaseDistribution
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.categorical import Categorical
from torch.distributions.uniform import Uniform

class SXBase(BaseDistribution):
    """Base for latent variables (S, Z).

    Arguments:
    ----------
    S_base : BaseDistribution object
        Base distribution of S.
    Z_base : BaseDistribution object
        Base distribution of Z.

    Attributes:
    -----------
    S_base : BaseDistribution object
        Base distribution of S.
    Z_base : BaseDistribution object
        Base distribution of Z.

    Methods:
    --------
    forward(num_samples=1)
        Returns (S,Z) sample from base distribution together with its log-probability.
    log_prob(SZ)
        Computes log-probability of sample `SZ`.
    sample(num_samples=1)
        Samples from base distribution.
    """
    
    def __init__(self, S_base, Z_base):
        super().__init__()
        
        self.S_base = S_base
        self.Z_base = Z_base
    
    def forward(self, num_samples=1):
        """Samples from base distribution and calculates log probability.

        Arguments:
        ----------
        num_samples : int
            Number of samples to draw from the distriubtion

        Returns:
        --------
        s : torch.tensor, size (num_samples, self.S_base.dim)
            Samples drawn from the S distribution
        z : torch.tensor, size (num_samples, self.Z_base.dim)
            Samples drawn from the Z distribution
        log_p_sz : torch.tensor size (num_samples,)
            Log-probability of joint samples (S,Z)
        """

        s = self.S_base.sample(num_samples=num_samples)
        log_p_s = self.S_base.log_prob(s)

        z = self.Z_base.sample(num_samples=num_samples)
        log_p_z = self.Z_base.log_prob(z)
        
        log_p_sz = log_p_s + log_p_z
        
        return (s, z), log_p_sz

    def log_prob(self, sz):
        """Computes log-probability of sample sz.

        Arguments:
        ----------
        sz : tuple of two torch.tensors of size (num_samples, self.S_base.dim) and (num_samples, self.Z_base.dim)
            Sample to compute log-probability.

        Returns:
        --------
        log_p_sz : torch.tensor, size (num_samples,)
            Log-probability of sample.
        """

        s, z = sz

        log_p_s = self.S_base.log_prob(s)
        log_p_z = self.Z_base.log_prob(z)
        log_p_sz = log_p_s + log_p_z
        
        return log_p_sz


class CategoricalBase(BaseDistribution):
    """Base for categorical variable taking values in {0, 1, ..., d-1}.
    
    The random variable can be one-hot encoded.

    Arguments:
    ---------
    p : torch.tensor, size (d,)
        Array of probabilities for values {0, 1, ..., d-1}.
    one_hot_encoding : bool (default: True)
        If True, the categorical variable is one-hot encoded

    Attributes:
    -----------
    p : torch.tensor, size (d,)
        Array of probabilities for values {0, 1, ..., self.dim}.
    one_hot_encoding : bool (default: True)
        If True, the categorical variable is one-hot encoded
    dim : int
        Dimension of variable (d if one_hot_encoding is True, 1 otherwise)

    Methods:
    --------
    encode(z, return_idx=False)
        One-hot encode categorical values in sample `z`, if `one_hot_encoding` is True.
    decode(z)
        One-hot decode the one-hot encoded sample `z`, if `one_hot_encoding` is True.
    forward(num_samples=1)
        Returns sample from base distribution together with its log-probability.
    log_prob(z)
        Computes log-probability of sample `z`.
    sample(num_samples=1, return_index=False)
        Samples from base distribution.
    """
    

    def __init__(self, p, one_hot_encoded=True):
        super().__init__()
        
        # Define buffers (untrainable parameters) using `register_buffer` for torch <=2.4.
        #self.p = torch.nn.parameter.Buffer(p)
        self.register_buffer('p', p)
        self.one_hot_encoded = one_hot_encoded
        self.dim = self.p.numel() if self.one_hot_encoded else 1

    def encode(self, z, return_idx=False):
        """One-hot encode categorical values in sample `z`, if `one_hot_encoding` is True.

        Arguments:
        ----------
        z : torch.tensor, size (num_samples, 1)
            Sample of categorical variable, i.e. with values in {0, 1, ..., d-1}.

        Returns:
        --------
        z_encoded : torch.tensor, size (num_samples, d)
            One-hot encoded sample if `one_hot_encoding` is True, `z` otherwise.
        """
        
        if self.one_hot_encoded:
            z_encoded = torch.zeros((z.shape[0], self.dim), device=z.device).scatter(1, z, 1)
        else:
            z_encoded = z/len(self.p)
        
        if return_idx:
            return z.to(torch.int), z_encoded
        else:
            return z_encoded
        
    def decode(self, z):

        if self.one_hot_encoded:
            return torch.nonzero(z)[:, 1].unsqueeze(1)
        else:
            return (z*len(self.p)).to(torch.int)

    def forward(self, num_samples=1):

        z = self.encode(Categorical(self.p).sample((num_samples, 1)))
        log_p = self.log_prob(z)
        
        return z, log_p

    def sample(self, num_samples=1, return_idx=False):
        """Sample from base distribution.

        Arguments:
        ----------
        num_samples : int
            Number of samples.
        return_idx : bool (default: false)
            If true, return categorical values of 
        """

        if return_idx:
            z_idx, z = self.encode(Categorical(self.p).sample((num_samples, 1)), return_idx=return_idx)
            return z_idx, z
        else:
            z = self.encode(Categorical(self.p).sample((num_samples, 1)), return_idx=return_idx)
            return z
        

    def log_prob(self, z):
        
        log_p = Categorical(self.p).log_prob(self.decode(z)).squeeze()    
        
        return log_p


class GaussianBase(BaseDistribution):
    """Base for multivariate standard Gaussian variable.
    
    Arguments:
    ---------
    dim : int
        Dimension of variable.
        
    Attributes:
    -----------
    dim : int
    loc : torch.tensor, size (dim,)
        Mean vector (zero vector).
    cov : torch.tensor, size (dim, dim)
        Covariance matrix (identity matrix).
    
    Methods:
    --------
    forward(num_samples=1)
        Returns sample from base distribution together with its log-probability.
    log_prob(z)
        Computes log-probability of sample `z`.
    sample(num_samples=1)
        Samples from base distribution.
    """
    
    def __init__(self, dim):
        super().__init__()
        
        self.dim = dim
        # Define buffers (untrainable parameters) using `register_buffer` for torch <=2.4.
        #self.loc = torch.nn.parameter.Buffer(torch.zeros(self.dim))
        #self.cov = torch.nn.parameter.Buffer(torch.eye(self.dim))
        self.register_buffer('loc', torch.zeros(self.dim))
        self.register_buffer('cov', torch.eye(self.dim))
        self.temperature = None


    def forward(self, num_samples=1):

        z = MultivariateNormal(self.loc, covariance_matrix=self.cov).sample((num_samples,))
        log_p = self.log_prob(z)
        
        return z, log_p

    def log_prob(self, z):
        
        log_p = -0.5*self.dim*torch.log(torch.tensor(2*torch.pi, device=z.device)) - 0.5*torch.sum(z*z, dim=1)
        
        return log_p


class UniformBase(BaseDistribution):
    """Base for univariate uniform variable taking values in [low, high].

    Arguments:
    ---------
    low : float
        Lower bound of [low, high]
    high : float
        Upper bound of [low, high]

    Attributes:
    -----------
    low : torch.tensor
        Lower bound of [low, high]
    high : torch.tensor
        Upper bound of [low, high]

    Methods:
    --------
    forward(num_samples=1)
        Returns sample from base distribution together with its log-probability.
    log_prob(z)
        Computes log-probability of sample `z`.
    sample(num_samples=1, return_index=False)
        Samples from base distribution.
    """
    

    def __init__(self, low=None, high=None):
        super().__init__()
        
        low = torch.tensor(0.) if low is None else torch.tensor(low)
        high = torch.tensor(1.) if high is None else torch.tensor(high)
        self.register_buffer('low', low)
        self.register_buffer('high', high)
        self.dim = 1

    def forward(self, num_samples=1):

        z = Uniform(self.low, self.high).sample((num_samples, 1))
        log_p = self.log_prob(z)
        
        return z, log_p

    def sample(self, num_samples=1, return_idx=False):
        """Sample from base distribution.

        Arguments:
        ----------
        num_samples : int
            Number of samples.
        return_idx : bool (default: false)
            If true, return categorical values of 
        """
        z = Uniform(self.low, self.high).sample((num_samples, 1))

        return z

    def log_prob(self, z):
        
        log_p = Uniform(self.low, self.high).log_prob(z).squeeze()
        
        return log_p