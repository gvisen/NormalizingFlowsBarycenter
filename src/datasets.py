import torch
from torch.utils.data import Dataset, DataLoader, Subset
from .bases import CategoricalBase
import random

class Sampler():

    def __init__(self):
        self.dim = None

    def sample(self, num_samples=1):
        raise NotImplementedError


class MNISTSampler(Sampler):
    def __init__(self, dataset, digit_idx, device='cpu'):

        self.dim = (1, 32, 32)
        self.device = device
        indices = torch.where(dataset.targets == digit_idx)[0]
        self.loader = DataLoader(Subset(dataset, indices), batch_size=len(dataset), num_workers=40)
        
        with torch.no_grad():
            self.dataset = torch.cat([X for (X, _) in self.loader])

    def sample(self, num_samples=10):
        ind = random.choices(range(len(self.dataset)), k=num_samples)
        with torch.no_grad():
            batch = self.dataset[ind].clone().to(self.device).float()
        return batch

        

class SXDataset(Dataset):
    """(S, X) dataset with S categorial and X arbitrary.

    Arguments:
    ----------
    X_samplers : list of Sampler objects.
        X_samplers[i] is the distribution of X given S=i.
        Each sampler object must implement a `sample(num_samples=1)` method
        and a `dim` attribute.
    S_base : CategoricalBase object.
        Base of categorical variable S.
    num_samples : int
        Number of samples.

    Attributes:
    -----------
    X_samplers : list of Sampler objects.
        X_samplers[i] is the distribution of X given S=i.
        Each sampler object must implement a `sample(num_samples=1)` method
        and a `dim` attribute.
    S_base : CategoricalBase object.
        Base of categorical variable S.
    X_dim : int
        Dimension of X.
    S_dim : int
        Dimension of S.

    Methods:
    --------
    Dataset methods.
    """

    def __init__(self, X_samplers, S_base, num_samples=1):
        super().__init__()

        assert isinstance(S_base, CategoricalBase)
        
        self.S_base = S_base
        self.X_samplers = X_samplers
        
        S_idx, S = self.S_base.sample(num_samples=num_samples, return_idx=True)
        S_idx = S_idx.squeeze()

        if isinstance(X_samplers[0].dim , int):
            self.X_dim = (X_samplers[0].dim,)
        else:
            self.X_dim = X_samplers[0].dim

        self.S_dim = self.S_base.dim
        
        X = torch.empty((num_samples,) + self.X_dim, device=S.device)

        for S_val in range(self.S_base.p.numel()):
            if (S_idx == S_val).any():
                X[S_idx == S_val, :] = self.X_samplers[S_val].sample(torch.sum(S_idx==S_val).item()).to(X.device)

        self.S = S
        self.X = X
        
    def __getitem__(self, idx):
        return self.S[idx], self.X[idx]
    
    def __len__(self):
        return self.S.shape[0]
    
    def to(self, device):
        self.S = self.S.to(device)
        self.X = self.X.to(device)