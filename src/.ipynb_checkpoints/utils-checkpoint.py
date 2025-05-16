import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

class AffineTransform(torch.nn.Module):

    def __init__(self, weight, bias):
        super().__init__()
        self.register_buffer('weight', weight)
        self.register_buffer('bias', bias)

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight, bias=self.bias)

def sqrtm(A):
    L, V = torch.linalg.eig(A)
    #return torch.real(V @ torch.diag(torch.sqrt(L)) @ V.T)
    sqrt_A = V @ torch.diag(torch.sqrt(L)) @ V.T
    if torch.max(torch.abs(torch.imag(sqrt_A))) > 1e-10:
        print('Matrix sqrt is not real!')
    return torch.real(sqrt_A)

class ConditionalMLP(nn.Module):
    """Conditional MLP.
    """

    def __init__(self, layers, context_dim=None, init_zeros=True, activation='relu'):
        super().__init__()

        self.model = torch.nn.Sequential()

        if context_dim is None:
            raise ValueError('Specify context dimension.')
        
        # add input layer and hidden layers
        layers[0] += context_dim # add context to input
        for n1, n2 in zip(layers[:-2], layers[1:-1]):
            layer = nn.Linear(n1, n2)
            self.model.append(layer)
            # self.model.append(torch.nn.BatchNorm1d(n2))
            if activation == 'relu':
                self.model.append(nn.ReLU())
            elif activation == 'leaky_relu':
                self.model.append(nn.LeakyReLU(negative_slope=0.05))
            elif activation == 'tanh':
                self.model.append(nn.Tanh())
            else:
                raise ValueError(f'Unknown activation function: {activation}')
            
        
        # add linear output layer
        layer = nn.Linear(layers[-2], layers[-1])       
        if init_zeros:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        self.model.append(layer)

        # layer for affine residual connection
        self.reslayer = nn.Linear(context_dim, layers[-1])
        nn.init.zeros_(self.reslayer.weight)
        nn.init.zeros_(self.reslayer.bias)

    def forward(self, x, context=None):
        # standard MLP
        # return self.model(x)

        # affine residual
        return self.reslayer(context) + self.model(torch.cat((x, context), dim=1))

class ConditionalConvNet2d(nn.Module):
    """
    Convolutional Neural Network with leaky ReLU nonlinearities
    """

    def __init__(
        self,
        channels,
        kernel_size,
        context_dim=None,
        leaky=0.0,
        init_zeros=True,
        actnorm=False,
        weight_std=None,
    ):
        """Constructor

        Args:
          channels: List of channels of conv layers, first entry is in_channels
          kernel_size: List of kernel sizes, same for height and width
          leaky: Leaky part of ReLU
          init_zeros: Flag whether last layer shall be initialized with zeros
          scale_output: Flag whether to scale output with a log scale parameter
          logscale_factor: Constant factor to be multiplied to log scaling
          actnorm: Flag whether activation normalization shall be done after each conv layer except output
          weight_std: Fixed std used to initialize every layer
        """
        super().__init__()
        # Build network
        net = nn.ModuleList([])
        
        for i in range(len(kernel_size) - 1):
            conv = nn.Conv2d(
                channels[i] + context_dim if i==0 else channels[i], # add context dimension to input channels
                channels[i + 1],
                kernel_size[i],
                padding=kernel_size[i] // 2,
                bias=(not actnorm),
            )
            if weight_std is not None:
                conv.weight.data.normal_(mean=0.0, std=weight_std)
            net.append(conv)
            if actnorm:
                net.append(utils.ActNorm((channels[i + 1],) + (1, 1)))
            net.append(nn.LeakyReLU(leaky))
        i = len(kernel_size)
        net.append(
            nn.Conv2d(
                channels[i - 1],
                channels[i],
                kernel_size[i - 1],
                padding=kernel_size[i - 1] // 2,
            )
        )
        if init_zeros:
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)
        self.net = nn.Sequential(*net)

    def forward(self, x, context=None):
        # Transform (N, C) context into (N, C, H, W) context image
        context_img = context.reshape((context.shape[0], context.shape[1], 1, 1))
        context_img = context_img.repeat(1, 1, x.shape[2], x.shape[3])

        # Run CNN on input image and context image, concatenated along channel dimension
        return self.net(torch.concat((x, context_img), dim=1))

     

def check_mem(device, msg=None):    
    # Clean GPU cache
    if torch.cuda.is_available() and device.type == 'cuda':
        torch.cuda.empty_cache()
    else:
        print('Using CPU: no memory check.')
        return None
    
    print('------------------------------------------------------------')
    if msg is not None:
        print('MEMORY CHECK: ' + msg)
    else:
        print('MEMORY CHECK')     
    
    free_memory, _ = torch.cuda.mem_get_info(device)
    allocated_memory = torch.cuda.memory_allocated()
    print(f'Current device: {device}')
    print(f'Free memory: {free_memory*1e-9:.2f} GB')
    print(f'Allocated memory: {allocated_memory*1e-9:.2f} GB')
    print('------------------------------------------------------------')