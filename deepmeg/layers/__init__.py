import torch.nn as nn
import torch.nn.init as init
import math
import torch
from deepmeg.convtools import compute_output_shape, conviter

class AutoCov1D(nn.Module):
    """Autocovariance 1D layer.

    A PyTorch implementation of a 1D autocovariance layer, taking two windows of a 1D sequence and computing
    the covariance between their projections onto latnet subspace for each neuron of the layer.

    Args:
        out_channels (int): The number of neurons in the layer, determining the number of output channels.
        window_size (int): The size of the windows to extract from the 1D sequence.
        latent_dim (int): The dimension of the latent space to project the window
            into before computing the autocovariance. If not provided, the latent_dim is set to window_size // 2.
        overlap (int): The number of overlapping timesteps between the two windows. Defaults to 0.
        stride (int): The stride to use when sliding the windows along the 1D sequence. Defaults to 1.
        padding (int): The number of padding timesteps to add to both sides of the 1D sequence. Defaults to 0.
        bias (bool): Whether to include a bias term for each neuron in the layer. Defaults to True.

    Attributes:
        weight (nn.Parameter): The weight matrix of shape (window_size, latent_dim, out_channels) that projects the windows into the latent space.
        bias (nn.Parameter): The bias vector of shape (out_channels), added to each autocovariance computation.

    """
    def __init__(self, out_channels, window_size, latent_dim = None, overlap=0, stride=1, padding=0, bias=True):
        super().__init__()
        self.window_size = window_size
        self.latent_dim = latent_dim if latent_dim is not None else window_size // 2
        self.overlap = overlap
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.empty(self.window_size, self.latent_dim, out_channels))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the layer parameters.
        Uses the Kaiming initialization method for the weight matrix and a uniform initialization method for the bias vector.
        """
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)

    def forward(self, X):
        """Forward pass of the model.

        Args:
            input_ (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        # inputs shape (N_batch, width) ~ (N_batch, n_times)
        X_part1 = X[:, :-(self.window_size - self.overlap)] # shape: (batches, times)
        X_part2 = X[:, self.window_size - self.overlap:,]

        assert X_part1.shape == X_part2.shape, f'Windowed sequence is corrupted: {X_part1.shape} vs {X_part2.shape}'

        convargs = ((X_part1.shape[1], X_part1.shape[0]), self.window_size, self.padding, self.stride)
        n_windows = compute_output_shape(*convargs)[0]
        out_tensor = torch.zeros(X.shape[0], n_windows, self.weight.shape[-1]) # ~ batches x n_windows x out_channels

        # iter over neurons
        for n_neuron in range(self.weight.shape[-1]):
            for p, q, k in conviter(*convargs):
                # p shape: n_times, ...
                # q shape: n_windows, 0
                # k shape: window_size, ...
                time_range = p[0]
                window_range = k[0]
                n_window = q[0]

                window1 = X_part1[:, time_range]
                window2 = X_part2[:, time_range]
                weight = self.weight[window_range, :, n_neuron]

                proj1 = torch.matmul(window1, weight) # (N batch x time_range) @ (window_range x latent_dim x n_neuron) = (N batch x latent_dim)
                proj2 = torch.matmul(window2, weight)
                cov = torch.mean(
                    (proj1 - torch.mean(proj1, 1, keepdim=True))*
                    (proj2 - torch.mean(proj2, 1, keepdim=True)),
                    1,
                    keepdim=True
                ) # ~ (N_batch, 1)
                out_tensor[:, n_window, n_neuron] = torch.squeeze(cov, -1) + self.bias[n_neuron] if self.bias is not None else torch.squeeze(cov, -1)

        return out_tensor