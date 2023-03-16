import torch
import torch.nn as nn
from . import BaseModel


class HilbertNet(BaseModel):
    """
    A compact interpretable convolutional neural network performing
        spatial unmixing of input signals, applying finite-impulse-response filters to each component,
        and computing their envelopes.

    Args:
        n_channels (int): Number of channels in the input data. Input data shape supposed to be: (N batches, N channels, N times).
        n_latent (int): Number of latent features in the unmixing layer.
        n_times (int): Length of the time axis of the input data.
        filter_size (int): Size of the FIR filters in the temporal convolutional layer.
        pool_factor (int): Factor to reduce the length of the time axis by selecting every n-th output.
        n_outputs (int): Number of outputs of the final fully connected layer.

    Attributes:
        pool_factor (int): Factor to reduce the length of the time axis by selecting every n-th output.
        unmixing_layer (nn.Module): 1D depthwise convolutional layer for spatial unmixing (filtering) the input tensor.
        temp_conv (nn.Module): 1D temporal convolutional layer for temporal filtering (FIR-filter-like) the spatially filtered tensor.
        temp_conv_activation (nn.Module): Absolute value for temporally filtered components.
        env_conv (nn.Module): 1D temporal convolutional layer for envelopes computation (low-pass FIR-filter-like).
        fc_layer (nn.Module): Final fully connected layer for producing the outputs.

    References:
        [1] Petrosyan, A., Sinkin, M., Lebedev, M., & Ossadtchi, A. (2021).
            Decoding and interpreting cortical signals with a compact convolutional neural network.
            Journal of Neural Engineering, 18(2), 026019. https://doi.org/10.1088/1741-2552/abe20e
    """

    def __init__(self, n_channels, n_latent, n_times, filter_size, pool_factor, n_outputs):
        super().__init__()
        self.pool_factor = pool_factor
        self.unmixing_layer = nn.Conv1d(n_channels, n_latent, kernel_size=1)
        self.temp_conv = nn.Conv1d(n_latent, n_latent, kernel_size=filter_size, groups=n_latent, padding='same')
        self.temp_conv_activation = nn.LeakyReLU(-1)
        self.env_conv = nn.Conv1d(n_latent, n_latent, kernel_size=filter_size, groups=n_latent, padding='same')
        final_out_features = (n_times//pool_factor)*n_latent if not n_times%pool_factor else (n_times//pool_factor + 1)*n_latent
        self.fc_layer = nn.Linear(final_out_features, n_outputs)


    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, n_channels, n_times).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, n_outputs).
        """
        x = self.unmixing_layer(x)
        x = self.temp_conv(x)
        x = self.temp_conv_activation(x)
        x = self.env_conv(x)
        x = x[:, :, ::self.pool_factor].contiguous()
        x = torch.flatten(x, 1)
        return self.fc_layer(x)


class LFCNN(BaseModel):
    """
    A compact convolutional neural network for the Linear Finite-impulse-response Convolutional Neural Network (LF-CNN) [1].

    Args:
        n_channels (int): Number of channels in the input data. Input data shape supposed to be: (N batches, N channels, N times).
        n_latent (int): Number of latent features in the unmixing layer.
        n_times (int): Length of the time axis of the input data.
        filter_size (int): Size of the FIR filters in the temporal convolutional layer.
        pool_factor (int): Factor to reduce the length of the time axis by selecting every n-th output.
        n_outputs (int): Number of outputs of the final fully connected layer.

    Attributes:
        pool_factor (int): Factor to reduce the length of the time axis by selecting every n-th output.
        unmixing_layer (nn.Module): 1D depthwise convolutional layer for spatial unmixing (filtering) the input tensor.
        temp_conv (nn.Module): 1D temporal convolutional layer for temporal filtering (FIR-filter-like) the spatially filtered tensor.
        fc_layer (nn.Module): Final fully connected layer for producing the outputs.

    References:
        [1] I. Zubarev, et al., Adaptive neural network classifier for
        decoding MEG signals. Neuroimage. (2019) May 4;197:425-434
    """

    def __init__(
        self,
        n_channels: int,
        n_latent: int,
        n_times: int,
        filter_size: int,
        pool_factor: int,
        n_outputs: int
    ):
        super().__init__()
        self.pool_factor = pool_factor
        self.unmixing_layer = nn.Conv1d(n_channels, n_latent, kernel_size=1, bias=True)
        self.temp_conv = nn.Conv1d(n_latent, n_latent, kernel_size=filter_size, bias=True, groups=n_latent, padding='same')
        final_out_features = (n_times//pool_factor)*n_latent if not n_times%pool_factor else (n_times//pool_factor + 1)*n_latent
        self.fc_layer = nn.Linear(final_out_features, n_outputs)


    def forward(self, x: torch.Tensor):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, n_channels, n_times).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, n_outputs).
        """
        x = self.unmixing_layer(x)
        x = self.temp_conv(x)
        x = x[:, :, ::self.pool_factor].contiguous()
        x = torch.flatten(x, 1)
        return self.fc_layer(x)

