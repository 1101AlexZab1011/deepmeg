import torch
import torch.nn as nn
from . import BaseModel
from ..layers import AutoCov1D, Fourier
from ..utils.convtools import conviter


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


class TimeCompNet(BaseModel):
    """
    A compact convolutional neural network implementing spatial decomposition of entire signals, temporal filtering and temporal compression of each component.

    Args:
        n_channels (int): Number of channels in the input data. Input data shape supposed to be: (N batches, N channels, N times).
        n_latent (int): Number of latent features in the unmixing layer.
        n_times (int): Length of the time axis of the input data.
        filter_size (int): Size of the FIR filters in the temporal convolutional layer.
        pool_factor (int): Factor by which the output of the time-compression layers is compressed.
        n_outputs (int): Number of outputs of the final fully connected layer.

    Attributes:
        pool_factor (int): Factor to reduce the length of the time axis by selecting every n-th output.
        unmixing_layer (nn.Module): 1D depthwise convolutional layer for spatial unmixing (filtering) the input tensor.
        temp_conv (nn.Module): 1D temporal convolutional layer for temporal filtering (FIR-filter-like) the spatially filtered tensor.
        timesel_list (nn.ModuleList): A list of sequential layers used to estimate the time-compression parameters.
        fc_layer (nn.Module): Final fully connected layer for producing the outputs.

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
        self.timesel_list = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_times, (n_times//pool_factor)),
                torch.nn.Sigmoid()
            )
            for _ in range(n_latent)
        ])
        final_out_features = (n_times//pool_factor)*n_latent
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
        x = torch.stack(
            [
                timesel(x[:, i, :]) for i, timesel in enumerate(self.timesel_list)
            ],
            1
        )
        x = torch.flatten(x, 1)
        return self.fc_layer(x)


class SPIRIT(BaseModel):
    """
    A compact convolutional neural network implementing SPatial decomposition of entire signals,
        finite Impulse Response filtering and temporal Information Transfer estimation of each component (SPIRIT).

    Args:
        n_channels (int): Number of channels in the input data. Input data shape supposed to be: (N batches, N channels, N times).
        n_latent (int): Number of latent features in the unmixing layer.
        n_times (int): Length of the time axis of the input data.
        filter_size (int): Size of the FIR filters in the temporal convolutional layer.
        window_size (int): Size of the window used in the AutoCov1D layer.
        latent_dim (int): Dimension of the latent space.
        pool_factor (int): Factor by which the output of the time-compression layers is compressed.
        n_outputs (int): Number of outputs of the final fully connected layer.

    Attributes:
        unmixing_layer (nn.Conv1d): 1D depthwise convolutional layer for spatial unmixing (filtering) the input tensor.
        temp_conv (nn.Conv1d): 1D temporal convolutional layer for temporal filtering (FIR-filter-like) the spatially filtered tensor.
        timesel_list (nn.ModuleList): A list of sequential layers used to estimate the time-selection parameters.
        expander (nn.Conv1d): A 1D convolutional layer used to expand the time-selection parameters.
        windows (List[slice]): A list of slices used to split the input data into windows corresponding to their information transfer estimates.
        timecomp_list (nn.ModuleList): A list of sequential layers used to estimate the time-compression parameters.
        fc_layer (nn.Linear): Final fully connected layer for producing the outputs.

    """
    def __init__(
        self,
        n_channels: int,
        n_latent: int,
        n_times: int,
        filter_size: int,
        window_size: int,
        latent_dim: int,
        pool_factor: int,
        n_outputs: int
    ):
        super().__init__()
        self.unmixing_layer = nn.Conv1d(n_channels, n_latent, kernel_size=1, bias=True)
        self.temp_conv = nn.Conv1d(n_latent, n_latent, kernel_size=filter_size, bias=True, groups=n_latent, padding='same')
        self.timesel_list = nn.ModuleList([
            nn.Sequential(
                AutoCov1D(1, window_size, latent_dim, 0, window_size, bias=True),
            )
            for _ in range(n_latent)
        ])
        self.expander = nn.Conv1d(
            in_channels=n_latent,
            out_channels=n_latent,
            kernel_size=(2,),
            padding=1
        )
        self.windows = [win for win, _, _ in conviter((n_times-(window_size),), window_size, 0, window_size)]
        self.windows.append(slice(self.windows[-1].start + window_size, None))
        self.timecomp_list = nn.ModuleList([
            nn.Sequential(
                nn.Linear(
                    n_times,
                    (n_times//pool_factor),
                    bias=False
                ),
                nn.Dropout(.25),
                torch.nn.Sigmoid()
            )
            for i in range(n_latent)
        ])
        final_out_features = (n_times//pool_factor)*n_latent
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
        x_est = torch.stack(
            [
                timesel(x[:, i, :]) for i, timesel in enumerate(self.timesel_list)
            ],
            1
        )

        x_est = torch.squeeze(x_est, -1)
        x_est = self.expander(x_est)

        x_out = list()
        for i, window in enumerate(self.windows):
            x_out.append(
                x[:, :, window]
                *torch.unsqueeze(x_est[:, :, i], -1)
            )

        x = torch.cat(x_out, -1)

        x = torch.stack(
            [
                timesel(x[:, i, :]) for i, timesel in enumerate(self.timecomp_list)
            ],
            1
        )
        x = torch.flatten(x, 1)
        return self.fc_layer(x)
