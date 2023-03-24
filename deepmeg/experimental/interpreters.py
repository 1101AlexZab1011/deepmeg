import matplotlib
import sys
from copy import deepcopy
from tqdm import tqdm
from ..utils.printout import nostdout
from ..utils.colors import generate_cmap
from .models import SPIRIT, TimeCompNet
from ..interpreters import LFCNNInterpreter
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import scipy as sp
import mne
import matplotlib.pyplot as plt


class SPIRITInterpreter(LFCNNInterpreter):
    """
        Initialize SPIRITInterpreter object.

        Parameters:
            model (SPIRIT): Trained SPIRIT model.
            dataset (Dataset): Data used to test the SPIRIT model.
            info (mne.Info): Information about recordings, typically contained in the "info" property of the corresponding instance (E.g. epochs.info).
    """
    def __init__(self, model: SPIRIT, dataset: Dataset, info: mne.Info):
        super().__init__(model, dataset, info)
        self._temporal_patterns = None
        self._latent_sources_weighted = None

    @torch.no_grad()
    def compute_temporal(self):

        loader = DataLoader(self.dataset, len(self.dataset))
        n_latent = self.model.unmixing_layer.weight.shape[0]
        X, y = next(iter(loader))
        x = self.model.unmixing_layer(X)
        x = self.model.temp_conv(x)
        x_est = torch.stack(
            [
                timesel(x[:, i, :]) for i, timesel in enumerate(self.model.timesel_list)
            ],
            1
        )

        x_est = torch.squeeze(x_est, -1)
        x_est = self.model.expander(x_est)

        x_out = list()
        for i, window in enumerate(self.model.windows):
            x_out.append(
                x[:, :, window]
                *torch.unsqueeze(x_est[:, :, i], -1)
            )

        x = torch.cat(x_out, -1)
        temporal_patterns = torch.abs(x_est).numpy()
        latent_sources_weighted = x.numpy()
        return temporal_patterns, latent_sources_weighted

    def __validate_temporal(self):
        """
        Validates the tempwise loss by computing it if it has not been computed previously.

        The `_tempwise_loss` attribute is set to the result of the `compute_tempwise_loss` method.

        """
        if self._temporal_patterns is None or self._latent_sources_weighted is None:
            self._temporal_patterns, self._latent_sources_weighted = self.compute_temporal()

    @property
    def latent_sources_weighted(self):
        """
        Get the tempwise loss.

        Returns:
            List[List[float]]: The tempwise loss.
        """
        self.__validate_temporal()
        return self._latent_sources_weighted

    @property
    def temporal_patterns(self):
        """
        Get the tempwise loss.

        Returns:
            List[List[float]]: The tempwise loss.
        """
        self.__validate_temporal()
        return self._temporal_patterns

    def plot_branch(
        self,
        branch_num: int,
        spec_plot_elems: list[str] = ['input', 'output', 'response'],
        title: str = None
    ) -> matplotlib.figure.Figure:
        """
        Plot the branchwise information for a specific branch of the model.

        Parameters:
        branch_num (int): the branch number to plot (order of branches is determined by `branchwise_loss`).
        spec_plot_elems (List[str]): a list of plot elements to include in the spectrum plot.
        title (str): optional title for the plot.

        Returns:
        matplotlib.figure.Figure: the plot.

        """
        info = deepcopy(self.info)
        info.__setstate__(dict(_unlocked=True))
        info['sfreq'] = 1.
        order = np.argsort(self.branchwise_loss)
        patterns_sorted = self.spatial_patterns[:, order]
        latent_sources_sorted = self.latent_sources[:, order, :]
        latent_sources_filt_sorted = self.latent_sources_filtered[:, order, :]
        latent_sources_weight_sorted = self.latent_sources_weighted[:, order, :]
        temporal_patterns_sorted = self.temporal_patterns[:, order, :]
        fake_evo = mne.evoked.EvokedArray(np.expand_dims(patterns_sorted[:, branch_num], 1), info, tmin=0)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        times = np.arange(0, latent_sources_sorted.shape[-1]/self.info['sfreq'], 1/self.info['sfreq'])
        ax2.plot(
            times,
            sp.stats.zscore(latent_sources_sorted.mean(0)[branch_num]),
            linewidth=2, alpha=.25
        )

        ax2.plot(
            times,
            sp.stats.zscore(latent_sources_filt_sorted.mean(0)[branch_num]),
            color='tab:blue',
            linewidth=1.5, alpha=.5
        )
        ax2.plot(
            times,
            sp.stats.zscore(latent_sources_weight_sorted.mean(0)[branch_num]),
            color='tab:blue',
            linewidth=1,
        )

        branch_tempwise_estimate = temporal_patterns_sorted.mean(0)[branch_num]
        interp_cubic = lambda y: sp.interpolate.interp1d(np.linspace(0, times[-1], len(y)), y, 'cubic')(times)

        ax2.plot(
            times,
            sp.stats.zscore(interp_cubic(branch_tempwise_estimate)),
            color='tab:red',
            linewidth=1.25,
            linestyle='--',
            alpha=.75
        )

        ax2.set_ylabel('Amplitude, zscore')
        ax2.set_xlabel('Time, s')
        ax2.legend(['spatially filtered', 'temporally filtered', 'temporally weighted', 'temporal pattern'], loc='upper right')

        spec_legend = list()
        x = np.arange(0, self.frequency_range[-1], .1)

        interp_cubic = lambda y: sp.interpolate.interp1d(self.frequency_range, y, 'cubic')(x)

        plt.xlim(0, 100)
        if 'input' in spec_plot_elems:
            spec_legend.append('input')
            data = sp.stats.zscore(np.real(self.filter_inputs[order[branch_num]].mean(0)))
            data -= data.min()
            ax3.plot(
                x,
                sp.stats.zscore(
                    interp_cubic(data)
                ),
                color='tab:blue',
                alpha=.25
            )
        if 'output' in spec_plot_elems:
            spec_legend.append('output')
            data = sp.stats.zscore(np.real(self.filter_outputs[order[branch_num]].mean(0)))
            data -= data.min()
            ax3.plot(
                x,
                sp.stats.zscore(
                    interp_cubic(data)
                ),
                color='tab:blue',
                linewidth=.75
            )
        if 'response' in spec_plot_elems:
            spec_legend.append('response')
            data = sp.stats.zscore(np.real(self.filter_responses[order[branch_num]]))
            data -= data.min()
            ax3.plot(
                x,
                interp_cubic(data),
                color='tab:red',
                alpha=.75,
                linestyle='--'
            )
        if 'pattern' in spec_plot_elems:
            spec_legend.append('pattern')
            data = sp.stats.zscore(np.real(self.filter_patterns[order[branch_num]].mean(0)))
            data -= data.min()
            ax3.plot(
                x,
                sp.stats.zscore(
                    interp_cubic(data)
                ),
                color='tab:blue',
                alpha=.75,
                linestyle=':'
            )
        ax3.legend(spec_legend, loc='upper right')
        ax3.set_ylabel('Amplitude, zscore')
        ax3.set_xlabel('Frequency, Hz')
        ax3.set_xlim(0, 100)

        fake_evo.plot_topomap(
            times=0,
            axes=ax1,
            colorbar=False,
            scalings=1,
            time_format="",
            outlines='head',
            cmap=generate_cmap(
                '#1f77b4',
                '#ffffff',
                '#d62728'
            )
        )
        if title:
            fig.suptitle(f'Branch {branch_num}')

        fig.tight_layout()

        return fig


class LFCNNWInterpreter(SPIRITInterpreter):
    @torch.no_grad()
    def compute_temporal(self):
        loader = DataLoader(self.dataset, len(self.dataset))
        n_latent = self.model.unmixing_layer.weight.shape[0]
        x, y = next(iter(loader))

        x = self.model.unmixing_layer(x)
        x = self.model.temp_conv(x)
        temporal_patterns, latent_sources_weighted = list(), list()
        for i, lw in enumerate(self.model.temp_sel):
            source = x[:, i, :]
            x1 = source*lw.w1 + lw.b
            x2 = torch.sigmoid(source*lw.w2)
            temporal_patterns.append(x2[:, ::self.model.pool_factor].contiguous())
            latent_sources_weighted.append(x1*x2)
        return torch.stack(temporal_patterns, -2).numpy(), torch.stack(latent_sources_weighted, -2).numpy()

    def plot_branch(
        self,
        branch_num: int,
        spec_plot_elems: list[str] = ['input', 'output', 'response'],
        title: str = None
    ) -> matplotlib.figure.Figure:

        info = deepcopy(self.info)
        info.__setstate__(dict(_unlocked=True))
        info['sfreq'] = 1.
        order = np.argsort(self.branchwise_loss)
        patterns_sorted = self.spatial_patterns[:, order]
        latent_sources_sorted = self.latent_sources[:, order, :]
        latent_sources_filt_sorted = self.latent_sources_filtered[:, order, :]
        latent_sources_weight_sorted = self.latent_sources_weighted[:, order, :]
        temporal_patterns_sorted = self.temporal_patterns[:, order, :]
        fake_evo = mne.evoked.EvokedArray(np.expand_dims(patterns_sorted[:, branch_num], 1), info, tmin=0)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        times = np.arange(0, latent_sources_sorted.shape[-1]/self.info['sfreq'], 1/self.info['sfreq'])
        ax2.plot(
            times,
            sp.stats.zscore(latent_sources_sorted.mean(0)[branch_num]),
            linewidth=2, alpha=.25
        )
        ax2.plot(
            times,
            sp.stats.zscore(latent_sources_filt_sorted.mean(0)[branch_num]),
            color='tab:blue',
            linewidth=1
        )

        branch_tempwise_estimate = temporal_patterns_sorted.mean(0)[branch_num]
        interp_cubic = lambda y: sp.interpolate.interp1d(np.linspace(0, times[-1], len(y)), y, 'cubic')(times)

        ax2.plot(
            times,
            sp.stats.zscore(interp_cubic(branch_tempwise_estimate)),
            color='tab:red',
            linewidth=1.25,
            linestyle='--',
            alpha=.75
        )

        ax2.set_ylabel('Amplitude, zscore')
        ax2.set_xlabel('Time, s')
        ax2.legend(['spatially filtered', 'temporally filtered', 'temporal pattern'], loc='upper right')

        spec_legend = list()
        x = np.arange(0, self.frequency_range[-1], .1)

        interp_cubic = lambda y: sp.interpolate.interp1d(self.frequency_range, y, 'cubic')(x)

        plt.xlim(0, 100)
        if 'input' in spec_plot_elems:
            spec_legend.append('input')
            data = sp.stats.zscore(np.real(self.filter_inputs[order[branch_num]].mean(0)))
            data -= data.min()
            ax3.plot(
                x,
                sp.stats.zscore(
                    interp_cubic(data)
                ),
                color='tab:blue',
                alpha=.25
            )
        if 'output' in spec_plot_elems:
            spec_legend.append('output')
            data = sp.stats.zscore(np.real(self.filter_outputs[order[branch_num]].mean(0)))
            data -= data.min()
            ax3.plot(
                x,
                sp.stats.zscore(
                    interp_cubic(data)
                ),
                color='tab:blue',
                linewidth=.75
            )
        if 'response' in spec_plot_elems:
            spec_legend.append('response')
            data = sp.stats.zscore(np.real(self.filter_responses[order[branch_num]]))
            data -= data.min()
            ax3.plot(
                x,
                interp_cubic(data),
                color='tab:red',
                alpha=.75,
                linestyle='--'
            )
        if 'pattern' in spec_plot_elems:
            spec_legend.append('pattern')
            data = sp.stats.zscore(np.real(self.filter_patterns[order[branch_num]].mean(0)))
            data -= data.min()
            ax3.plot(
                x,
                sp.stats.zscore(
                    interp_cubic(data)
                ),
                color='tab:blue',
                alpha=.75,
                linestyle=':'
            )
        ax3.legend(spec_legend, loc='upper right')
        ax3.set_ylabel('Amplitude, zscore')
        ax3.set_xlabel('Frequency, Hz')
        ax3.set_xlim(0, 100)

        fake_evo.plot_topomap(
            times=0,
            axes=ax1,
            colorbar=False,
            scalings=1,
            time_format="",
            outlines='head',
            cmap=generate_cmap(
                '#1f77b4',
                '#ffffff',
                '#d62728'
            )
        )
        if title:
            fig.suptitle(f'Branch {branch_num}')

        fig.tight_layout()

        return fig


class TimeCompNetInterpreter(LFCNNInterpreter):
    """
        Initialize TimeCompNetInterpreter object.

        Parameters:
            model (TimeCompNet): Trained TimeCompNet model.
            dataset (Dataset): Data used to test the TimeCompNet model.
            info (mne.Info): Information about recordings, typically contained in the "info" property of the corresponding instance (E.g. epochs.info).
        """
    def __init__(self, model: TimeCompNet, dataset: Dataset, info: mne.Info):
        super().__init__(model, dataset, info)
        self._tempwise_loss = None

    @torch.no_grad()
    def compute_tempwise_loss(self):
        """
        This method computes the tempwise loss within each branch of TimeCompNet (branch consits of two connected spatial and temporal filters and ended with tempral compression layer).
        It computes loss of each timepoint in time compression matrix of each branch by computing loss without all branches except of n-th one and without i-th timepoint.
        It is one of the easiest ways to estimate relevance of each timepoint within each branch.

        Returns:
            numpy.ndarray: A 2-dimensional numpy array of shape (n_latent, n_timepoints) where n_latent is the number of branches in the model and n_timepoints is time dimensionality.
        """

        loader = DataLoader(self.dataset, len(self.dataset))
        n_latent = self.model.unmixing_layer.weight.shape[0]
        X, y = next(iter(loader))
        temp_conv_output = self.model.temp_conv(self.model.unmixing_layer(X))
        n_times = temp_conv_output.shape[-1]
        branchwise = list()
        unmixing_weights_original = deepcopy(self.model.unmixing_layer.weight)
        unmixing_bias_original = deepcopy(self.model.unmixing_layer.bias)
        temp_conv_bias_original = deepcopy(self.model.temp_conv.bias)

        for i in tqdm(range(n_latent), initial=1, total=n_latent, file=sys.stdout):
            with nostdout():
                if i == 0:
                    self.model.unmixing_layer.weight[1:, :, :] = 0
                    self.model.unmixing_layer.bias[1:] = 0
                    self.model.temp_conv.bias[1:] = 0
                elif i == n_latent - 1:
                    self.model.unmixing_layer.weight[:-1, :, :] = 0
                    self.model.unmixing_layer.bias[:-1] = 0
                    self.model.temp_conv.bias[:-1] = 0
                else:
                    self.model.unmixing_layer.weight[:i, :, :] = 0
                    self.model.unmixing_layer.weight[i+1:, :, :] = 0
                    self.model.unmixing_layer.bias[:i] = 0
                    self.model.unmixing_layer.bias[i+1:] = 0
                    self.model.temp_conv.bias[:i] = 0
                    self.model.temp_conv.bias[i+1:] = 0

                timesel = self.model.timesel_list[i]
                tempwise = list()

                for j in range(n_times):
                    timesel_timepoint_original = deepcopy(timesel[0].weight[:, j])
                    timesel[0].weight[:, j] = 0
                    tempwise.append(self.model.evaluate(loader)['loss'])
                    timesel[0].weight[:, j] = timesel_timepoint_original

                branchwise.append(tempwise)
                self.model.unmixing_layer.weight[:, :, :] = unmixing_weights_original
                self.model.unmixing_layer.bias[:] = unmixing_bias_original
                self.model.temp_conv.bias[:] = temp_conv_bias_original

        return np.array(branchwise)

    def __validate_tempwise_estimate(self):
        """
        Validates the tempwise loss by computing it if it has not been computed previously.

        The `_tempwise_loss` attribute is set to the result of the `compute_tempwise_loss` method.

        """
        if self._tempwise_loss is None:
            print('Estimating temporal compression weights, it will take some time')
            self._tempwise_loss = self.compute_tempwise_loss()

    @property
    def tempwise_loss(self):
        """
        Get the tempwise loss.

        Returns:
            List[List[float]]: The tempwise loss.
        """
        self.__validate_tempwise_estimate()
        return self._tempwise_loss

    def plot_branch(
        self,
        branch_num: int,
        spec_plot_elems: list[str] = ['input', 'output', 'response'],
        title: str = None
    ) -> matplotlib.figure.Figure:
        """
        Plot the branchwise information for a specific branch of the model.

        Parameters:
        branch_num (int): the branch number to plot (order of branches is determined by `branchwise_loss`).
        spec_plot_elems (List[str]): a list of plot elements to include in the spectrum plot.
        title (str): optional title for the plot.

        Returns:
        matplotlib.figure.Figure: the plot.

        """
        info = deepcopy(self.info)
        info.__setstate__(dict(_unlocked=True))
        info['sfreq'] = 1.
        order = np.argsort(self.branchwise_loss)
        patterns_sorted = self.spatial_patterns[:, order]
        latent_sources_sorted = self.latent_sources[:, order, :]
        latent_sources_filt_sorted = self.latent_sources_filtered[:, order, :]
        tempwise_losses_sorted = self.tempwise_loss[order]
        fake_evo = mne.evoked.EvokedArray(np.expand_dims(patterns_sorted[:, branch_num], 1), info, tmin=0)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        times = np.arange(0, latent_sources_sorted.shape[-1]/self.info['sfreq'], 1/self.info['sfreq'])
        ax2.plot(
            times,
            sp.stats.zscore(latent_sources_sorted.mean(0)[branch_num]),
            linewidth=2, alpha=.25
        )
        ax2.plot(
            times,
            sp.stats.zscore(latent_sources_filt_sorted.mean(0)[branch_num]),
            color='tab:blue',
            linewidth=1
        )
        kernel = 6
        branch_tempwise_losses = np.concatenate([
            [np.nan for _ in range(kernel//2)],
            np.convolve(sp.stats.zscore(tempwise_losses_sorted[branch_num]), np.ones(kernel)/kernel, mode='same')[kernel//2:][:-kernel//2],
            [np.nan for _ in range(kernel//2)]
        ])
        ax2.plot(
            times,
            branch_tempwise_losses,
            color='tab:red',
            linewidth=1.25,
            linestyle='--',
            alpha=.75
        )
        ax2.set_ylabel('Amplitude, zscore')
        ax2.set_xlabel('Time, s')
        ax2.legend(['spatially filtered', 'temporally filtered', 'loss-based estimate'], loc='upper right')

        spec_legend = list()
        x = np.arange(0, self.frequency_range[-1], .1)

        interp_cubic = lambda y: sp.interpolate.interp1d(self.frequency_range, y, 'cubic')(x)

        plt.xlim(0, 100)
        if 'input' in spec_plot_elems:
            spec_legend.append('input')
            data = sp.stats.zscore(np.real(self.filter_inputs[order[branch_num]].mean(0)))
            data -= data.min()
            ax3.plot(
                x,
                sp.stats.zscore(
                    interp_cubic(data)
                ),
                color='tab:blue',
                alpha=.25
            )
        if 'output' in spec_plot_elems:
            spec_legend.append('output')
            data = sp.stats.zscore(np.real(self.filter_outputs[order[branch_num]].mean(0)))
            data -= data.min()
            ax3.plot(
                x,
                sp.stats.zscore(
                    interp_cubic(data)
                ),
                color='tab:blue',
                linewidth=.75
            )
        if 'response' in spec_plot_elems:
            spec_legend.append('response')
            data = sp.stats.zscore(np.real(self.filter_responses[order[branch_num]]))
            data -= data.min()
            ax3.plot(
                x,
                interp_cubic(data),
                color='tab:red',
                alpha=.75,
                linestyle='--'
            )
        if 'pattern' in spec_plot_elems:
            spec_legend.append('pattern')
            data = sp.stats.zscore(np.real(self.filter_patterns[order[branch_num]].mean(0)))
            data -= data.min()
            ax3.plot(
                x,
                sp.stats.zscore(
                    interp_cubic(data)
                ),
                color='tab:blue',
                alpha=.75,
                linestyle=':'
            )
        ax3.legend(spec_legend, loc='upper right')
        ax3.set_ylabel('Amplitude, zscore')
        ax3.set_xlabel('Frequency, Hz')
        ax3.set_xlim(0, 100)

        fake_evo.plot_topomap(
            times=0,
            axes=ax1,
            colorbar=False,
            scalings=1,
            time_format="",
            outlines='head',
            cmap=generate_cmap(
                '#1f77b4',
                '#ffffff',
                '#d62728'
            )
        )
        if title:
            fig.suptitle(f'Branch {branch_num}')

        fig.tight_layout()

        return fig

