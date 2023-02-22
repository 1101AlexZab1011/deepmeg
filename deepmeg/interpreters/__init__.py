import sys
import torch
from torch.utils.data import DataLoader, Dataset
import mne
import scipy as sp
import numpy as np
from copy import deepcopy
from ..models.interpretable import LFCNN, HilbertNet, TimeCompNet, SPIRIT
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from deepmeg.utils.printout import nostdout
from deepmeg.utils.colors import generate_cmap



class LFCNNInterpreter:
    """
        Initialize LFCNNInterpreter object.

        Parameters:
            model (LFCNN): Trained LFCNN model.
            dataset (Dataset): Data used to train the LFCNN model.
            info (mne.Info): Information about recordings, typically contained in the "info" property of the corresponding instance (E.g. epochs.info).

        """
    def __init__(self, model: LFCNN, dataset: Dataset, info: mne.Info):
        self.model = model
        self.dataset = dataset
        self.info = info
        self._latent_sources = None
        self._latent_sources_filtered = None
        self._spatial_patterns = None
        self._spatial_filters = None
        self._frequency_range = None
        self._filter_inputs = None
        self._filter_responses = None
        self._filter_outputs = None
        self._filter_patterns = None
        self._branchwise_loss = None

    @torch.no_grad()
    def compute_patterns(self):
        """
        Compute the spatial patterns and filters of the LFCNN model.
        Spatial patterns are achieved from spatial filters with following formula:

            A = K_x * W * K_s

        Where A is matrix of spatial patterns, W is a matrix of spatial filters (model.unmixing_layer.weight, biases can be omitted due to linearity),
        K_x and K_s are covariance matrices of input data and latent sources respectively [1].
        According to [2], K_x should also be filtered with weights from temporal filtering layer (model.temp_conv)

        Returns:
            tuple: Tuple of two numpy arrays, containing the spatial patterns and filters.

        References:
            [1] Haufe, S., Meinecke, F., Görgen, K., Dähne, S., Haynes, J. D., Blankertz, B., & Bießmann, F. (2014).
                On the interpretation of weight vectors of linear models in multivariate neuroimaging.
                NeuroImage, 87, 96-110. https://doi.org/10.1016/j.neuroimage.2013.10.067
            [2] Petrosyan, A., Sinkin, M., Lebedev, M., & Ossadtchi, A. (2021).
                Decoding and interpreting cortical signals with a compact convolutional neural network.
                Journal of Neural Engineering, 18(2), 026019. https://doi.org/10.1088/1741-2552/abe20e
        """
        x, y = next(iter(DataLoader(self.dataset, len(self.dataset))))
        x_flatten = x.reshape(x.shape[1], x.shape[0]*x.shape[-1])
        latent_sources = self.model.unmixing_layer(x)
        latent_sources_filtered = self.model.temp_conv(latent_sources)
        latent_sources_flatten = latent_sources_filtered.reshape(latent_sources.shape[1], latent_sources.shape[0]*latent_sources.shape[-1])
        self._latent_sources = latent_sources.numpy()
        self._latent_sources_filtered = latent_sources_filtered.numpy()
        unmixing_matrix = self.model.unmixing_layer.weight.numpy()[:, :, 0]
        filters = unmixing_matrix.T
        # covariance of latent_sources should aim to I, due to linear independance
        x = x.permute(1, 0, -1)
        x_flatten = x.reshape(x.shape[0], x.shape[1]*x.shape[-1])
        patterns = list()
        for comp_num in range(len(self.model.unmixing_layer.weight)):
            x_filt_flatten = torch.zeros_like(x_flatten)

            for ch_num in range(x.shape[0]):

                x_filt_flatten[ch_num] = torch.nn.functional.conv1d(
                    torch.unsqueeze(x_flatten[ch_num], 0),
                    torch.unsqueeze(self.model.temp_conv.weight[comp_num].detach(), 0),
                    padding='same'
                )
            patterns.append(torch.cov(x_filt_flatten)@self.model.unmixing_layer.weight[comp_num])

        patterns = torch.squeeze(torch.stack(patterns, 1))@torch.cov(latent_sources_flatten)

        return patterns.numpy(), filters

    @torch.no_grad()
    def compute_branchwise_loss(self):
        """
        This method computes the branchwise loss for each branch of LF-CNN (branch consits of two connected spatial and temporal filters).
        It computes loss of each branch by subtracting loss of original model from loss of the same model without nth branch.
        It is one of the easiest ways to estimate relevance of the branch.

        Returns:
            numpy.ndarray: A 1-dimensional numpy array of shape (n_latent, ) where n_latent is the number of branches in the model.
        """
        loader = DataLoader(self.dataset, len(self.dataset))
        base_loss = self.model.evaluate(loader)['loss']
        n_latent = self.model.unmixing_layer.weight.shape[0]

        losses = list()
        for i in range(n_latent):
            branch_unmixing_weights_original = deepcopy(self.model.unmixing_layer.weight[i, :, :])
            self.model.unmixing_layer.weight[i, :, :] = 0

            if self.model.unmixing_layer.bias is not None:
                branch_unmixing_bias_original = deepcopy(self.model.unmixing_layer.bias[i])
                self.model.unmixing_layer.bias[i] = 0
            if self.model.temp_conv.bias is not None:
                branch_temp_conv_bias_original = deepcopy(self.model.temp_conv.bias[i])
                self.model.temp_conv.bias[i] = 0

            losses.append(self.model.evaluate(loader)['loss'])
            self.model.unmixing_layer.weight[i, :, :] = branch_unmixing_weights_original

            if self.model.unmixing_layer.bias is not None:
                self.model.unmixing_layer.bias[i] = branch_unmixing_bias_original
            if self.model.temp_conv.bias is not None:
                self.model.temp_conv.bias[i] = branch_temp_conv_bias_original

        return base_loss - np.array(losses)

    @torch.no_grad()
    def compute_specta(self):
        """
        Compute the spectral parameters of temporal filters (model.temp_conv.weight, biases can be omitted due to linearity).

        Returns:
            frange (ndarray): The frequency range for the power spectral density.
            finputs (list): The power spectral density of the latent sources (spatially filtered data).
            fresponces (list): The absolute value of the frequency response of the filters.
                (FFT of the dephased Wiener filter).
            fpatterns (list): The product of the power spectral density and the absolute value of the frequency response.
                (spectral power density of the signal at the input of the Wiener optimal filter).
            foutputs (list): The product of the power spectral density and the squared magnitude of the frequency response.
                (spectral power density of the signal at the output of the digital filter).

        """
        filters = torch.squeeze(self.model.temp_conv.weight).detach().numpy()
        finputs, fresponces, fpatterns, foutputs = list(), list(), list(), list()
        frange = None

        for branch_num in range(self.latent_sources.shape[1]):
            lat_tc = self.latent_sources[:, branch_num, :]
            kern = filters[branch_num]
            frange, psd = sp.signal.welch(lat_tc, fs=self.info['sfreq'], nperseg=self.latent_sources.shape[-1])
            _, h = sp.signal.freqz(kern, 1, worN=self.latent_sources.shape[-1]//2 + 1)
            finputs.append(psd)
            fresponces.append(np.abs(h))
            fpatterns.append(finputs[-1]*fresponces[-1])
            foutputs.append(finputs[-1]*h*np.conj(h))

        return frange, finputs, fresponces, foutputs, fpatterns

    def __validate_spatial(self):
        """
        Computes the spatial patterns and filters for the given data.
        If the spatial patterns or filters are already computed and stored, this step is skipped.
        """
        if self._spatial_patterns is None or self._spatial_filters is None:
            self._spatial_patterns, self._spatial_filters = self.compute_patterns()

    def __validate_spectral(self):
        """
        This method validates the spectal properties of the data. If the frequency range, filter inputs, filter responses,
            filter outputs, or filter patterns have not been computed, the method calls compute_specta to compute them.
        """
        if self._frequency_range is None\
            or self._filter_inputs is None\
            or self._filter_responses is None\
            or self._filter_outputs is None\
            or self._filter_patterns is None:
            self._frequency_range,\
                self._filter_inputs,\
                self._filter_responses,\
                self._filter_outputs,\
                self._filter_patterns = self.compute_specta()

    def __validate_branchwise_estimate(self):
        """
        Validates the branchwise loss by computing it if it has not been computed previously.

        The `_branchwise_loss` attribute is set to the result of the `compute_branchwise_loss` method.

        """
        if self._branchwise_loss is None:
            self._branchwise_loss = self.compute_branchwise_loss()

    @property
    def latent_sources(self):
        """
        Get the latent sources.

        Returns:
            torch.Tensor: The latent sources with shape (n_epochs, n_latent, n_times).
        """
        # shape: n_epochs, n_latent, n_times
        self.__validate_spatial()
        return self._latent_sources
    @property
    def latent_sources_filtered(self):
        """
        Get the filtered latent sources.

        Returns:
            torch.Tensor: The filtered latent sources with shape (n_epochs, n_latent, n_times).
        """
        self.__validate_spatial()
        return self._latent_sources_filtered
    @property
    def spatial_patterns(self):
        """
        Get the spatial patterns.

        Returns:
            torch.Tensor: The spatial patterns with shape (n_channels, n_latent).
        """
        # shape: n_channels, n_latent
        self.__validate_spatial()
        return self._spatial_patterns
    @property
    def spatial_filters(self):
        """
        Get the spatial filters.

        Returns:
            torch.Tensor: The spatial filters with shape (n_channels, n_latent).
        """
        self.__validate_spatial()
        return self._spatial_filters
    @property
    def frequency_range(self):
        """
        Get the frequency range.

        Returns:
            numpy.ndarray: The frequency range for spectrum.
        """
        self.__validate_spectral()
        return self._frequency_range
    @property
    def filter_inputs(self):
        """
        Get the temporal filters inputs.

        Returns:
            List[numpy.ndarray]: The filters inputs.
        """
        self.__validate_spectral()
        return self._filter_inputs
    @property
    def filter_responses(self):
        """
        Get the temporal filters responses.

        Returns:
            List[numpy.ndarray]: The filters responses.
        """
        self.__validate_spectral()
        return self._filter_responses
    @property
    def filter_outputs(self):
        """
        Get the temporal filters outputs.

        Returns:
            List[numpy.ndarray]: The filters outputs.
        """
        self.__validate_spectral()
        return self._filter_outputs
    @property
    def filter_patterns(self):
        """
        Get the temporal filters patterns.

        Returns:
            List[numpy.ndarray]: The filtering patterns.
        """
        self.__validate_spectral()
        return self._filter_patterns
    @property
    def branchwise_loss(self):
        """
        Get the branchwise loss.

        Returns:
            List[float]: The branchwise loss.
        """
        self.__validate_branchwise_estimate()
        return self._branchwise_loss

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
        fake_evo = mne.evoked.EvokedArray(np.expand_dims(patterns_sorted[:, branch_num], 1), info, tmin=0)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        times = np.linspace(0, latent_sources_sorted.shape[-1]/self.info['sfreq'], latent_sources_filt_sorted.shape[-1])
        ax2.plot(
            times,
            sp.stats.zscore(latent_sources_sorted.mean(0)[branch_num]),
            linewidth=2, alpha=0.25
        )
        ax2.plot(
            times,
            sp.stats.zscore(latent_sources_filt_sorted.mean(0)[branch_num]),
            color='tab:blue',
            linewidth=1
        )
        ax2.set_ylabel('Amplitude, zscore')
        ax2.set_xlabel('Time, s')
        ax2.legend(['spatially filtered', 'temporally filtered'], loc='upper right')

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
                alpha=.75,
                linestyle='--',
                color='tab:red'
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


class HilbertNetInterpreter(LFCNNInterpreter):
    """
        Initialize HilbertNetInterpreter object.

        Parameters:
            model (HilbertNet): Trained HilbertNet model.
            dataset (Dataset): Data used to train the LFCNN model.
            info (mne.Info): Information about recordings, typically contained in the "info" property of the corresponding instance (E.g. epochs.info).

        """
    def __init__(self, model: HilbertNet, dataset: Dataset, info: mne.Info):
        super().__init__(model, dataset, info)
        self._envelopes = None

    @property
    @torch.no_grad()
    def envelopes(self):
        """
        Get the latent sources envelopes.

        Returns:
            torch.Tensor: The latent sources envelopes with shape (n_epochs, n_latent, n_times).
        """
        if self._envelopes is None:
            x = self.model.temp_conv_activation(torch.Tensor(self.latent_sources_filtered))
            x = self.model.temp_conv_activation(x)
            self._envelopes = self.model.env_conv(x).numpy()
        return self._envelopes

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
        envelopes_sorted = self.envelopes[:, order, :]
        fake_evo = mne.evoked.EvokedArray(np.expand_dims(patterns_sorted[:, branch_num], 1), info, tmin=0)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        times = np.linspace(0, latent_sources_sorted.shape[-1]/self.info['sfreq'], latent_sources_filt_sorted.shape[-1])
        ax2.plot(
            times,
            sp.stats.zscore(latent_sources_sorted.mean(0)[branch_num]),
            linewidth=2, alpha=0.25
        )
        ax2.plot(
            times,
            sp.stats.zscore(latent_sources_filt_sorted.mean(0)[branch_num]),
            color='tab:blue',
            linewidth=1.5,
            alpha=.5
        )
        ax2.plot(
            times,
            sp.stats.zscore(envelopes_sorted.mean(0)[branch_num]),
            color='tab:blue',
            linewidth=1
        )
        ax2.set_ylabel('Amplitude, zscore')
        ax2.set_xlabel('Time, s')
        ax2.legend(['spatially filtered', 'temporally filtered', 'envelope'], loc='upper right')

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
                alpha=.75,
                linestyle='--',
                color='tab:red'
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
