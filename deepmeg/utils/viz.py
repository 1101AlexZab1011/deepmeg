from functools import partial
from typing import Callable, Sequence
import numpy as np
import scipy as sp
import mne
from collections import namedtuple
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from collections.abc import Iterable
import copy
from .colors import generate_cmap
from ..preprocessing.transforms import rowwise_zscore
from .params import NetworkParameters

def plot_patterns(
    patterns: np.ndarray,
    info: mne.Info,
    order: int = None,
    axes: plt.Axes = None,
    cmap: str | mpl.colors.Colormap = 'RdBu_r',
    sensors: bool = True,
    colorbar: True = False,
    res: int = 64,
    size: int = 1,
    cbar_fmt: str = '%3.1f',
    name_format: str = 'Latent\nSource %01d',
    show: bool = True,
    show_names: bool = False,
    outlines: str = 'head',
    contours: int = 6,
    image_interp: str = 'linear'
) -> mpl.figure.Figure:
    """
    Plot spatial patterns for a group of sources.

    Args:
        patterns (numpy.ndarray): The source patterns to plot. The shape of the
            array should be (n_sources, n_channels).
        info (mne.Info): The measurement information.
        order (int, optional): The order in which to plot the patterns. If None,
            plot the patterns in the order in which they appear in the array.
            Defaults to None.
        axes (matplotlib.axes.Axes, optional): The matplotlib axes on which to
            plot the patterns. If None, create new axes. Defaults to None.
        cmap (str or matplotlib.colors.Colormap, optional): The colormap to use
            for the plot. Defaults to 'RdBu_r'.
        sensors (bool, optional): Whether to plot sensor locations. Defaults to
            True.
        colorbar (bool, optional): Whether to plot a colorbar. Defaults to False.
        res (int, optional): The resolution of the topomap. Defaults to 64.
        size (int, optional): The size of the plot. Defaults to 1.
        cbar_fmt (str, optional): The format string for colorbar tick labels.
            Defaults to '%3.1f'.
        name_format (str, optional): The format string for the title of each
            plot. Defaults to 'Latent\nSource %01d'.
        show (bool, optional): Whether to show the plot. Defaults to True.
        show_names (bool, optional): Whether to show the names of the channels.
            Defaults to False.
        outlines (str, optional): The name of the head outline to use. Defaults
            to 'head'.
        contours (int, optional): The number of contour lines to use. Defaults to 6.
        image_interp (str, optional): The interpolation method to use when
            plotting the topomap. Defaults to 'linear'.

    Returns:
        matplotlib.figure.Figure: The matplotlib figure containing the plot.
    """
    if order is None:
        order = range(patterns.shape[1])
    info = copy.deepcopy(info)
    info.__setstate__(dict(_unlocked=True))
    info['sfreq'] = 1.
    patterns = mne.EvokedArray(patterns, info, tmin=0)
    return patterns.plot_topomap(
        times=order,
        axes=axes,
        cmap=cmap, colorbar=colorbar, res=res,
        cbar_fmt=cbar_fmt, sensors=sensors, units=None, time_unit='s',
        time_format=name_format, size=size, show_names=show_names,
        outlines=outlines,
        contours=contours, image_interp=image_interp, show=show)


Limit = namedtuple('Limit', 'x y')


class SortingCallback:
    """
    Callback class for sorting data in an InterpretationPlotter object.

    Args:
        button (Button): The button object for triggering the callback.
        plotter (InterpretationPlotter): The InterpretationPlotter object containing the data to be sorted.
        fig (Figure): The matplotlib Figure object for displaying the data.
        indices (list[int]): The initial indices to use for sorting.

    Attributes:
        sorted_indices (list[int]): The sorted indices.

    Raises:
        AttributeError: If the sorted_indices attribute is attempted to be directly set.

    """

    def __init__(
        self,
        button: Button,
        plotter: 'InterpretationPlotter',
        fig: plt.Figure,
        indices: list[int]
    ):
        self._button = button
        self._plotter = plotter
        self._fig = fig
        self._bar_ax = self._plotter.main_axes[3]
        self._imshow_ax = self._plotter.main_axes[2]
        self._event = None
        self._sorted_indices = indices

    def __call__(self, event: mpl.backend_bases.MouseEvent):
        self._event = event

        if '▼' in self._button.label._text:
            self.decreasing()
        else:
            self.increasing()

    @property
    def sorted_indices(self):
        """
        The sorted indices.

        Returns:
            list[int]: The sorted indices.

        """
        return self._sorted_indices

    @sorted_indices.setter
    def sorted_indices(self, value):
        """
        Setter for the sorted_indices attribute. Raises an AttributeError if called.

        Args:
            value: The value to set the sorted_indices attribute to.

        Raises:
            AttributeError: Always raised, since the sorted_indices attribute cannot be directly set.

        """
        raise AttributeError('Impossible to set indices directly')

    def increasing(self):
        """
        Sort the data in increasing order and update the plot.

        """
        self._button.label.set_text('Sort ▼')
        self._sorted_indices = sorted(range(len(self._plotter.sums)), key=lambda k: self._plotter.sums[k], reverse=True)
        self.update()

    def decreasing(self):
        """
        Sort the data in decreasing order and update the plot.

        """
        self._button.label.set_text('Sort ▲')
        self._sorted_indices = sorted(range(len(self._plotter.sums)), key=lambda k: self._plotter.sums[k])
        self.update()

    def update(self):
        """
        Update the plot based on the sorted indices.

        """
        self._imshow_ax.clear()
        self._imshow_ax.imshow(self._plotter.params.spatial.patterns.T[self._sorted_indices, :], aspect='auto', cmap=self._plotter.cmap)
        self._plotter.main_axes[4].clear()
        temp_map = np.empty(self._plotter.params.spatial.patterns.T.shape)
        temp_map[:] = np.nan
        self._plotter.main_axes[4].imshow(temp_map, aspect='auto')
        self._bar_ax.clear()
        self._bar_ax.barh(
            range(len(self._plotter.sums)),
            np.abs(self._plotter.sums)[self._sorted_indices], color=self._plotter.colors[self._sorted_indices],
            height=.9
        )
        self._plotter.init_main_canvas(self)
        self._fig.canvas.draw()


class FilterButtonCallback:
    """
    Callback for a button that filters or resets filtering of temporal and induced data for a given cluster.

    Args:
        button (Button): The button widget.
        plotter (InterpretationPlotter): The plotter that shows the data.
        fig (plt.Figure): The figure that the plotter is drawn on.
        sorter (SortingCallback): The callback that sorts clusters.
        iy (int): The index of the cluster to filter.
        ax21 (plt.Axes): The first axes showing the induced data.
        ax22 (plt.Axes): The second axes showing the filtered induced data.
        ax22_t (plt.Axes): The second axes showing the time course of the cluster.
        ax23 (plt.Axes): The third axes showing the pattern of the cluster.
        ax24 (plt.Axes): The fourth axes showing the plot of the pattern on the topography.
        cb (mpl.colorbar.Colorbar): The colorbar that shows the color code of the induced data.
        shift (float): The shift parameter for the plot.
        f_max (int): The maximum frequency for induced data.
        crop (int): The number of samples to crop at the beginning and end of the data.
        legend (list[str]): The legend for the plot.
        filtered (bool): Whether the data is filtered or not.
    """
    def __init__(
        self,
        button: Button,
        plotter: 'InterpretationPlotter',
        fig: plt.Figure,
        sorter: SortingCallback,
        iy: int,
        ax21: plt.Axes,
        ax22: plt.Axes,
        ax22_t: plt.Axes,
        ax23: plt.Axes,
        ax24: plt.Axes,
        cb: mpl.colorbar.Colorbar,
        shift: float,
        f_max: int,
        crop: int,
        legend: list[str],
        filtered: bool = False,
        f_scale: tuple[float, float] = None,
    ):
        self.button = button
        self.plotter = plotter
        self.fig = fig
        self.sorter = sorter
        self.iy = iy
        self.axes = [ax21, ax22, ax22_t, ax23, ax24]
        self.cb = cb
        self.shift = shift
        self.f_max = f_max
        self.crop = crop
        self.legend = legend
        self.filtered = filtered,
        self.f_scale = f_scale

    def __call__(self, event: mpl.backend_bases.MouseEvent):
        """
        Method that updates the figure when the button is clicked.

        Args:
            event (mpl.backend_bases.MouseEvent): The mouse click event.
        """
        self.filtered = not self.filtered
        text = 'Filter' if self.filtered else 'Redo'
        self.button.label.set_text(text)
        evoked = self.plotter.params.temporal.time_courses.mean(0)[self.sorter.sorted_indices[self.iy]][self.crop:-self.crop] if self.filtered else\
            self.plotter.params.temporal.time_courses_filtered.mean(0)[self.sorter.sorted_indices[self.iy]][self.crop:-self.crop]

        induced = self.plotter.params.temporal.induceds.copy()[
            self.sorter.sorted_indices[self.iy],
            :self.f_max,
            :
        ] if self.filtered else\
            self.plotter.params.temporal.induceds_filtered[
            self.sorter.sorted_indices[self.iy],
            :self.f_max,
            :
        ]
        induced = rowwise_zscore(induced)[:, self.crop:-self.crop]
        self.axes[1].axes.clear()
        self.axes[2].axes.clear()
        if self.f_scale is not None:
            pos = self.axes[1].imshow(
                induced,
                origin='lower',
                cmap=self.plotter.cmap,
                interpolation='bicubic',
                aspect='auto',
                interpolation_stage='rgba',
                vmin=self.f_scale[0], vmax=self.f_scale[1]
            )
        else:
            pos = self.axes[1].imshow(
                induced,
                origin='lower',
                cmap=self.plotter.cmap,
                interpolation='bicubic',
                aspect='auto',
                interpolation_stage='rgba'
            )
        self.axes[1].contour(
            induced,
            [
                np.percentile(induced, 25),
                np.percentile(induced, 75)
            ],
            origin='lower',
            linewidths=.1,
            cmap=self.plotter.cmap,
        )
        self.axes[2].plot(
            sp.stats.zscore(evoked),
            linewidth=.75
        )
        self.cb.remove()
        self.cb = self.fig.colorbar(
            pos,
            ax=self.axes[1],
            pad=0.12,
            orientation='horizontal',
            aspect=75,
            fraction=.12
        )
        self.plotter.init_additional_canvas(*self.axes, self.cb, self.shift, self.crop, self.legend)
        self.fig.canvas.draw()


class InterpretationPlotter:
    """
    A class for creating plots of the interpretations of the model's hidden units.

    Args:
        params (NetworkParameters): The network parameters to interpret.
        summarize (str | Sequence[int | float] | Callable[[np.ndarray], np.ndarray], optional):
            A method to summarize the spatial patterns. Defaults to 'sum'.

    Attributes:
        limits (Limit): The limit of the spatial patterns.
        info (dict): A dictionary containing information about the network parameters.
        params (NetworkParameters): The network parameters to interpret.
        sums (ndarray): The summarized spatial patterns.
        colors (ndarray): The colors used for the summary.
        main_fig (matplotlib.figure.Figure): The main figure object.
        main_axes (list[matplotlib.axes.Axes]): A list of the main axes objects.
        cmap (matplotlib.colors.ListedColormap): The colormap used for the plot.

    Methods:
        init_main_canvas(sorting_callback):
            Initializes the main canvas with the given sorting callback.
        init_additional_canvas(ax21, ax22, ax22_t, ax23, ax24, cb, shift, crop, legend):
            Initializes the additional canvas with the given parameters.
    """
    def __init__(
        self,
        params: NetworkParameters,
        summarize: str | Sequence[int | float] | Callable[[np.ndarray], np.ndarray] = 'sum',
    ):
        """
        Constructs an InterpretationPlotter.

        Args:
            params (NetworkParameters): The network parameters to interpret.
            summarize (str | Sequence[int | float] | Callable[[np.ndarray], np.ndarray], optional):
                A method to summarize the spatial patterns. Defaults to 'sum'.
        """
        self.limits = Limit(*params.spatial.patterns.shape)
        self.info = params.info
        self.params = params

        if summarize == 'sum':
            self.sums = np.sum(self.params.spatial.patterns, axis=0)
        elif summarize == 'sumabs':
            self.sums = np.sum(np.abs(self.params.spatial.patterns), axis=0)
        elif summarize == 'abssum':
            self.sums = np.abs(np.sum(self.params.spatial.patterns, axis=0))
        elif isinstance(summarize, Iterable) and len(summarize) == self.limits.y:
            self.sums = np.array(summarize)
        elif isinstance(summarize, Callable):
            self.sums = summarize(self.params.spatial.patterns)
        else:
            if isinstance(summarize, str):
                raise NotImplementedError(
                    f'The "{summarize}" method not implemented. '
                    'Available methods: "sum", "sumabs", "abssum"'
                )
            else:
                raise NotImplementedError(
                    f'The method for processing "{type(summarize)}" data for summary is not implemented. '
                )

        self.colors = np.array(['#d62728' if sum_ >= 0 else '#1f77b4' for sum_ in self.sums])
        self.main_fig = plt.figure()
        gs = self.main_fig.add_gridspec(2, 2, hspace=0, wspace=0, width_ratios=[1, .1], height_ratios=[.025, 1])
        # (ax01, ax02), (ax1, ax2)
        (ax0, ax1), (ax2, ax3) = gs.subplots(sharex='col', sharey='row')
        ax2_ = ax2.twinx()
        self.main_axes = [ax0, ax1, ax2, ax3, ax2_]
        self.cmap = generate_cmap(
            '#1f77b4',
            '#ffffff',
            '#d62728'
        )

    def init_main_canvas(self, sorting_callback: SortingCallback):
        """Initialize the main canvas for displaying the latent sources.

        Args:
            sorting_callback: An instance of the SortingCallback class, which
                provides access to the sorted indices of the latent sources.

        Returns:
            None
        """
        ax01, ax02, ax1, ax2, ax2_ = self.main_axes
        ax01.axes.xaxis.set_visible(False)
        ax01.axes.yaxis.set_visible(False)
        ax02.axes.xaxis.set_visible(False)
        ax02.axes.yaxis.set_visible(False)

        ax01.spines['right'].set_visible(False)
        ax01.spines['left'].set_visible(False)
        ax01.spines['top'].set_visible(False)
        ax01.spines['bottom'].set_visible(False)
        ax02.spines['right'].set_visible(False)
        ax02.spines['left'].set_visible(False)
        ax02.spines['top'].set_visible(False)
        ax02.spines['bottom'].set_visible(False)

        ax1.spines['right'].set_visible(False)
        ax1.set_xlabel('Channels')
        ax1.set_ylabel('Latent Sources')
        ax1.spines['left'].set_alpha(0.2)
        ax1.spines['bottom'].set_alpha(0.2)
        ax1.spines['top'].set_alpha(0.2)
        ax1.axes.yaxis.set_alpha(0.2)
        ax1.set_yticks(np.arange(self.limits.y))
        ax1.set_yticklabels(labels=[i + 1 for i in sorting_callback.sorted_indices])
        ax1.tick_params(axis='both', which='both', length=5, color='#00000050')

        ax2.axes.yaxis.set_visible(False)
        ax2.axes.xaxis.set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.spines['top'].set_alpha(0.2)
        ax2.spines['right'].set_alpha(0.2)
        ax2.spines['bottom'].set_alpha(0.2)
        ax2_.axes.yaxis.set_visible(False)
        ax2_.spines['left'].set_visible(False)
        ax2_.axes.xaxis.set_visible(False)
        ax2_.spines['top'].set_alpha(0.2)
        ax2_.spines['right'].set_alpha(0.2)
        ax2_.spines['bottom'].set_alpha(0.2)

    def init_additional_canvas(
        self,
        ax21: plt.Axes,
        ax22: plt.Axes,
        ax22_t: plt.Axes,
        ax23: plt.Axes,
        ax24: plt.Axes,
        cb: mpl.colorbar.Colorbar,
        shift: float,
        crop: int,
        legend: list[str]
    ):
        """
        Initializes the additional canvas axes for the GUI.

        Args:
            ax21 (plt.Axes): The first axis.
            ax22 (plt.Axes): The second axis.
            ax22_t (plt.Axes): The third axis.
            ax23 (plt.Axes): The fourth axis.
            ax24 (plt.Axes): The fifth axis.
            cb (mpl.colorbar.Colorbar): The color bar.
            shift (float): The amount to shift the time axis.
            crop (int): The number of samples to crop from each side of the time axis.
            legend (list[str]): The legend to display on the fourth axis.

        Returns:
            None
        """
        ax22.set_aspect('auto')
        ax22_t.set_aspect('auto')
        ax22_t.set_ylabel('Amplitude (z-score)', labelpad=12.5, rotation=270)
        ax22_t.spines['top'].set_alpha(.2)
        ax22_t.spines['right'].set_alpha(.2)
        ax22_t.spines['left'].set_alpha(.2)
        ax22_t.spines['bottom'].set_alpha(.2)
        ax22_t.tick_params(axis='both', which='both', length=5, color='#00000050')
        ax22.spines['top'].set_alpha(.2)
        ax22.spines['right'].set_alpha(.2)
        ax22.spines['left'].set_alpha(.2)
        ax22.spines['bottom'].set_alpha(.2)
        ax22.tick_params(axis='both', which='both', length=5, color='#00000050')
        cb.outline.set_color('#00000020')
        cb.ax.tick_params(axis='both', which='both', length=5, color='#00000050')
        times = np.unique(np.round(self.params.temporal.times, 1))
        ranges = np.linspace(0, len(self.params.temporal.times), len(times)).astype(int)

        if shift is True:
            times = np.round(times - times.mean(), 2)
        elif isinstance(shift, (int, float)):
            times = np.round(times + shift, 2)

        ax22.set_xticks(ranges)
        ax22.set_xticklabels(times)
        ax22.set_xlabel('Time (s)')
        ax22.set_ylabel('Frequency (Hz)')
        ax23.legend(legend, loc='upper right')
        ax23.spines['top'].set_alpha(.2)
        ax23.spines['right'].set_alpha(.2)
        ax23.spines['left'].set_alpha(.2)
        ax23.spines['bottom'].set_alpha(.2)
        ax23.tick_params(axis='both', which='both', length=5, color='#00000050')
        ax23.set_xlabel('Frequency (Hz)')
        ax23.set_ylabel('Amplitude (z-score)')
        ax23.set_xlim([0, 70])
        ax22_t.set_xlim([2 * crop, len(self.params.temporal.times) - 2 * crop])

        ax24.spines['top'].set_alpha(.2)
        ax24.spines['right'].set_alpha(.2)
        ax24.spines['left'].set_alpha(.2)
        ax24.spines['bottom'].set_alpha(.2)
        ax24.set_xticks(ranges)
        ax24.set_xticklabels(times)
        ax24.set_xlabel('Time (s)')
        ax24.set_ylabel('Amplitude (z-score)')
        ax24.tick_params(axis='both', which='both', length=5, color='#00000050')

    def onclick(
        self,
        event: mpl.backend_bases.MouseEvent,
        sorting_callback: SortingCallback,
        spec_plot_elems: list[str],
        f_max: int = 70,
        timeshift: float = 0.,
        crop: float = 0.05
    ):
        """The onclick function performs various plotting operations when a mouse button is clicked over a figure.

        Args:
            event (mpl.backend_bases.MouseEvent): An object representing the event triggered by the click.
            sorting_callback (SortingCallback): An object that controls the sorting of neurons in the spatial patterns.
            spec_plot_elems (list[str]): A list of string names indicating which spectral data elements to plot.
            f_max (int, optional): The maximum frequency to plot in the induced activity plot. Default is 70.
            timeshift (float, optional): The amount of time by which to shift the temporal plots. Default is 0.
            crop (float, optional): The fraction of the total length of the temporal data to crop. Default is 0.05.

        Returns:
            None
        """

        self.main_axes[4].clear()

        _, iy = event.xdata, event.ydata

        if (event.inaxes == self.main_axes[2] or event.inaxes == self.main_axes[3] or event.inaxes == self.main_axes[4]) \
            and event.xdata is not None \
            and event.ydata is not None \
            and 0 < event.xdata < self.limits.x \
                and -.5 < event.ydata < self.limits.y:

            iy = int(np.rint(iy))

            if self.colors[sorting_callback._sorted_indices[iy]] == '#d62728':
                cmap = mpl.colors.ListedColormap(self.cmap.colors[::-1])
            else:
                cmap = self.cmap

            temp_map = np.empty(self.params.spatial.patterns.T.shape)
            temp_map[:] = np.nan
            temp_map[iy] = 1
            self.main_axes[4].imshow(temp_map, aspect='auto', cmap=cmap, alpha=.4)
            self.main_fig.canvas.draw()

            fig2 = plt.figure(constrained_layout=False)
            gs2 = fig2.add_gridspec(
                nrows=10,
                ncols=16,
                bottom=.1,
                wspace=.05,
                hspace=.1
            )

            ax21 = fig2.add_subplot(gs2[0:4, :-9])

            plot_patterns(
                self.params.spatial.patterns,
                self.info,
                sorting_callback.sorted_indices[iy],
                ax21,
                name_format='',
                cmap=self.cmap
            )

            ax22 = fig2.add_subplot(gs2[0:5, -7:])
            ax22_t = ax22.twinx()

            induced = self.params.temporal.induceds[sorting_callback.sorted_indices[iy]]

            induced_norm = rowwise_zscore(induced)

            crop = int(crop*induced.shape[1]/2)
            induced_norm = induced_norm[:f_max, crop:-crop]
            np.nan_to_num(induced_norm, copy=False)

            vmin = np.percentile(induced_norm, 1)
            vmax = np.percentile(induced_norm, 99)

            pos = ax22.imshow(
                induced_norm,
                origin='lower',
                cmap=self.cmap,
                interpolation='bicubic',
                aspect='auto',
                interpolation_stage='rgba',
                vmin=vmin,
                vmax=vmax
            )

            ax22.contour(
                induced_norm,
                [
                    np.percentile(induced_norm, 25),
                    np.percentile(induced_norm, 75)
                ],
                origin='lower',
                linewidths=.1,
                cmap=self.cmap,
            )
            ax22_t.plot(
                sp.stats.zscore(self.params.temporal.time_courses.mean(0)[sorting_callback.sorted_indices[iy]][crop:-crop]),
                linewidth=.75
            )
            cb = fig2.colorbar(
                pos,
                ax=ax22,
                pad=0.12,
                orientation='horizontal',
                aspect=75,
                fraction=.12
            )

            ax23 = fig2.add_subplot(gs2[5:, -7:])
            spec_range = np.arange(0, self.params.spectral.range[-1], .1)
            interp_cubic = lambda y: sp.interpolate.interp1d(self.params.spectral.range, y, 'cubic')(spec_range)
            spec_legend = list()

            if 'input' in spec_plot_elems:
                spec_legend.append('input')
                data = sp.stats.zscore(np.real(self.params.spectral.inputs[sorting_callback.sorted_indices[iy]].mean(0)))
                data -= data.min()
                ax23.plot(
                    spec_range,
                    sp.stats.zscore(
                        interp_cubic(data)
                    ),
                    color='tab:blue',
                    alpha=.25
                )

            if 'output' in spec_plot_elems:
                spec_legend.append('output')
                data = sp.stats.zscore(np.real(self.params.spectral.outputs[sorting_callback.sorted_indices[iy]].mean(0)))
                data -= data.min()
                ax23.plot(
                    spec_range,
                    sp.stats.zscore(
                        interp_cubic(data)
                    ),
                    color='tab:blue',
                    linewidth=.75
                )
            if 'response' in spec_plot_elems:
                spec_legend.append('response')
                data = sp.stats.zscore(np.real(self.params.spectral.responses[sorting_callback.sorted_indices[iy]]))
                data -= data.min()
                ax23.plot(
                    spec_range,
                    interp_cubic(data),
                    alpha=.75,
                    linestyle='--',
                    color='tab:red'
                )
            if 'pattern' in spec_plot_elems:
                spec_legend.append('pattern')
                data = sp.stats.zscore(np.real(self.params.spectral.patterns[sorting_callback.sorted_indices[iy]].mean(0)))
                data -= data.min()
                ax23.plot(
                    spec_range,
                    sp.stats.zscore(
                        interp_cubic(data)
                    ),
                    color='tab:blue',
                    alpha=.75,
                    linestyle=':'
                )
            ax23.legend(spec_legend, loc='upper right')
            ax23.set_xlim(0, 100)

            ax24 = fig2.add_subplot(gs2[5:, :-9])
            times = self.params.temporal.times
            temp_legend = list()
            ax24.plot(
                sp.stats.zscore(
                    self.params.temporal.time_courses.mean(0)[sorting_callback.sorted_indices[iy]]
                ),
                linewidth=2, alpha=0.25
            )
            temp_legend.append('spatially filtered')
            ax24.plot(
                sp.stats.zscore(
                    self.params.temporal.time_courses_filtered.mean(0)[sorting_callback.sorted_indices[iy]]
                ),
                color='tab:blue',
                linewidth=1
            )
            temp_legend.append('temporally filtered')
            if self.params.temporal.patterns is not None:
                ax24.plot(
                    sp.stats.zscore(
                        self.params.temporal.patterns[sorting_callback.sorted_indices[iy]],
                    ),
                    color='tab:red',
                    linewidth=1.25,
                    linestyle='--',
                    alpha=.75
                )
                temp_legend.append('temporal pattern')
            ax24.legend(temp_legend, loc='upper right')
            self.init_additional_canvas(ax21, ax22, ax22_t, ax23, ax24, cb, timeshift, crop, spec_legend)

            axfilt = fig2.add_axes([0.86, 0.925, 0.04, 0.025])# posx, posy, width, height
            filt_button = Button(axfilt, 'Filter')
            axfilt._button = filt_button
            filt_callback = FilterButtonCallback(
                filt_button,
                self,
                fig2,
                sorting_callback,
                iy,
                ax21, ax22, ax22_t, ax23, ax24,
                cb,
                timeshift,
                f_max,
                crop,
                spec_legend,
                f_scale=(vmin, vmax),
            )
            filt_button.on_clicked(filt_callback)

            plt.show()

    def plot(
        self,
        title: str = '',
        timeshift: float = 0.,
        f_max: float = 70,
        spec_plot_elems: list[str] = ['input', 'output', 'response'],
        show: bool = True
    ):
        """
        Plots the data on the main figure.

        Args:
            title (str, optional): Title of the plot. Defaults to ''.
            timeshift (float, optional): Shift in time for the plot. Defaults to 0..
            f_max (float, optional): Maximum frequency for the plot. Defaults to 70.
            spec_plot_elems (list[str], optional): List of plot elements to show. Defaults to ['input', 'output', 'response'].
            show (bool, optional): Whether to show the plot or not. Defaults to True.

        Returns:
            matplotlib.figure.Figure: The plotted figure.
        """
        sort_button = Button(self.main_axes[1], 'Sort')
        sorting_callback = SortingCallback(
            sort_button,
            self,
            self.main_fig,
            sorted(range(len(self.sums)), reverse=False)
        )
        self.init_main_canvas(sorting_callback)
        sort_button.on_clicked(sorting_callback)
        self.main_axes[2].imshow(self.params.spatial.patterns.T, aspect='auto', cmap=self.cmap)
        temp_map = np.empty(self.params.spatial.patterns.T.shape)
        temp_map[:] = np.nan
        self.main_axes[4].imshow(temp_map, aspect='auto')
        self.main_axes[3].barh(sorting_callback.sorted_indices, np.abs(self.sums), color=self.colors, height=.9)

        cid1 = self.main_fig.canvas.mpl_connect(
            'button_press_event',
            partial(
                self.onclick,
                sorting_callback=sorting_callback,
                spec_plot_elems=spec_plot_elems,
                f_max=f_max,
                timeshift=timeshift
            )
        )
        cid2 = self.main_fig.canvas.mpl_connect('close_event', lambda e: self.main_fig.canvas.mpl_disconnect(cid1))

        self.main_fig.canvas.mpl_disconnect(cid2)

        if show:
            plt.show()

        self.main_fig.suptitle(title)

        return self.main_fig
