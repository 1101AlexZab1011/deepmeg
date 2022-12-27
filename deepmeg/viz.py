import scipy as sp
import mne
import matplotlib.pyplot as plt
import matplotlib as mp
import matplotlib as mpl
import copy
import numpy as np
from typing import Optional, Union, NoReturn
from .params import CroppingParameters, SpatialParameters, TemporalParameters, WaveForms, moving_average
from dataclasses import dataclass
from matplotlib.widgets import Button
from collections.abc import Iterable


def plot_patterns(
    patterns, info, order=None, axes=None, cmap='RdBu_r', sensors=True,
    colorbar=False, res=64,
    size=1, cbar_fmt='%3.1f', name_format='Latent\nSource %01d',
    show=True, show_names=False,
    outlines='head', contours=6,
    image_interp='linear'
) -> mpl.figure.Figure:
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


def plot_spectra(temporal_parameters, order, title='', xlim=None, ylim=None, legend=None):

    if not len(order) % 3:
        n_cols = 3
        n_rows = len(order) // 3
    elif len(order) == 3:
        n_rows = 1
        n_cols = 3
    else:
        n_rows = len(order) // 3 + 1
        n_cols = 3

    if legend is None:
        legend = ['Filter input', 'Filter output', 'Filter responce']

    fig, axs = plt.subplots(n_rows, n_cols)

    if len(axs.shape) == 1:
        axs = np.reshape(axs, (1, -1))

    fig.set_size_inches(n_cols * 5, n_rows * 3.75)
    fig.suptitle(title)
    fig.tight_layout()
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=.5)

    current_comp = 0

    for i in range(n_rows):
        for j in range(n_cols):
            if current_comp < len(order):
                n_component = order[current_comp]
                axs[i, j].set_title(f'Latent Source {n_component}')
                axs[i, j].plot(
                    temporal_parameters.franges,
                    temporal_parameters.finputs[n_component],
                    temporal_parameters.franges,
                    temporal_parameters.foutputs[n_component],
                    temporal_parameters.franges,
                    temporal_parameters.fresponces[n_component],
                )

                if xlim:
                    axs[i, j].set_xlim(xlim)

                if ylim:
                    axs[i, j].set_ylim(ylim)
            else:
                axs[i, j].set_xticks([])
                axs[i, j].set_yticks([])
                axs[i, j].spines['top'].set_visible(False)
                axs[i, j].spines['right'].set_visible(False)
                axs[i, j].spines['bottom'].set_visible(False)
                axs[i, j].spines['left'].set_visible(False)

            current_comp += 1

    fig.legend(legend, loc='upper right')
    return fig


def plot_tempospectral(
    spatial_parameters: Union[SpatialParameters, list[SpatialParameters]],
    temporal_parameters: Union[TemporalParameters, list[TemporalParameters]],
    orders: Union[np.ndarray, list[np.ndarray]],
    info: mne.Info,
    subject_names: Optional[Union[str, list[str]]] = None,
    class_names: Optional[Union[str, list[str]]] = None,
    title: Optional[str] = None,
    xlim: Optional[Union[int, float]] = None,
    ylim: Optional[Union[int, float]] = None,
    legend: Optional[Union[int, float]] = None,
    spatial_data_type: Optional[str] = 'patterns',
    topomap_kwargs: Optional[dict] = None,
) -> mp.figure.Figure:

    def wrap_in_list(content):
        return [content] if not isinstance(content, list) else content

    def validate_length(*iterables):
        length = len(iterables[0])
        for i, sample in enumerate(iterables[1:]):
            if len(sample) != length:
                raise ValueError(
                    'Length validation failed: all elements must have length equal '
                    f'to {length}, but element {i} has length: {len(sample)}'
                )
        return length

    spatial_parameters = wrap_in_list(spatial_parameters)
    temporal_parameters = wrap_in_list(temporal_parameters)
    info = wrap_in_list(info)

    if isinstance(orders, np.ndarray):
        n_components = len(orders)
    elif isinstance(orders[0], np.ndarray):
        n_components = len(orders[0])
    else:
        raise ValueError('"orders" must be either np.ndarray or list of np.ndarray')

    orders = wrap_in_list(orders)
    n_subjects = validate_length(spatial_parameters, temporal_parameters, info, orders)

    if subject_names is None:
        subject_names = [f'Subject {i}' for i in range(n_subjects)]
    elif isinstance(subject_names, str):
        subject_names = [f'{subject_names} {i}' for i in range(n_subjects)]
    elif isinstance(subject_names, list):

        if len(subject_names) != n_subjects:
            raise ValueError('Not all subjects have names provided')

    if class_names is None:
        class_names = [f'Class {i}' for i in range(n_components)]
    elif isinstance(class_names, str):
        subject_names = [f'{class_names} {i}' for i in range(n_components)]
    elif isinstance(class_names, list):

        if len(class_names) != n_components:
            raise ValueError('Not all classes have names provided')

    if not n_components % 3:
        n_cols = 3
        n_rows = n_components // 3
    elif n_components == 3:
        n_rows = 1
        n_cols = 3
    elif n_components == 4:
        n_rows = 1
        n_cols = 4
    elif not n_components % 4:
        n_cols = 4
        n_rows = n_components // 4
    elif not n_components % 2:
        n_cols = 2
        n_rows = n_components // 2
    else:
        n_rows = n_components // 3 + 1
        n_cols = 3

    n_rows_per_subject = 2 * n_rows
    n_rows = n_subjects * n_rows_per_subject

    if legend is None:
        legend = ['Filter input', 'Filter output', 'Filter responce']

    fig, axs = plt.subplots(n_rows, n_cols)

    if len(axs.shape) == 1:
        axs = np.reshape(axs, (1, -1))

    fig.set_size_inches(n_cols * 5, n_rows * 3.75)

    subplots_map = np.ones((n_rows, n_cols)).astype(bool)

    current_comp = 0
    for i in range(0, n_rows, 2):
        for j in range(n_cols):

            if current_comp >= n_components:
                subplots_map[i, j], subplots_map[i + 1, j] = False, False

            current_comp += 1

        if current_comp >= n_components:
            current_comp = 0

    @dataclass
    class Tracker(object):
        top: int
        bottom: int
        subject: int

    tracker = Tracker(0, 0, 0)
    for i in range(n_rows):
        for j in range(n_cols):

            if subplots_map[i, j]:

                if i % 2 and j == 0:
                    axs[i, j].set_ylabel(subject_names[tracker.subject])
                    axs[i, j].tick_params(axis='y', pad=300)
                elif not i % 2 and j == 0:
                    axs[i, j].set_ylabel(subject_names[tracker.subject])

                if not i % 2:
                    axs[i, j].set_title(
                        'Latent Source '
                        f'{orders[tracker.subject][tracker.top]} ({class_names[tracker.top]})'
                    )
                    axs[i, j].plot(
                        temporal_parameters[tracker.subject].franges,
                        temporal_parameters[tracker.subject]
                        .finputs[orders[tracker.subject][tracker.top]],
                        temporal_parameters[tracker.subject].franges,
                        temporal_parameters[tracker.subject]
                        .foutputs[orders[tracker.subject][tracker.top]],
                        temporal_parameters[tracker.subject].franges,
                        temporal_parameters[tracker.subject]
                        .fresponces[orders[tracker.subject][tracker.top]],
                    )

                    axs[i, j].legend(legend, loc='upper right')

                    if xlim:
                        axs[i, j].set_xlim(xlim)

                    if ylim:
                        axs[i, j].set_ylim(ylim)
                    tracker.top += 1

                else:
                    subject_info = copy.deepcopy(info[tracker.subject])
                    subject_info['sfreq'] = 1.

                    if spatial_data_type == 'patterns':
                        data = spatial_parameters[tracker.subject].patterns
                    elif spatial_data_type == 'filters':
                        data = spatial_parameters[tracker.subject].filters

                    patterns = mne.EvokedArray(data, subject_info, tmin=0)

                    topomap_parameters = dict(
                        times=orders[tracker.subject][tracker.bottom],
                        time_format='',
                        cmap='RdBu_r', colorbar=False, res=64,
                        units=None, time_unit='s',
                        size=1, outlines='head', contours=6,
                        image_interp='bilinear', show=False,
                        axes=axs[i, j]
                    )

                    if topomap_kwargs is not None:
                        topomap_parameters.update(topomap_kwargs)

                    patterns.plot_topomap(**topomap_parameters)
                    tracker.bottom += 1
            else:
                axs[i, j].set_xticks([])
                axs[i, j].set_yticks([])
                axs[i, j].spines['top'].set_visible(False)
                axs[i, j].spines['right'].set_visible(False)
                axs[i, j].spines['bottom'].set_visible(False)
                axs[i, j].spines['left'].set_visible(False)

        if tracker.bottom >= n_components:
            assert tracker.bottom == tracker.top, 'Tracker\'s top and bottom do not match'
            tracker.top = 0
            tracker.bottom = 0
            tracker.subject += 1

    if title is not None:
        fig.suptitle(title, fontsize=20)

    return fig


def plot_spatial_weights(
    spatial_parameters: SpatialParameters,
    temporal_parameters: TemporalParameters,
    waveforms: WaveForms,
    info: mne.Info,
    croppings: Optional[CroppingParameters] = None,
    summarize: Optional[Union[str, list[float]]] = 'sum',
    title: Optional[str] = 'Spatial Patterns',
    temp_params: Optional[list[str]] = ['input', 'response', 'output'],
    show: Optional[bool] = True,
    logscale: Optional[bool] = False,
    shift_induced_times: Optional[bool | float] = False,
    filtered_induced: Optional[bool] = False
) -> Union[mp.figure.Figure, NoReturn]:

    mp.use('Qt5Agg')

    def init_canvas(ax01, ax02, ax1, ax2):
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
        ax1.set_yticks(np.arange(y_lim))
        ax1.set_yticklabels(labels=[i + 1 for i in sorting_callback.sorted_indices])
        ax1.tick_params(axis='both', which='both', length=5, color='#00000050')

        ax2.axes.yaxis.set_visible(False)
        ax2.axes.xaxis.set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.spines['top'].set_alpha(0.2)
        ax2.spines['right'].set_alpha(0.2)
        ax2.spines['bottom'].set_alpha(0.2)

    class SortingCallback:

        def __init__(
            self,
            button: Button,
            fig: plt.Figure,
            bar_ax: mp.axes.Axes,
            imshow_ax: mp.axes.Axes,
            indices: list[int]
        ):
            self._button = button
            self._fig = fig
            self._bar_ax = bar_ax
            self._imshow_ax = imshow_ax
            self._event = None
            self._sorted_indices = indices

        def __call__(self, event):
            self._event = event

            if '▼' in self._button.label._text:
                self.decreasing()
            else:
                self.increasing()

        @property
        def sorted_indices(self):
            return self._sorted_indices

        @sorted_indices.setter
        def sorted_indices(self, value):
            raise AttributeError('Impossible to set indices directly')

        def increasing(self):
            self._button.label.set_text('Sort ▼')
            self._sorted_indices = sorted(range(len(sums)), key=lambda k: sums[k], reverse=True)
            self.update()

        def decreasing(self):
            self._button.label.set_text('Sort ▲')
            self._sorted_indices = sorted(range(len(sums)), key=lambda k: sums[k])
            self.update()

        def update(self):
            self._imshow_ax.clear()
            self._imshow_ax.imshow(data.T[self._sorted_indices, :], aspect='auto', cmap='RdBu_r')
            self._bar_ax.clear()
            self._bar_ax.barh(
                range(len(sums)),
                np.abs(sums)[self._sorted_indices], color=colors[self._sorted_indices],
                height=.9
            )
            init_canvas(ax01, ax02, ax1, ax2)
            self._fig.canvas.draw()

    def onclick(event):
        flim = 70
        crop = .05
        shift = shift_induced_times

        if ax1.lines:

            for i in range(len(ax1.lines)):
                ax1.lines.remove(ax1.lines[i])

        _, iy = event.xdata, event.ydata

        if (event.inaxes == ax1 or event.inaxes == ax2) \
            and event.xdata is not None \
            and event.ydata is not None \
            and 0 < event.xdata < x_lim \
                and -.5 < event.ydata < y_lim:
            iy = int(np.rint(iy))
            induced = waveforms.induced.copy()[
                sorting_callback.sorted_indices[iy],
                :flim,
                :
            ] if not filtered_induced else\
                waveforms.induced_filt.copy()[
                sorting_callback.sorted_indices[iy],
                :flim,
                :
            ]
            crop *= induced.shape[1] / 2

            for i, ind_course in enumerate(induced):
                induced[i] /= ind_course.mean()

            color = colors[sorting_callback._sorted_indices[iy]]
            line = mp.lines.Line2D([0, x_lim], [iy, iy], color=color, linewidth=2.56*data.shape[0]/data.shape[1], alpha=.4)
            ax1.add_line(line)
            fig1.canvas.draw()
            fig2 = plt.figure(constrained_layout=False)
            if croppings is None:
                gs2 = fig2.add_gridspec(
                    nrows=10,
                    ncols=3,
                    bottom=.1,
                    wspace=.05,
                    hspace=.1
                )
                ax21 = fig2.add_subplot(gs2[:, :-1])
                ax22 = fig2.add_subplot(gs2[0:5, -1])
                ax23 = fig2.add_subplot(gs2[5:, -1])
            else:
                gs2 = fig2.add_gridspec(
                    nrows=10,
                    ncols=16,
                    bottom=.1,
                    wspace=.05,
                    hspace=.1
                )
                ax21 = fig2.add_subplot(gs2[0:4, :-8])
                ax22 = fig2.add_subplot(gs2[0:5, -7:])
                ax23 = fig2.add_subplot(gs2[5:, -7:])
                ax24 = fig2.add_subplot(gs2[5:, :-9])

            plot_patterns(
                data,
                info,
                sorting_callback.sorted_indices[iy],
                ax21,
                name_format=''
            )
            ax22_t = ax22.twinx()
            evoked = waveforms.evoked[sorting_callback.sorted_indices[iy]] if not filtered_induced else\
                waveforms.evoked_filt[sorting_callback.sorted_indices[iy]]
            ax22_t.plot(
                sp.stats.zscore(evoked),
                '#454545'
            )
            pos = ax22.imshow(
                induced,
                cmap='RdBu_r',
                origin='lower'
            )
            cb = fig2.colorbar(
                pos,
                ax=ax22,
                pad=0.12,
                orientation='horizontal',
                aspect=75,
                fraction=.12
            )
            ax22.set_aspect('auto')
            ax22_t.set_aspect('auto')
            # ax22_t.set_ylim(top=1, bottom=-1)
            legend = list()
            for param in temp_params:
                match param:
                    case 'input':
                        legend.append('Filter input')
                        ax23.plot(
                            temporal_parameters.franges,
                            sp.stats.zscore(temporal_parameters.finputs[sorting_callback.sorted_indices[iy]]),
                            color='tab:blue'
                        )
                    case 'output':
                        legend.append('Filter output')
                        ax23.plot(
                            temporal_parameters.franges,
                            sp.stats.zscore(temporal_parameters.foutputs[sorting_callback.sorted_indices[iy]]),
                            color='tab:orange'
                        )
                    case 'response':
                        legend.append('Filter response')
                        ax23.plot(
                            temporal_parameters.franges,
                            sp.stats.zscore(
                                temporal_parameters.fresponces[sorting_callback.sorted_indices[iy]]
                            ),
                            color='tab:green'
                        )
                    case 'pattern':
                        legend.append('Filtering pattern')
                        ax23.plot(
                            sp.stats.zscore(
                                temporal_parameters.fpatterns[sorting_callback.sorted_indices[iy]]
                            ),
                            color='tab:pink'
                        )
                    case _:
                        raise NotImplementedError(f'Temporal parameter {param} is not implemented')

            ax22_t.set_ylabel('Amplitude', labelpad=12.5, rotation=270)
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
            times = np.unique(np.round(waveforms.times, 1))
            ranges = np.linspace(0, len(waveforms.times), len(times)).astype(int)

            if shift is True:
                times = np.round(times - times.mean(), 2)
            elif isinstance(shift, (int, float)):
                times = np.round(times - shift, 2)

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
            # ax23.set_ylim(top=1.2)
            ax23.set_xlim([0, 70])
            ax22_t.set_xlim([2 * crop, len(waveforms.times) - 2 * crop])

            if logscale:
                ax23.set_yscale('log')

            if croppings:
                ax24.spines['top'].set_alpha(.2)
                ax24.spines['right'].set_alpha(.2)
                ax24.spines['left'].set_alpha(.2)
                ax24.spines['bottom'].set_alpha(.2)
                envelope = croppings.envelopes[sorting_callback.sorted_indices[iy]]
                envelope = np.reshape(envelope, (-1, len(waveforms.times)))
                ax24.plot(
                    moving_average(envelope.mean(0), 5)
                )
                ax24.plot(
                    moving_average(croppings.eig_estimate[sorting_callback.sorted_indices[iy]])
                )
                ax24.plot(
                    moving_average(croppings.loss_estimate[sorting_callback.sorted_indices[iy]])
                )
                ax24.set_xticks(ranges)
                ax24.set_xticklabels(times)
                ax24.set_xlabel('Time (s)')
                ax24.set_ylabel('Amplitude (z-score)')
                ax24.tick_params(axis='both', which='both', length=5, color='#00000050')
                ax24.legend(['Envelope evoked', 'Temporal pattern', 'Loss-based estimate'], loc='upper right')

            fig2.suptitle(f'Latent source {sorting_callback.sorted_indices[iy] + 1}')
            plt.show()

    data = spatial_parameters.patterns.copy()
    x_lim, y_lim = data.shape
    fig1 = plt.figure()
    fig1.suptitle(title)
    gs = fig1.add_gridspec(2, 2, hspace=0, wspace=0, width_ratios=[1, .1], height_ratios=[.025, 1])

    (ax01, ax02), (ax1, ax2) = gs.subplots(sharex='col', sharey='row')

    if summarize == 'sum':
        sums = np.sum(data, axis=0)
    elif summarize == 'sumabs':
        sums = np.sum(np.abs(data), axis=0)
    elif summarize == 'abssum':
        sums = np.abs(np.sum(data, axis=0))
    elif isinstance(summarize, Iterable) and len(summarize) == y_lim:
        sums = np.array(summarize)
    else:
        raise NotImplementedError(
            f'The "{summarize}" method not implemented. '
            'Available methods: "sum", "sumabs", "abssum"'
        )
    colors = np.array(['#f2827a' if sum_ >= 0 else '#8bbae5' for sum_ in sums])

    sort_button = Button(ax02, 'Sort')
    sorting_callback = SortingCallback(
        sort_button,
        fig1,
        ax2,
        ax1,
        sorted(range(len(sums)), reverse=False)
    )
    sort_button.on_clicked(sorting_callback)

    init_canvas(ax01, ax02, ax1, ax2)

    ax1.imshow(data.T, aspect='auto', cmap='RdBu_r')
    ax2.barh(sorting_callback.sorted_indices, np.abs(sums), color=colors, height=.9)

    cid1 = fig1.canvas.mpl_connect('button_press_event', onclick)
    cid2 = fig1.canvas.mpl_connect('close_event', lambda e: fig1.canvas.mpl_disconnect(cid1))

    if show:
        plt.show()

    fig1.canvas.mpl_disconnect(cid2)

    return fig1