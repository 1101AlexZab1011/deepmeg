import contextlib
from functools import partial
import os
import sys
from tqdm import tqdm
from typing import Generator, TextIO, Callable
import numpy as np
import plotille as pt
from collections.abc import Sequence


class DummyFile(object):
    """
    A class that wraps a file object, replacing the 'write' method to avoid print() second call (useless \\n).
    """
    file: TextIO
    def __init__(self, file: TextIO):
        """
        Initialize the DummyFile object.

        Args:
            - file (TextIO) : The file object to be wrapped.
        """
        self.file = file

    def write(self, x: str):
        """
        Writes the given string to the wrapped file object, avoiding print() second call (useless \\n).

        Args:
            - x (str) : the string to be written.
        """
        # Avoid print() second call (useless \n)
        if len(x.rstrip()) > 0:
            tqdm.write(x, file=self.file)

    def flush(self):
        self.file.flush()
    def isatty(self):
        return self.file.isatty()

@contextlib.contextmanager
def nostdout() -> Generator:
    """
    A context manager that temporarily redirects the stdout to a DummyFile object.
    """
    if not isinstance(sys.stdout, DummyFile):
        save_stdout = sys.stdout
    else:
        save_stdout = sys.stdout.file

    sys.stdout = DummyFile(save_stdout)
    yield
    sys.stdout = save_stdout


def edit_previous_line(text: str, line: int = 1, *, return_str: bool = False) -> str | None:
    """
    Edits the content of a previous line in the console output.

    Args:
        text (str): The updated text to replace the content of the previous line.
        line (int, optional): The number of lines to move up from the current line. Default is 1.
        return_str (bool, optional): Flag indicating whether to return the modified string instead of printing it.
                                     Default is False.

    Returns:
        None or str: If `return_str` is True, the modified string with appropriate line breaks is returned.
                    Otherwise, the modified string is printed to the console.
    """
    out = f'\033[{line}F\033[K{text}'
    for i in range(line - 1):
        out += '\n'
    if return_str:
        return out
    else:
        print(out)

def delete_previous_line(line: int = 1, *, return_str: bool = False) -> str | None:
    """
    Deletes the content of a previous line in the console output.

    Args:
        line (int, optional): The number of lines to move up from the current line. Default is 1.
        return_str (bool, optional): Flag indicating whether to return the modified string instead of printing it.
                                     Default is False.

    Returns:
        None or str: If `return_str` is True, the modified string with appropriate line breaks is returned.
                    Otherwise, the modified string is printed to the console.
    """
    out = f'\033[{line}F\033[M\033[A'
    for i in range(line - 1):
        out += '\n'
    if return_str:
        return out
    else:
        print(out)


def erase_previous_line(line: int = 1, *, return_str: bool = False) -> str | None:
    """
    Erases the content of a previous line in the console output.

    Args:
        line (int, optional): The number of lines to move up from the current line. Default is 1.
        return_str (bool, optional): Flag indicating whether to return the modified string instead of printing it.
                                     Default is False.

    Returns:
        None or str: If `return_str` is True, the modified string with appropriate line breaks is returned.
                    Otherwise, the modified string is printed to the console.
    """
    out = f'\033[{line}F\033[K'
    for i in range(line - 1):
        out += '\n'
    if return_str:
        return out
    else:
        print(out)


def add_line_above(
    text: str = '',
    line: int = 1,
    *,
    return_str: bool = False
) -> str | None:
    """
    Adds a line with specified text above the current line in the console output.

    Args:
        text (str, optional): The text to be added above the current line. Default is an empty string.
        line (int, optional): The number of lines to move up from the current line. Default is 1.
        return_str (bool, optional): Flag indicating whether to return the modified string instead of printing it.
                                     Default is False.

    Returns:
        None or str: If `return_str` is True, the modified string with appropriate line breaks is returned.
                    Otherwise, the modified string is printed to the console.
    """
    out = f'\033[{line}F\033[L{text}'
    for i in range(line):
        out += '\n'
    if return_str:
        return out
    else:
        print(out)


def insert_elem(lst: list, from_index: int, to_index: int) -> None:
    lst.insert(to_index, lst.pop(from_index))

def swith_elems(lst: list, idx1: int, idx2: int) -> None:
    lst[idx1], lst[idx2] = lst[idx2], lst[idx1]

def str_tick(min_, max_, n=0, transform=None):
    if transform:
        min_ = transform(min_)
    formatter = '{' + f':.{n}f' + '}'
    return formatter.format(min_)

def scatter(
    *args: np.ndarray,
    width: int = 80,
    height: int = 40,
    X_label: str = 'X',
    Y_label: str = 'Y',
    linesep: str = os.linesep,
    x_min: float = None,
    x_max: float = None,
    y_min: float = None,
    y_max: float = None,
    color: str | list[str] = None,
    bg: str = None,
    color_mode: str = 'names',
    origin: bool = True,
    markers: str = None,
    xtick_fmt: Callable[[int, int], str] = str_tick,
    ytick_fmt: Callable[[int, int], str] = partial(str_tick, n=2),
    curve_labels : list[str] = None,
    current_value_index: int = None,
) -> str:
    """
    Creates a scatter plot in the console using the plottile library.

    Args:
        *args (np.ndarray): Multiple arrays representing X and Y coordinates of points to be plotted.
        width (int, optional): Width of the plot in characters. Default is 80.
        height (int, optional): Height of the plot in characters. Default is 40.
        X_label (str, optional): Label for the X-axis. Default is 'X'.
        Y_label (str, optional): Label for the Y-axis. Default is 'Y'.
        linesep (str, optional): Line separator character. Default is os.linesep.
        x_min (float, optional): Minimum value for the X-axis.
        x_max (float, optional): Maximum value for the X-axis.
        y_min (float, optional): Minimum value for the Y-axis.
        y_max (float, optional): Maximum value for the Y-axis.
        color (str or list[str], optional): Color or list of colors for the scatter plot. Default is None.
        bg (str, optional): Background color of the plot. Default is None.
        color_mode (str, optional): Color mode for the plot. Default is 'names'.
        origin (bool, optional): Flag indicating whether to display the origin (0, 0) on the plot. Default is True.
        markers (str, optional): Marker style or list of marker styles for the scatter plot. Default is None.
        xtick_fmt (Callable[[int, int], str], optional): Function to format X-axis tick labels. Default is str_tick.
        ytick_fmt (Callable[[int, int], str], optional): Function to format Y-axis tick labels. Default is str_tick.
        curve_labels (list[str], optional): List of labels for each curve in the plot. Default is None.
        current_value_index (int, optional): Index of the current value to be displayed on the plot. Default is None.

    Returns:
        str: The scatter plot representation as a string.

    Raises:
        ValueError: If the number of parameters to plot is not even.
        ValueError: If markers for curves are inconsistent with the number of curves.
        ValueError: If labels for curves are inconsistent with the number of curves.
        ValueError: If colors and lines to plot are inconsistent.
    """

    if len(args) == 1:
        t = [np.linspace(0, len(args), len(args))]
        y = [args]
    elif len(args) % 2:
        raise ValueError('Number of parameters to plot should be even')
    else:
        t = [x for i, x in enumerate(args) if not i%2]
        y = [x for i, x in enumerate(args) if i%2]

    if markers is None:
        markers = [None for _ in range(len(y))]
    elif isinstance(markers, str):
        markers = [markers for _ in range(len(y))]
    elif len(markers) != len(y):
        raise ValueError('Markers for curves are inconsistent with the number of curves')

    if curve_labels is None:
        curve_labels = [None for _ in range(len(y))]
    elif isinstance(curve_labels, str):
        curve_labels = [curve_labels for _ in range(len(y))]
    elif len(curve_labels) != len(y):
        raise ValueError(f'Labels for curves are inconsistent with the number of curves')

    if isinstance(color, str) or not isinstance(color, Sequence):
        color = [color for _ in range(len(t))]
    elif len(color) != len(t):
        raise ValueError('Colors and lines to plot are inconsistent')

    fig = pt.Figure()
    fig.x_ticks_fkt = xtick_fmt
    fig.y_ticks_fkt = ytick_fmt
    fig.width = width
    fig.height = height
    fig.x_label = X_label
    fig.y_label = Y_label
    fig.linesep = linesep
    fig.origin = origin

    if x_min is not None:
        fig.set_x_limits(min_=x_min)
    if x_max is not None:
        fig.set_x_limits(max_=x_max)
    if y_min is not None:
        fig.set_y_limits(min_=y_min)
    if y_max is not None:
        fig.set_y_limits(max_=y_max)

    fig.background = bg
    fig.color_mode = color_mode

    if color is None and bg is None:
        fig.with_colors = False

    for X, Y, c, marker, label in zip(t, y, color, markers, curve_labels):
        fig.scatter(X, Y, c, marker=marker, label=label)
        if current_value_index:
            fig.text(
                [X[current_value_index]],
                [Y[current_value_index]],
                ['x {:.3f}'.format(Y[current_value_index])],
                lc='white'
            )

    return fig.show(legend=True)

class MetricsConsolePlotter:
    """A class for plotting deep learning metrics and losses into the console while learning.

    Args:
        n_epochs (int): The total number of epochs.
        width (int, optional): The width of the plot in characters. Default is 40.
        height (int, optional): The height of the plot in characters. Default is 5.
        loss_colors (str or list[str], optional): Colors for loss curves. Default is None.
        metric_colors (str or list[str], optional): Colors for metric curves. Default is None.
        loss_label (str, optional): Label for the loss plot. Default is 'Loss'.
        metric_label (str, optional): Label for the metric plot. Default is 'Metric'.
        metric_names (list[str], optional): Names for the metric curves. Default is ['train', 'val'].
        loss_names (list[str], optional): Names for the loss curves. Default is ['train', 'val'].
        metric_range (tuple(float, float), optional): Range for the metric plot. Default is (0, 102).

    Methods:
        __call__(self, loss, metric):
            Plots the loss and metric values and returns the plot data as a list of strings.
    """
    @staticmethod
    def __validate_list(value):
        """Validate and convert a value to a list if necessary.

        Args:
            value: The value to validate.

        Returns:
            list: The validated list.
        """
        if isinstance(value, Sequence) and not isinstance(value, str):
            if isinstance(value, list):
                return value
            elif isinstance(value, tuple):
                return list(value)
            else:
                return [val for val in value]
        else:
            return [value]

    @staticmethod
    def __min_value(values: list[list[float]]) -> float:
        """Find the minimum value in a nested list of floats.

        Args:
            values (list[list[float]]): The nested list of floats.

        Returns:
            float: The minimum value.
        """
        return min([min(data) for data in values])

    @staticmethod
    def __max_value(values: list[list[float]]) -> float:
        """Find the maximum value in a nested list of floats.

        Args:
            values (list[list[float]]): The nested list of floats.

        Returns:
            float: The maximum value.
        """
        return max([max(data) for data in values])

    @staticmethod
    def __zip_data(data1: list, data2: list) -> list:
        """Zip two lists of metric & losses data together.

        Args:
            data1 (list): The first list of data.
            data2 (list): The second list of data.

        Returns:
            list: The zipped list.
        """
        out = list()

        for sample1, sample2 in zip(data1, data2):
            out.append(sample1)
            out.append(sample2)

        return out

    @staticmethod
    def __update_history(history: list[list[float]], values: list[float]) -> None:
        """Update the ongoing history to plot with new values.

        Args:
            history (list[list[float]]): The history of values.
            values (list[float]): The new values to add to the history.

        Returns:
            None
        """
        if history:
            for i, v in enumerate(values):
                history[i].append(v)
        else:
            for v in values:
                history.append([v])

    @staticmethod
    def __fill_seq(sequence1: Sequence, sequence2: Sequence, value: float = 0):
        """Fill a sequence with a value to match the length of another sequence.

        Args:
            sequence1 (Sequence): The first sequence.
            sequence2 (Sequence): The second sequence.
            value (float, optional): The value to fill. Default is 0.

        Returns:
            list: The filled sequence.
        """
        if len(sequence1) > len(sequence2):
            sequence1, sequence2 = sequence2, sequence1
        diff = len(sequence2) - len(sequence1)
        if isinstance(sequence1, tuple):
            out = list(sequence1)
        elif isinstance(sequence1, list):
            out = sequence1
        else:
            out = [point for point in sequence1]

        return out + [value for _ in range(diff)]

    def __init__(
        self,
        n_epochs: int,
        width: int = 40,
        height: int = 5,
        loss_colors: str | list[str] = None,
        metric_colors: str | list[str] = None,
        loss_label = 'Loss',
        metric_label = 'Metric',
        metric_names = ['train', 'val'],
        loss_names = ['train', 'val'],
        metric_range = (0, 102),
    ):
        """Initialize the MetricsConsolePlotter.

        Args:
            n_epochs (int): The total number of epochs.
            width (int, optional): The width of the plot in characters. Default is 40.
            height (int, optional): The height of the plot in characters. Default is 5.
            loss_colors (str or list[str], optional): Colors for loss curves. Default is None.
            metric_colors (str or list[str], optional): Colors for metric curves. Default is None.
            loss_label (str, optional): Label for the loss plot. Default is 'Loss'.
            metric_label (str, optional): Label for the metric plot. Default is 'Metric'.
            metric_names (list[str], optional): Names for the metric curves. Default is ['train', 'val'].
            loss_names (list[str], optional): Names for the loss curves. Default is ['train', 'val'].
            metric_range (tuple(float, float), optional): Range for the metric plot. Default is (0, 102).

        Returns:
            None
        """
        self.n_epochs = n_epochs
        self.t = np.linspace(0, n_epochs, n_epochs)
        self.losses = list()
        self.metrics = list()
        self.width = width
        self.height = height
        self.n_lines = None
        self.loss_colors = self.__validate_list(loss_colors) if loss_colors is not None else loss_colors
        self.metric_colors = self.__validate_list(metric_colors) if metric_colors is not None else metric_colors
        self.loss_label = loss_label
        self.metric_label = metric_label
        self.metric_names = metric_names
        self.loss_names = loss_names
        self.metric_range = metric_range

    def __call__(self, loss: float | list[float], metric: float | list[float]) -> list[str]:
        """Plot the loss and metric values and return the plot data as a list of strings.

        Args:
            loss (float or list[float]): The loss value(s) to plot.
            metric (float or list[float]): The metric value(s) to plot.

        Returns:
            list[str]: The plot data as a list of strings.
        """
        loss = self.__validate_list(loss)
        metric = self.__validate_list(metric)
        self.__update_history(self.losses, loss)
        self.__update_history(self.metrics, metric)

        if self.loss_colors is not None and len(self.loss_colors) < len(self.losses):
            self.loss_colors += [self.loss_colors[-1] for _ in range(len(self.losses) - len(self.loss_colors))]

        if self.metric_colors is not None and len(self.metric_colors) < len(self.metrics):
            self.metric_colors += [self.metric_colors[-1] for _ in range(len(self.metrics) - len(self.metric_colors))]

        y_min = self.__min_value(self.losses)
        y_max = self.__max_value(self.losses)

        if y_min == y_max:
            y_min -= 1e3
            y_max += 1e3

        y_min -= .2*abs(y_min)
        y_max += .2*abs(y_max)

        losses = [self.__fill_seq(loss, self.t, y_min-1) for loss in self.losses]

        metrics = [self.__fill_seq(metric, self.t, -100) for metric in self.metrics]

        xlabel = f'Epoch {len(self.losses[0])}/{self.n_epochs}'
        plot_data_1 = scatter(
            *self.__zip_data([self.t for _ in range(len(self.losses))], losses),
            color=self.loss_colors,
            height=self.height, width=self.width, X_label=xlabel, Y_label=self.loss_label,
            x_min=0, x_max=self.n_epochs,
            y_min=y_min, y_max=y_max,
            curve_labels=self.loss_names,
            current_value_index=len(self.metrics[0]) - 1
        )
        plot_data_2 = scatter(
            *self.__zip_data([self.t for _ in range(len(self.metrics))], metrics),
            color=self.metric_colors,
            height=self.height, width=self.width, X_label=xlabel, Y_label=self.metric_label,
            x_min=0, x_max=self.n_epochs,
            y_min=self.metric_range[0], y_max=self.metric_range[1],
            ytick_fmt=partial(str_tick, n=0, transform=lambda x: round(x/10)*10),
            curve_labels=self.metric_names,
            current_value_index=len(self.metrics[0]) - 1
        )
        sep = '| '
        lines_1 = plot_data_1.split(os.linesep)
        lines_2 = plot_data_2.split(os.linesep)
        legend_lines = max(len(self.metrics), len(self.losses))

        if len(lines_2) < len(lines_1):
            for _ in range(len(lines_1) - len(lines_2)):
                lines_2.append('')
        elif len(lines_1) < len(lines_2):
            for _ in range(len(lines_2) - len(lines_1)):
                lines_1.append('')

        lines_1 = [sep + line for line in lines_1]
        max_line = len(lines_1[-4-legend_lines]) + len(xlabel)
        lines_1[0] += ' ' * (  - len(lines_1[0]) + max_line) + sep
        lines_1[1] += ' ' * (max_line - len(lines_1[1])) + sep

        for row_idx in range(self.height):
            lines_1[2 + row_idx] += ' ' * (max_line - 15 - self.width) + sep

        for i in range(5):
            d = -i - legend_lines - 1
            lines_1[d] += ' ' * (max_line - len(lines_1[d])) + sep

        if len(self.losses) >= len(self.metrics):
            vals = [-i-1 for i in range(legend_lines)]
        else:
            vals = [-(i + 1) for i in range(legend_lines) if i > len(self.metrics) - len(self.losses) - 1]

        for i in range(legend_lines):
            d = - i - 1
            if d in vals:
                add = 9
            else:
                add = 0
            # lines_1[d] = lines_1[d][0] + ' '*(max_line - len(lines_1[d]) + add - 3) + lines_1[d][1:]
            # if lines_2[d]:
            #     lines_2[d] = ' ' * (max_line - len(lines_2[d]) + add - 3) + lines_2[d]
            lines_1[d] += ' ' * (max_line - len(lines_1[d]) + add) + sep


        insert_elem(lines_1, -legend_lines-3, 0)
        insert_elem(lines_2, -legend_lines-3, 0)

        for i in range(legend_lines+2):
            insert_elem(lines_1, -1, 0)
            insert_elem(lines_2, -1, 0)

        lines_1.insert(0, lines_1[legend_lines+2])
        lines_2.insert(0, lines_2[legend_lines+2])

        plot_data = [l1 + l2 for (l1, l2) in zip(lines_1, lines_2)]

        return plot_data
