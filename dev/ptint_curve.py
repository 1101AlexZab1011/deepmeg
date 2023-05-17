import os
import sys
sys.path.insert(1, os.path.realpath(os.path.pardir))
sys.path.insert(1, './')
import plotille as pt
import numpy as np
import time
from deepmeg.utils.printout import nostdout
from collections.abc import Sequence
from functools import partial

def edit_previous_line(text: str, line: int = 1, *, return_str: bool = False):
    out = f'\033[{line}F\033[K{text}'
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
):
    out = f'\033[{line}F\033[L{text}'
    for i in range(line):
        out += '\n'
    if return_str:
        return out
    else:
        print(out)


n_lines = None
x = np.linspace(0, 10, 100)

# fig = pt.Figure()
# fig.height = 10
# fig.width = 40

# sin_strings = pt.scatter(x, np.sin(x + 10), lc='red', height = 10, width= 20).split('\n')
# # sin_strings = fig.show().split('\n')
# # fig.clear()
# cos_strings = pt.scatter(x, np.cos(x + 10), lc='red', height = 10, width= 20).split('\n')
# # cos_strings = fig.show().split('\n')

# if n_lines is None:
#     n_lines = len(sin_strings)
#     lines = [ line1 + line2.rjust(40, 'p') for line1, line2 in zip(sin_strings, cos_strings)]
#     print('\n'.join(lines))


# for i in range(1000):

#     fig = pt.Figure()
#     fig.height = 10
#     fig.width = 40

#     fig.scatter(x, np.sin(x + i), lc='red')
#     sin_strings = fig.show().split('\n')
#     fig.clear()
#     fig.scatter(x, np.cos(x + i), lc='red')
#     cos_strings = fig.show().split('\n')
#     max_line = len(lines_1[-1])

#     if n_lines is None:
#         n_lines = len(sin_strings)
#         lines = [ line1 + line2.rjust(40, ' ') for line1, line2 in zip(sin_strings, cos_strings)]
#         print('\n'.join(lines))
#     else:
#         for i, (sstring, cstring) in enumerate(zip(sin_strings, cos_strings)):
#             if len(cstring) < 40:
#                 # cstring = ('.' * (40 - len(cstring))) + cstring
#                 cstring = '                   ' + cstring
#             edit_previous_line(cstring, n_lines - i)
#     time.sleep(.1)

# class Plotter:
#     def __init__(self, n_iters):
#         self.n_iters = n_iters
#         self.n_lines = None
#     def __iter__(self):
#         self.__current_index = 0
#         return self

#     def __next__(self):
#         if self.__current_index < self.n_iters:
#             fig = pt.Figure()
#             fig.height = 10
#             fig.width = 20
#             x = np.linspace(0, 10, 100)

#             fig.scatter(x, np.sin(x + self.__current_index), lc='red')
#             strings = fig.show().split('\n')
#             if self.n_lines is None:
#                 self.n_lines = len(strings)
#                 print(fig.show())
#             else:
#                 for i, string in enumerate(strings):
#                     edit_previous_line(string, (self.n_lines - i))
#             self.__current_index += 1
#             return self.__current_index - 1
#         else:
#             raise StopIteration

from tqdm import tqdm

# n_lines = None
# x = np.linspace(0, 10, 100)
# for i in tqdm(range(100)):

#     with nostdout():

#             fig = pt.Figure()
#             fig.height = 10
#             fig.width = 20

#             fig.scatter(x, np.sin(x + i), lc='red')

#             strings = fig.show().split('\n')
#             if n_lines is None:
#                 n_lines = len(strings)
#                 print(fig.show())
#             else:
#                 for i, string in enumerate(strings):
#                     edit_previous_line(string, n_lines - i)
#             time.sleep(.1)

# last_element = lines.pop()
        # lines.insert(0, last_element)
        # lines_1.append(lines_1[-1])

def insert_elem(lst: list, from_index: int, to_index: int) -> None:
    lst.insert(to_index, lst.pop(from_index))

def swith_elems(lst: list, idx1: int, idx2: int) -> None:
    lst[idx1], lst[idx2] = lst[idx2], lst[idx1]

def str_tick(min_, max_, n=0, transform=None):
    if transform:
        min_ = transform(min_)
    formatter = '{' + f':.{n}f' + '}'
    return formatter.format(min_)

def scatter(*args, width=80, height=40, X_label='X', Y_label='Y', linesep=os.linesep,
         x_min=None, x_max=None, y_min=None, y_max=None,
         color=None, bg=None, color_mode='names', origin=True,
         markers=None, xtick_fmt=str_tick, ytick_fmt=partial(str_tick, n=2),
         curve_labels=None,
         current_value_index=None,
        ):

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
    @staticmethod
    def __validate_list(value):
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
        # print(values)
        # print([min(data) for data in values], min([min(data) for data in values]))
        return min([min(data) for data in values])

    @staticmethod
    def __max_value(values: list[list[float]]) -> float:
        return max([max(data) for data in values])

    @staticmethod
    def __zip_data(data1: list, data2: list) -> list:
        out = list()

        for sample1, sample2 in zip(data1, data2):
            out.append(sample1)
            out.append(sample2)

        return out

    @staticmethod
    def __update_history(history: list[list[float]], values: list[float]) -> None:

        if history:
            for i, v in enumerate(values):
                history[i].append(v)
        else:
            for v in values:
                history.append([v])

    @staticmethod
    def __fill_seq(sequence1: Sequence, sequence2: Sequence, value: float = 0):
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
        loss_names = ['train', 'val']
    ):
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

    def __call__(self, loss: float | list[float], metric: float | list[float]) -> list[str]:
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
            y_min=0, y_max=102,
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



from deepmeg.training.callbacks import Callback, PrintingCallback
from typing import Callable

class VisualizingCallback(PrintingCallback):
    def __init__(
        self,
        n_epochs: int,
        width: int = 40,
        height: int = 5,
        loss_colors: str | list[str] = None,
        metric_colors: str | list[str] = None,
        loss_label = 'Loss',
        metric_label = 'Metric',
        metric_names = ['train_acc', 'val_acc'],
        loss_names = ['train_loss', 'val_loss'],
        sep = '   |    ',
        format_fn: Callable[[int, dict[str, float]], str] = None,
        print_history: bool = True
    ):
        self.print_history = print_history
        self.n_lines = None
        self.plotter = MetricsConsolePlotter(
            n_epochs,
            width,
            height,
            loss_colors,
            metric_colors,
            loss_label,
            metric_label,
            metric_names,
            loss_names
        )
        super().__init__(sep, format_fn)

    def on_epoch_end(self, epoch_num, metrics):
        if self.print_history:
            if self.format_fn:
                text = self.format_fn(epoch_num, metrics, self.sep)
            else:
                text = self.default_format_fn(epoch_num, metrics, self.sep)

        l, a = list(), list()
        for metric in metrics:
            if metric in self.metric_names:
                a.append(metrics[metric])
            elif metric in self.loss_names:
                l.append(metrics[metric])

        plot_data = self.plotter(l, a)

        if not n_lines:
            n_lines = len(plot_data)
            print((os.linesep).join(plot_data))
        else:
            for i, line in enumerate(plot_data):
                edit_previous_line(line, n_lines - i)

        if self.print_history:
            add_line_above(text, n_lines)



if __name__ == '__main__':
    plotter = MetricsConsolePlotter(
        150,
        metric_colors = ['blue', 'yellow', 'green'],
        loss_colors = ['red', 'magenta', 'cyan', 'white'],
        metric_label = 'Accuracy',
        loss_label = 'Mse',
        height=10,
        metric_names = ['train_acc', 'val_acc', 'test_acc'],#, 'dev'],#, 'dev2'],
        loss_names=['train_mse', 'val_mse', 'test_mse', 'dev_mse']
    )
    t = np.arange(150)
    loss = [
        10 - (np.sqrt(t) + np.random.random(150)),
        8 - (np.sqrt(t)/2 + np.random.random(150)),
        10 - (np.sqrt(t)/2.2 + np.random.random(150)),
        5 - (np.sqrt(t)/2.2 + np.random.random(150)),
    ]
    # loss = [10 - (t + np.random.random(150)), 10 - (t/2 + np.random.random(150))]
    acc = [10*np.sqrt(t) + np.random.random(150), 10*np.sqrt(t)/2, 10*np.sqrt(t)/4, 10*np.sqrt(t)/6, 5*np.sqrt(t)/2]
    for a in acc:
        a[a>100] = 100
    # acc = np.sqrt(t)

    n_lines = None

    for n in tqdm(range(150), initial=0, total=150, file=sys.stdout):
        with nostdout():
            l = loss[0][n], loss[1][n], loss[2][n], loss[3][n]
            a = acc[0][n], acc[1][n], acc[2][n]#, acc[3][n]#, acc[4][n]

            plot_data = plotter(l, a)

            if not n_lines:
                n_lines = len(plot_data)
                print((os.linesep).join(plot_data))
            else:
                for i, line in enumerate(plot_data):
                    edit_previous_line(line, n_lines - i)
            epoch_num = n
            sep = '   |    '
            metrics = {'mse_train': l[0], 'acc_train': a[0], 'mse_val': l[1], 'acc_val': a[1]}
            add_line_above(
                f'Epoch {epoch_num}:'.ljust(10, ' ') + sep.join(list(map(
                    lambda x: f'{x[0]}: {x[1] : .4f}',
                    metrics.items()
                ))),
                n_lines
            )
            time.sleep(.5)


# def main():
#     fig = pt.Figure()
#     fig.width = 50
#     fig.height = 20

#     x = np.linspace(0, 2 * np.pi, 20)
#     y = np.sin(x)
#     fig.plot(x, y, lc='red')

#     xs = [x[5]]
#     ys = [y[5]]

#     fig.text(xs, ys, ['x {:.3f}'.format(val) for val in ys], lc='green')

#     print(fig.show(legend=True))


# if __name__ == '__main__':
#     main()