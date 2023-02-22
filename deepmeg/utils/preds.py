import pandas as pd
from alltools.console.colored import warn
from alltools.machine_learning import one_hot_decoder, one_hot_decoder
from alltools.machine_learning.confusion import ConfusionEstimator
from typing import Union, Optional
import numpy as np
import sklearn.metrics  as sm


class PredictionsParser(object):
    def __init__(
        self,
        y_true: Union[list[int], np.ndarray],
        y_pred: Union[list[int], np.ndarray],
        class_names: Optional[Union[str, list[str]]] = None
    ):

        y_true = self.__check_numpy(y_true)
        y_pred = self.__check_numpy(y_pred)

        if y_true.shape != y_pred.shape:
            raise ValueError('Predictions and actual values are inconsistent. Actual values shape: {y_true.shape}, predictions shape: {y_pred.shape}')

        if len(y_true.shape) != 1:
            y_true = one_hot_decoder(y_true)
            y_pred = one_hot_decoder(y_pred)

        self._y_true = y_true
        self._y_pred = y_pred

        self._accuracy = sm.accuracy_score(y_true, y_pred)

        classes_true = np.unique(self._y_true)
        classes_pred = np.unique(self._y_pred)

        if np.any(classes_true != classes_pred):
            warn(f'Warning: Classes are inconsistent.\n\tActual classes: {classes_true}\n\tPredicted classes: {classes_pred}.\nTake actual classes')

        del classes_pred

        self._classes = classes_true
        self._n_classes = len(classes_true)

        if class_names is not None:

            if isinstance(class_names, str):
                class_names = class_names.split(' ')

            if len(class_names) != self.n_classes:
                raise ValueError(f'Class names and classes are inconsistent: number of classes is {self.n_classes}, but {len(class_names)} names of classes were given')
        else:
            class_names = [f'Class {i}' for i in range(self.n_classes)]

        self._class_names = class_names

        self._confusion = pd.DataFrame(
            sm.confusion_matrix(self.y_true, self.y_pred),
            index = [f'Actual {class_name}' for class_name in self.class_names],
            columns = [f'Predicted {class_name}' for class_name in self.class_names]
        )

    @staticmethod
    def __check_numpy(arr: Union[list, tuple, np.ndarray]):
        if isinstance(arr, np.ndarray):
            return arr
        elif not isinstance(arr, np.ndarray) and isinstance(arr, (list, tuple)):
            return np.array(arr)
        else:
            raise ValueError(f'The given argument must be either a np.ndarray or a list, but {type(arr)} was given')

    @property
    def y_true(self):
        return self._y_true
    @y_true.setter
    def y_true(self, value):
        raise AttributeError('Impossible to set y_true directly')

    @property
    def y_pred(self):
        return self._y_pred
    @y_pred.setter
    def y_pred(self, value):
        raise AttributeError('Impossible to set y_pred directly')

    @property
    def accuracy(self):
        return self._accuracy
    @accuracy.setter
    def accuracy(self, value):
        raise AttributeError('Impossible to set accuracy directly')

    @property
    def classes(self):
        return self._classes
    @classes.setter
    def classes(self, value):
        raise AttributeError('Impossible to set classes directly')

    @property
    def n_classes(self):
        return self._n_classes
    @n_classes.setter
    def n_classes(self, value):
        raise AttributeError('Impossible to set number of classes directly')

    @property
    def class_names(self):
        return self._class_names
    @class_names.setter
    def class_names(self, value):
        raise AttributeError('Impossible to set names for classes directly')

    @property
    def confusion(self):
        return self._confusion
    @confusion.setter
    def confusion(self, value):
        raise AttributeError('Impossible to set confusion matrix directly')

    def summary(self, *args: str):
        df = self.confusion.copy()
        summary = pd.DataFrame(columns = self.class_names)
        summary.loc['Total'] = [df[column].sum() for column in df.columns]
        summary.loc['Accuracy'] = [None for _ in range(self.n_classes)]
        summary.loc['Specificity'] = [None for _ in range(self.n_classes)]
        summary.loc['Sensitivity'] = [None for _ in range(self.n_classes)]

        ec = self.estimate_confusion()

        args = list(args)
        for i, arg in enumerate(args):
            if not isinstance(arg, tuple):
                args[i] = arg, arg

        for arg_value, arg_name in args:
            if hasattr(ec[self.class_names[0]], arg_value):
                summary.loc[arg_name] = [None for _ in range(self.n_classes)]
            else:
                warn(f'WARNING: the {arg_value} property was not found in the confusion evaluator, so it was ignored')
                args.remove((arg_value, arg_name))


        for i, class_name in enumerate(self.class_names):
            summary[class_name].loc['Accuracy'] = ec[class_name].acc
            summary[class_name].loc['Specificity'] = ec[class_name].spec
            summary[class_name].loc['Sensitivity'] = ec[class_name].sens

            for arg_value, arg_name in args:
                summary[class_name].loc[arg_name] = getattr(ec[class_name], arg_value)

        return summary

    def estimate_confusion(self):

        return {
            class_name: ConfusionEstimator(
                # tp, tn, fp, fn
                self.confusion[self.confusion.columns[i]][self.confusion.index[i]],
                self.confusion[
                    list(self.confusion.columns[:i]) + list(self.confusion.columns[i+1:])
                    ].loc[
                        list(self.confusion.index[:i]) + list(self.confusion.index[i+1:])
                ].sum().sum(),
                self.confusion[self.confusion.columns[i]].loc[
                    list(self.confusion.index[:i]) + list(self.confusion.index[i+1:])
                ].sum(),
                self.confusion[
                    list(self.confusion.columns[:i]) + list(self.confusion.columns[i+1:])
                ].loc[self.confusion.index[i]].sum()
            )
            for i, class_name in enumerate(self.class_names)
        }

