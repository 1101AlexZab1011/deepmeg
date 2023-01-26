from typing import Callable, TypeVar
import torch
import numpy as np
from copy import deepcopy


Trainer = TypeVar('Trainer')

class Callback:
    """
    A base class for callbacks. The methods of this class are called by a `Trainer` during the training process.
    """
    def __init__(
        self
    ):
        """
        Initializes the callback.
        """
        self.trainer = None

    def set_trainer(self, trainer: 'Trainer'):
        """
        Sets the trainer object that will be calling the callback's methods.

        Args:
            trainer (Trainer): The trainer object.
        """
        self.trainer = trainer

    def on_batch_begin(
        self,
        X: torch.Tensor,
        Y: torch.Tensor
    ):
        """
        Called at the beginning of a batch of training/evaluating data.

        Args:
            X (torch.Tensor): Input data of the batch.
            Y (torch.Tensor): Target data of the batch.
        """
        ...

    def on_batch_end(
        self,
        Y: torch.Tensor,
        Y_pred: torch.Tensor,
        metrics: dict
    ):
        """
        Called at the end of a batch of training/evaluating data.

        Args:
            Y (torch.Tensor): Target data of the batch.
            Y_pred (torch.Tensor): Predicted data of the batch.
            metrics (dict): A dictionary containing the batch's metrics.
        """
        ...

    def on_epoch_begin(
        self,
        epoch_number: int
    ):
        """
        Called at the beginning of an epoch of training.

        Args:
            epoch_number (int): The number of the current epoch.
        """
        ...

    def on_epoch_end(
        self,
        epoch_number: int,
        metrics: dict
    ):
        """
        Called at the end of an epoch of training.

        Args:
            epoch_number (int): The number of the current epoch.
            metrics (dict): A dictionary containing the epoch's metrics.
        """
        ...

    def on_train_batch_begin(
        self,
        X: torch.Tensor,
        Y: torch.Tensor
    ):
        """
        Called at the beginning of a batch of training data, during the training phase.

        Args:
            X (torch.Tensor): Input data of the batch.
            Y (torch.Tensor): Target data of the batch.
        """
        ...

    def on_train_batch_end(
        self,
        Y: torch.Tensor,
        Y_pred: torch.Tensor,
        metrics: dict()
    ):
        """
        Called at the end of a batch of training data, during the training phase.

        Args:
            Y (torch.Tensor): Target data of the batch.
            Y_pred (torch.Tensor): Predicted data of the batch.
            metrics (dict): A dictionary containing the batch's metrics.
        """
        ...

    def on_evaluate_begin(
        self,
    ):
        """
        Called at the beginning of the evaluation phase.
        """
        ...

    def on_evaluate_end(
        self,
        metrics: dict
    ):
        """
        Called at the end of the evaluation phase.

        Args:
            metrics (dict): A dictionary containing the evaluation metrics.
        """
        ...

    def on_eval_batch_begin(
        self,
        X: torch.Tensor,
        Y: torch.Tensor
    ):
        """
        This method is called at the beginning of each evaluation batch.

        Args:
            X (torch.Tensor): The input data of the current batch.
            Y (torch.Tensor): The target data of the current batch.
        """
        ...

    def on_eval_batch_end(
        self,
        Y: torch.Tensor,
        Y_pred: torch.Tensor,
        metrics: dict
    ):
        """
        This method is called at the end of each evaluation batch.

        Args:
            Y (torch.Tensor): The true labels of the batch.
            Y_pred (torch.Tensor): The predicted labels of the batch.
            metrics (dict): A dictionary containing the current evaluation metrics of the model.
        """
        ...

    def on_loss_computed(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        Y_pred: torch.Tensor,
        loss: torch.Tensor
    ):
        """Called after the loss for a batch of data has been computed.

        This function is called after the loss for a batch of data has been computed, but before the gradients are computed and applied.
        This is useful for any computations that need to be done before the gradients are computed and applied.

        Args:
            X (torch.Tensor): A batch of input data.
            Y (torch.Tensor): A batch of target data.
            Y_pred (torch.Tensor): A batch of predicted data.
            loss (torch.Tensor): The computed loss for this batch of data.
        """
        ...


class MeasureCallback(Callback):
    """
    A callback that allows to add new measure to the history.
    """
    def __init__(self, criterion: Callable[[torch.Tensor, torch.Tensor], float], criterion_name: str):
        """
        Initialize the MeasureCallback class.

        Args:
            criterion (Callable): The criterion function to compute loss between predicted and actual value.
            criterion_name (str): The name of the criterion function.
        """
        super().__init__()
        self.criterion = criterion
        self.criterion_name = criterion_name

    def on_batch_end(
        self,
        Y: torch.Tensor,
        Y_pred: torch.Tensor,
        metrics: dict
    ):
        """
        Add the loss value to the metrics dictionary after each batch end.

        Args:
            Y (torch.Tensor): The actual value of the target.
            Y_pred (torch.Tensor): The predicted value of the target.
            metrics (dict): A dictionary containing the metrics of the training and evaluation process.
        """
        metrics[self.criterion_name] = self.criterion(Y, Y_pred).item()


class PrintingCallback(Callback):
    """
    A callback that prints the values of metrics after each epoch.
    """
    def __init__(self, sep = '   |    ', format_fn: Callable[[int, dict[str, float]], str] = None):
        """
        Initializes the PrintingCallback.

        Parameters:
            sep (str): separator used to separate the metric names from the metric values in the printout.
            format_fn (Callable[[int, dict[str, float]], str]): a function that takes in the current epoch number
                and a dictionary of metrics and returns a string that will be printed out.
                If not provided, a default function will be used.
        """
        super().__init__()
        self.sep = sep
        self.format_fn = format_fn

    @staticmethod
    def default_format_fn(epoch_num, metrics: dict[str, float], sep: str = ', ') -> str:
        """
        A default function that can be used for formatting the printout if no other function is provided.

        Parameters:
            epoch_num (int): the current epoch number.
            metrics (dict[str, float]): a dictionary of metrics and their values.
            sep (str): separator used to separate the metric-value pairs from each other in the printout.

        Returns:
            str: a formatted string containing the current epoch number and the values of all metrics.
        """
        return f'Epoch {epoch_num}:'.ljust(10, ' ') + sep.join(list(map(
            lambda x: f'{x[0]}: {x[1] : .4f}',
            metrics.items()
        )))

    def on_epoch_end(self, epoch_num, metrics):
        """
        This method is called at the end of each epoch.
        It prints the metrics of the epoch, if the format_fn attribute is present, it will use it to format the output,
        otherwise it will use the default_format_fn method.
        Args:
            epoch_num(int): the number of the epoch.
            metrics(dict): the metrics collected during the epoch.
        """
        if self.format_fn:
            print(self.format_fn(epoch_num, metrics, self.sep))
        else:
            print(self.default_format_fn(epoch_num, metrics, self.sep))


class EarlyStopping(Callback):
    """
    A callback that performing early stopping.
    """
    def __init__(self, patience=5, monitor='loss_train', min_delta=0, restore_best_weights=True):
        """
        Initializes the callback.

        Args:
            patience (int): Number of consecutive epochs with no improvement after which training will be stopped.
            monitor (str): The metric to monitor. If it stops improving, training will be stopped.
            min_delta (float): Minimum change in the monitored metric to qualify as an improvement.
            restore_best_weights (bool): Whether to restore the model's best weights when training is stopped.
        """
        super().__init__()
        self.patience = patience
        self.monitor = monitor
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.min_criterion_value = np.inf
        self.best_weights = None

    def set_trainer(self, trainer: 'Trainer'):
        """
        Binds the trainer to the callback.

        Args:
            trainer (Trainer): The trainer object to bind to the callback.
        """
        super().set_trainer(trainer)
        self.model = self.trainer.model

    def on_epoch_end(self, epoch_num, metrics):
        """
        Called at the end of each epoch during training.

        Parameters:
        -----------
        epoch_num : int
            The current epoch number.
        metrics : dict
            Dictionary of metric values computed during the epoch.
        """
        criterion_value = metrics[self.monitor]
        if criterion_value < self.min_criterion_value:
            self.min_criterion_value = criterion_value
            self.counter = 0
            self.best_weights = deepcopy(self.model.state_dict())

        elif criterion_value > (self.min_criterion_value + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    self.restore()

                self.trainer.interrupt()

    def restore(self):
        """
        restore best weights
        """
        self.model.load_state_dict(self.best_weights)


class L1Reg(Callback):
    """
    A callback that performing L1 regularization.
    """
    def __init__(self, layer_names = str | list[str], lambdas: float | list[float] =  None):
        """
        Initializes L1Reg callback

        Args:
            layer_names (Union[str, List[str]]): The name of the layer(s) that need to be regularized. To see possible layer names, check model.state_dict() keys.
            lambdas (Union[float, List[float]], optional): The lambda value(s) for the regularization. Defaults to None.
                If not provided, the default value will be used for all layers.
                If a single value is provided, it will be used for all layers.
                If a list of values is provided, it must have the same length as the layer_names list.
        Raises:
            ValueError: If the length of `layer_names` and `lambdas` is not the same.
        """
        super().__init__()
        self.layer_names = layer_names if isinstance(layer_names, list) else [layer_names]

        if lambdas is None:
            lambdas = [0.01 for _ in range(len(self.layer_names))]
        elif isinstance(lambdas, float):
            lambdas = [lambdas for _ in range(len(self.layer_names))]

        if len(lambdas) != len(self.layer_names):
            raise ValueError(f'Layers to regularize are inconsistent with amount of given lambdas ({len(self.layer_names)} ~ {len(lambdas)})')

        self.lambdas = lambdas

    def set_trainer(self, trainer: 'Trainer'):
        """
        Binds the trainer to the callback

        Args:
            trainer (Trainer): The trainer object
        """
        super().set_trainer(trainer)
        self.model = self.trainer.model

    def on_loss_computed(self, X, Y, Y_pred, loss):
        """
        Computes the loss value with L1 regularization

        Args:
            X (torch.Tensor): The input data
            Y (torch.Tensor): The target data
            Y_pred (torch.Tensor): The model's prediction
            loss (float): The current loss value

        Returns:
            float: The new loss value with L1 regularization added
        """
        state_dict = self.model.state_dict()

        for layer_name, lambda_ in zip(self.layer_names, self.lambdas):
            loss += lambda_*torch.norm(state_dict[layer_name], 1)

        return loss


class L2Reg(L1Reg):
    """
    A callback that performing L2 regularization.
    """
    def on_loss_computed(self, X, Y, Y_pred, loss):
        """
        Computes the loss value with L2 regularization.
        Args:
            X (torch.Tensor): The input data
            Y (torch.Tensor): The target data
            Y_pred (torch.Tensor): The model's prediction
            loss (float): The current loss value
        Returns:
            float: The new loss value with L2 regularization added
        """
        state_dict = self.model.state_dict()

        for layer_name, lambda_ in zip(self.layer_names, self.lambdas):
            loss += lambda_*torch.norm(state_dict[layer_name], 2)

        return loss