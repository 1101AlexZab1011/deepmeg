from collections import defaultdict
import sys
from typing import Any, Callable, Iterator, Optional, Sequence
import numpy as np
import torch
import torch.nn as nn
from .callbacks import Callback
from torch.utils.data import DataLoader
from tqdm import tqdm
from ..utils.printout import nostdout


class Trainer:
    """
    A class for training deep learning models with PyTorch.
    """

    def __init__(
        self, model: nn.Module, optimizer: torch.optim.Optimizer,
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        device: str, metric_functions: list[tuple[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]] = None,
        epoch_number: int = 0,
        lr_scheduler: Optional[Any] = None,
        callbacks: Callback | list[Callback] = None
    ):
        """
        Initialize the trainer.

        Args:
            model (nn.Module): the PyTorch model to be trained.
            optimizer (torch.optim.Optimizer): the optimizer to be used for training.
            criterion (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): the loss function to be used for training.
            device (str): the device on which the training should be done (e.g. 'cpu' or 'cuda').
            metric_functions (List[Tuple[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]]]): list of tuples containing
                the name of the metric and a callable that calculates that metric.
            epoch_number (int): the number of epochs for which the model should be trained.
            lr_scheduler (Optional[Any]): the learning rate scheduler to be used.
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.model.to(self.device)

        if metric_functions and not isinstance(metric_functions, Sequence):
            metric_functions = [metric_functions]

        self.metric_functions = metric_functions if metric_functions else list()
        self.epoch_number = epoch_number
        self._interrupt = False

        if callbacks:
            if isinstance(callbacks, Callback):
                callbacks.set_trainer(self)
                self.callbacks = [callbacks]
            elif isinstance(callbacks, Sequence):
                for callback in callbacks:
                    callback.set_trainer(self)

                self.callbacks = callbacks
            else:
                raise ValueError(f'Callbacks must be either Sequence or Callback instance, instead {type(callbacks)} is given')
        else:
            self.callbacks = callbacks

    def interrupt(self, interrupt: bool = True):
        """Immediately stop training/evaluation"""
        self._interrupt = interrupt

    @torch.no_grad()
    def evaluate_batch(self, val_iterator: Iterator, eval_on_n_batches: int) -> Optional[dict[str, float]]:
        """
        Evaluates the model on a single batch of validation data.

        Args:
            val_iterator (Iterator): an iterator over the validation data.
            eval_on_n_batches (int): the number of batches to evaluate the model on.

        Returns:
            Optional[Dict[str, float]]: a dictionary containing the evaluation metrics.
        """
        predictions = []
        targets = []

        losses = []

        for real_batch_number in range(eval_on_n_batches):

            try:
                xs, ys_true = next(val_iterator)

                xs = xs.to(self.device)
                ys_true = ys_true.to(self.device)

                if self.callbacks:
                    for callback in self.callbacks:
                        callback.on_batch_begin(xs, ys_true)
                        callback.on_eval_batch_begin(xs, ys_true)
                if self._interrupt:
                    break
            except StopIteration:
                if real_batch_number == 0:
                    return None
                else:
                    break

            ys_pred = self.model.eval()(xs)
            loss = self.criterion(ys_pred, ys_true)

            if self.callbacks:
                for callback in self.callbacks:
                    new_loss = callback.on_loss_computed(xs, ys_true, ys_pred, loss)

                    if new_loss is not None:
                        loss = new_loss

            losses.append(loss.item())

            predictions.append(ys_pred.cpu())
            targets.append(ys_true.cpu())

        predictions = torch.cat(predictions, dim=0)
        targets = torch.cat(targets, dim=0)

        metrics = {'loss': np.mean(losses)}

        for metric_name, metric_fn in self.metric_functions:
            metrics[metric_name] = metric_fn(predictions, targets).item()

        if self.callbacks:
            for callback in self.callbacks:
                callback.on_batch_end(targets, predictions, metrics)
                callback.on_eval_batch_end(targets, predictions, metrics)

        return metrics

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader, eval_on_n_batches: int = 1) -> dict[str, float]:
        """
        Evaluates the model on the validation data.

        Args:
            val_loader (DataLoader): a PyTorch DataLoader containing the validation data.
            eval_on_n_batches (int): the number of batches through which to evaluate the model.

        Returns:
            Dict[str, float]: a dictionary containing the evaluation metrics.
        """
        self.interrupt(False)
        metrics_sum = defaultdict(float)
        num_batches = 0

        val_iterator = iter(val_loader)

        if self.callbacks:
            for callback in self.callbacks:
                callback.on_evaluate_begin()

        while True:
            batch_metrics = self.evaluate_batch(val_iterator, eval_on_n_batches)

            if batch_metrics is None or self._interrupt:
                break

            for metric_name in batch_metrics:
                metrics_sum[metric_name] += batch_metrics[metric_name]

            num_batches += 1

        metrics = {}

        for metric_name in metrics_sum:
            metrics[metric_name] = metrics_sum[metric_name] / num_batches

        if self.callbacks:
            for callback in self.callbacks:
                callback.on_evaluate_end(metrics)

        return metrics

    def fit_batch(self, train_iterator: Iterator, update_every_n_batches: int) -> Optional[dict[str, float]]:
        """
        Trains the model on a single batch of training data.

        Args:
            train_iterator (Iterator): an iterator over the training data.
            update_every_n_batches (int): the number of batches to train the model on before performing weight update.

        Returns:
            Optional[Dict[str, float]]: a dictionary containing the training metrics.
        """
        self.optimizer.zero_grad()

        predictions = []
        targets = []

        losses = []

        for real_batch_number in range(update_every_n_batches):

            try:
                xs, ys_true = next(train_iterator)

                xs = xs.to(self.device)
                ys_true = ys_true.to(self.device)

                if self.callbacks:
                    for callback in self.callbacks:
                        callback.on_batch_begin(xs, ys_true)
                        callback.on_train_batch_begin(xs, ys_true)

                if self._interrupt:
                    raise StopIteration
            except StopIteration:
                if real_batch_number == 0:
                    return None
                else:
                    break

            ys_pred = self.model.train()(xs)
            loss = self.criterion(ys_pred, ys_true)

            if self.callbacks:
                for callback in self.callbacks:
                    new_loss = callback.on_loss_computed(xs, ys_true, ys_pred, loss)

                    if new_loss is not None:
                        loss = new_loss

            (loss / update_every_n_batches).backward()
            losses.append(loss.item())

            predictions.append(ys_pred.cpu())
            targets.append(ys_true.cpu())

        self.optimizer.step()

        predictions = torch.cat(predictions, dim=0)
        targets = torch.cat(targets, dim=0)

        metrics = {'loss': np.mean(losses)}

        for metric_name, metric_fn in self.metric_functions:
            metrics[metric_name] = metric_fn(predictions, targets).item()

        if self.callbacks:
            for callback in self.callbacks:
                callback.on_batch_end(targets, predictions, metrics)
                callback.on_train_batch_end(targets, predictions, metrics)

        return metrics

    def fit_epoch(self, train_loader: DataLoader, update_every_n_batches: int = 1) -> dict[str, float]:
        """
        Trains the model on a single epoch of training data.

        Args:
            train_loader (DataLoader): a PyTorch DataLoader containing the training data.
            update_every_n_batches (int): the number of batches to train the model on before performing weight update.
            log_every_n_batches (int): the number of batches to log training metrics.

        Returns:
            Dict[str, float]: a dictionary containing the average training metrics for the epoch.
        """

        metrics_sum = defaultdict(float)
        num_batches = 0

        train_iterator = iter(train_loader)

        while True:
            batch_metrics = self.fit_batch(train_iterator, update_every_n_batches)

            if batch_metrics is None or self._interrupt:
                break

            for metric_name in batch_metrics:
                metrics_sum[metric_name] += batch_metrics[metric_name]

            num_batches += 1

        metrics = {}

        for metric_name in metrics_sum:
            metrics[metric_name] = metrics_sum[metric_name] / num_batches

        return metrics

    def fit(self, train_loader: DataLoader, num_epochs: int,
            val_loader: DataLoader = None, update_every_n_batches: int = 1,
            eval_on_n_batches: int = 1, eval_every_n_epochs: int = 1,
            ) -> dict[str, np.ndarray]:
        """
        Trains the model for a specified number of epochs.

        Args:
            train_loader (DataLoader): a PyTorch DataLoader containing the training data.
            val_loader (DataLoader): a PyTorch DataLoader containing the validation data.
            num_epochs (int): the number of training epochs.
            update_every_n_batches (int) : the number of batches to train the model on before performing weight update.
            log_every_n_batches (int): the number of batches to log training metrics.
            eval_every_n_epochs (int): the number of training epochs between evaluations.
            eval_on_n_batches (int): the number of batches to evaluate the model on.

        Returns:
            Dict[str, List[Dict[str, float]]]: a dictionary containing the training and validation metrics for each epoch.
        """
        self.interrupt(False)

        summary = defaultdict(list)

        def save_metrics(metrics: dict[str, float], postfix: str = '') -> None:
            nonlocal summary, self

            for metric in metrics:
                metric_name, metric_value = f'{metric}{postfix}', metrics[metric]

                summary[metric_name].append(metric_value)
        import traceback
        try:
            for i in tqdm(range(num_epochs - self.epoch_number), initial=self.epoch_number, total=num_epochs, file=sys.stdout):
                with nostdout():

                    self.epoch_number += 1

                    if self._interrupt:
                        print(f'The training loop was completed at epoch {self.epoch_number} due to an interruption')
                        self.interrupt(False)
                        break

                    if self.callbacks:
                        for callback in self.callbacks:
                            callback.on_epoch_begin(i)

                    train_metrics = self.fit_epoch(train_loader, update_every_n_batches)

                    train_metrics = {f'{key}_train': value for key, value in train_metrics.items()}
                    val_metrics = False

                    with torch.no_grad():
                        save_metrics(train_metrics)

                        if val_loader is not None and not i%eval_every_n_epochs:
                            val_metrics = self.evaluate(val_loader, eval_on_n_batches)
                            val_metrics = {f'{key}_val': value for key, value in val_metrics.items()}
                            save_metrics(val_metrics)

                    if self.callbacks:
                        for callback in self.callbacks:
                            all_metrics = train_metrics | val_metrics if val_metrics else train_metrics
                            callback.on_epoch_end(i, all_metrics)

                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()

            summary = {metric: np.array(summary[metric]) for metric in summary}

            return summary
        except Exception as e:
            print(e)
            traceback.print_exc()
            self.interrupt()
