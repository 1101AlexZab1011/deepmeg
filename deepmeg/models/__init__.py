import os
from typing import Callable, Sequence
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from ..data.utils import make_train_and_val_loaders
from ..training.callbacks import Callback, PrintingCallback
from ..training.trainers import Trainer
import numpy as np

class BaseModel(torch.nn.Module):
    """
    A base class for creating models with PyTorch.
    This class provides common functionalities such as training, evaluation, saving and loading.

    Attributes:
        trainer (Trainer): The trainer object used to train the model.
    """

    def __init__(self):
        """
        Initializes the BaseModel.
        """
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model. This function should be implemented by subclasses.

        Args:
            x (torch.Tensor): Input to the model.

        Returns:
            torch.Tensor: Output of the model.
        """
        ...

    @staticmethod
    def _prepare_dataloaders(
        data: Dataset | DataLoader,
        batch_size: int,
        val_data: Dataset | DataLoader,
        val_batch_size: int
    ):
        """
        Prepare train and validation data loaders

        Args:
            data (Dataset | DataLoader): Input data. Can be either a PyTorch Dataset or a DataLoader object
            batch_size (int): batch size for the training data
            val_data (Dataset | DataLoader): Input validation data. Can be either a PyTorch Dataset or a DataLoader object
            val_batch_size (int): batch size for the validation data

        Returns:
            Tuple[DataLoader, DataLoader]: Tuple containing train and validation DataLoader objects

        Raises:
            ValueError: If data is DataLoader object and batch_size is given
            ValueError: If val_data is DataLoader object and val_batch_size is given
            TypeError: If input data is not a Dataset or DataLoader
            TypeError: If input val_data is not a Dataset or DataLoader

        """
        val_loader = None
        if isinstance(data, (Dataset, torch.utils.data.dataset.Subset)):
            if batch_size is None:
                batch_size = len(data)//10

            if val_data is not None or val_batch_size is None:
                train_loader = DataLoader(data, batch_size, shuffle=True)
            else:
                train_loader, val_loader = make_train_and_val_loaders(data, batch_size, val_batch_size)
        elif isinstance(data, DataLoader):
            if batch_size is not None:
                raise ValueError('If you give a Dataloader, you can not set batch_size')

            if (val_data is None and val_batch_size is not None):
                raise ValueError('Can not split data into train and test sets from dataloader. Give dataset instead or do not use validation')
            else:
                train_loader = data
        else:
            raise TypeError(f'Wrong type for input data: {type(data)}. Input data must be either Dataset or DataLoader')

        if val_loader is None:
            if isinstance(val_data, (Dataset, torch.utils.data.dataset.Subset)):
                if val_batch_size is None:
                    val_batch_size = len(val_data)//10
                val_loader = DataLoader(val_data, val_batch_size, shuffle=True)
            elif isinstance(val_data, DataLoader):
                if val_batch_size is not None:
                    raise ValueError('If you give a Dataloader, you can not set batch_size')
                val_loader = val_data
            elif val_data is None and val_batch_size is None :
                val_loader = None
            else:
                raise TypeError(f'Wrong type for validation data: {type(data)}. Validation data must be either Dataset or DataLoader')

        return train_loader, val_loader

    @staticmethod
    def _validate_metrics(
        metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] |\
            list[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] |\
            tuple[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] |\
            list[tuple[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]] = None
    ):
        """
        Validate the given metrics and return a list of tuples of name and function

        Parameters:
            metrics: A single function, a list of functions, a tuple of name and function, a list of tuples of name and function.
                The function should take in two arguments, `output` and `target` and return a single value.
                If None, returns None.

        Returns:
            A list of tuples of name and function
        """
        if isinstance(metrics, Callable):
            return [(metrics.__name__, metrics)]
        elif isinstance(metrics, Sequence):
            if isinstance(metrics[0], str) and isinstance(metrics[1], Callable):
                return [tuple(metrics)]
            else:
                all_metrics = list()
                for metric in metrics:
                    if isinstance(metric, Callable):
                        all_metrics.append((metric.__name__, metric))
                    elif isinstance(metric[0], str) and isinstance(metric[1], Callable):
                        all_metrics.append(tuple(metric))
                return all_metrics
        else:
            return None

    def compile(
        self,
        optimizer: torch.optim.Optimizer = None,
        loss: nn.Module = None,
        metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] |\
            list[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] |\
            tuple[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] |\
            list[tuple[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]] = None,
        callbacks: Callback | list[Callback] = None,
        trainer: Trainer = None,
        device: torch.device = None,
    ):
        """Compile the model with optimizer, loss, metrics, callbacks and trainer.

        Args:
            optimizer (torch.optim.Optimizer, optional): The optimizer used to train the model. Defaults to None. If None, the ADAM optimizer is used.
            loss (nn.Module, optional): Loss function used to compute the loss. Defaults to None. If None, the L1 loss function is used.
            metrics (Callable[[torch.Tensor, torch.Tensor], torch.Tensor] |
                list[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] |
                tuple[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] |
                list[tuple[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]], optional): Metrics used to evaluate the model's performance.
                Defaults to None.
            callbacks (Callback | list[Callback], optional): Callbacks to be called during training. Defaults to None.
                If None, only callback for printing training progress will be used
            trainer (Trainer, optional): The trainer class used to train the model. Defaults to None. If None, Trainer is used.
            device (torch.device, optional): Device on which to train the model. Defaults to None. If None and cuda is available, cuda is used. Otherwise, cpu is used.

        """
        if optimizer is None:
            optimizer =  torch.optim.Adam(self.parameters())
        elif isinstance(optimizer, Callable):
            optimizer = optimizer(self.parameters())

        if loss is None:
            loss = torch.nn.L1Loss()

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        metrics = self._validate_metrics(metrics)

        if callbacks is None:
            callbacks = [
                PrintingCallback()
            ]

        if trainer is None:
            trainer = Trainer

        self.trainer = trainer(
            self, optimizer, loss, device, metrics,
            callbacks=callbacks
        )

    def fit(
        self,
        data: Dataset | DataLoader,
        n_epochs: int = 25,
        batch_size: int = None,
        val_data: Dataset | DataLoader = None,
        val_batch_size: int = None,
        update_every_n_batches: int = 1,
        eval_on_n_batches: int = 1,
        eval_every_n_epochs: int = 1
    ) -> dict[str, np.ndarray]:
        """Trains the model on the given data.

        Args:
            data (Dataset | DataLoader): The dataset or data loader to be used for training.
            n_epochs (int): The number of epochs to train the model for.
            batch_size (int): The number of samples per batch. If None, it defaults to len(data)//10.
            val_data (Dataset | DataLoader): The validation dataset or data loader. If None, no validation is performed.
            val_batch_size (int): The number of samples per validation batch. If None, it defaults to len(val_data)//10.
            update_every_n_batches (int): Numberof batches after which the learning rate will be updated.

        Returns:
        dict[str, np.ndarray]: A dictionary containing the history of metric values during training

        Example:
        >>> model = MyModel()
        >>> model.fit(X_train, y_train, epochs=10, batch_size=32)
        ... update_every_n_batches=10, verbose=True)
        Epoch 1/10: 100%|██████████| 312/312 [00:01<00:00, 195.89it/s, loss=0.234]
        Epoch 2/10: 100%|██████████| 312/312 [00:01<00:00, 195.89it/s, loss=0.123]
        ...
        >>> model(X_test)
        array([0, 1, 1, ..., 1, 0, 1])
        """
        train_loader, val_loader = self._prepare_dataloaders(data, batch_size, val_data, val_batch_size)
        return self.trainer.fit(train_loader, n_epochs, val_loader, update_every_n_batches, eval_on_n_batches, eval_every_n_epochs)

    def evaluate(
        self,
        data: Dataset | DataLoader,
        batch_size: int = None,
        eval_on_n_batches: int = 1
    ):
        """
        Evaluate the model on the given dataset or dataloader.

        Args:
            data (Dataset | DataLoader): Input data. Can be either a PyTorch Dataset or a DataLoader object
            batch_size (int): batch size for the evaluation data. If not provided, the batch size is set to the length of the data divided by 10.
            eval_on_n_batches (int): number of batches to evaluate on. If set to None, evaluate on the entire dataset.

        Returns:
            dict: A dictionary containing the evaluation metrics and their values.
        """
        val_loader, _ = self._prepare_dataloaders(data, batch_size, None, None)
        return self.trainer.evaluate(val_loader, eval_on_n_batches)

    def save(self, path: str | os.PathLike):
        """
        save method

        Args:
        path (str | os.PathLike): path to the location where the model's parameters should be saved

        save the model's state_dict to the specified path
        """
        torch.save(self.state_dict(), path)

    def load(self, path: str | os.PathLike):
        """
        load method

        Args:
        path (str | os.PathLike): path to the location where the model's parameters are saved

        Load the model's parameters from the specified path
        """
        self.load_state_dict(torch.load(path))
