import os
from typing import Callable, Iterable
import mne
import torch
from torch.utils.data import Dataset
import numpy as np
from ..preprocessing.transforms import one_hot_encoder
from ..utils import check_path
from copy import deepcopy


class EpochsDataset(Dataset):
    """
    A PyTorch dataset class for working with data from MNE Epochs.
    """
    def __init__(
        self,
        epochs: str | os.PathLike | tuple[np.ndarray, np.ndarray] | mne.Epochs,
        transform: Callable[[torch.Tensor], torch.Tensor] = None, target_transform: Callable[[torch.Tensor], torch.Tensor]  = None,
        savepath: str | os.PathLike = './data'
    ):
        """
        Initialize the EpochsDataset object.

        Args:
            epochs (Union[str, os.PathLike, Tuple[np.ndarray, np.ndarray], mne.Epochs]): path to an MNE Epochs file, an MNE Epochs object, a tuple of numpy arrays containing the data and targets, or an MNE EpochsArray object.
            transform (Optional[Callable[[torch.Tensor], torch.Tensor]]): a function to be applied to the data samples.
            target_transform (Optional[Callable[[torch.Tensor], torch.Tensor]]): a function to be applied to the targets.
            savepath (Union[str, os.PathLike]): the path to the directory where the data samples and targets will be saved.
        Raises:
            ValueError: if the epochs file is not in a supported format (path to mne.Epochs, mne.Epochs or sample-target couples of torch.Tensor)

        """
        if isinstance(epochs, (str, os.PathLike)):
            epochs = mne.read_epochs(epochs)

        if isinstance(epochs, (mne.Epochs, mne.epochs.EpochsArray)):
            data = epochs.get_data()
            X = [torch.Tensor(sample) for sample in data]
            Y = one_hot_encoder(epochs.events[:, 2])
            Y = [torch.Tensor(event) for event in Y]
        elif isinstance(epochs, tuple):
            X = [torch.Tensor(sample) for sample in epochs[0]]
            Y = [torch.Tensor(target) for target in epochs[1]]
        else:
            raise ValueError(f'Unsupported type for data samples: {type(epochs)}')

        self.n_samples = len(X)
        self.savepath = savepath
        self.transform = transform
        self.target_transform = target_transform

        check_path(savepath)

        for i, (sample, target) in enumerate(zip(X, Y)):
            torch.save(sample, os.path.join(self.savepath, f'sample_{i}.pt'))
            torch.save(target, os.path.join(self.savepath, f'target_{i}.pt'))

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return self.n_samples

    def __getitem__(self, idx):
        """
        Retrieves the sample and target at the given index.

        Args:
            idx (int): the index of the sample and target to be retrieved.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The sample and target at the given index.
        """
        sample_path = os.path.join(self.savepath, f'sample_{idx}.pt')
        target_path = os.path.join(self.savepath, f'target_{idx}.pt')

        X = torch.load(sample_path)
        Y = torch.load(target_path)

        if self.transform:
            X = self.transform(X)

        if self.target_transform:
            Y = self.target_transform(Y)

        return X, Y

    def save(self, path: str | os.PathLike):
        """
        Saves the EpochsDataset object to a specified path.

        Args:
            path (Union[str, os.PathLike]): the path where the object should be saved.
        Raises:
            OSError: if the path to save the dataset object leads to the same directory in which its contents are saved
        """
        if os.path.dirname(path) == self.savepath.split('/')[-1]:
            raise OSError('Dataset can not be saved in the same directory with its contents')

        torch.save(self, path)

    @staticmethod
    def load(path: str | os.PathLike) -> 'EpochsDataset':
        """Reads epochs dataset in '.pt' format"""
        return torch.load(path)


def read_epochs_dataset(path: str | os.PathLike) -> EpochsDataset:
    """Reads epochs dataset in '.pt' format"""
    return torch.load(path)


class EpochsDatasetWithMeta(EpochsDataset):
    def __init__(
        self,
        epochs: str | os.PathLike | tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, Iterable] | mne.Epochs,
        transform: Callable[[torch.Tensor], torch.Tensor] = None, target_transform: Callable[[torch.Tensor], torch.Tensor]  = None,
        savepath: str | os.PathLike = './data'
    ):
        """
        A PyTorch dataset class for EEG data with additional metadata.

        Args:
            epochs: An instance of mne.Epochs or a tuple of EEG data X and target Y with optional metadata Z, or a file path to load mne.Epochs data.
            transform: A callable function to apply a transformation on the input data.
            target_transform: A callable function to apply a transformation on the target data.
            savepath: A path to the directory to save the processed data.

        Raises:
            ValueError: If the data type for samples is not supported.

        Attributes:
            n_samples: An integer representing the total number of data samples.
            savepath: A path to the directory to save the processed data.
            transform: A callable function to apply a transformation on the input data.
            target_transform: A callable function to apply a transformation on the target data.
        """
        if isinstance(epochs, (str, os.PathLike)):
            epochs = mne.read_epochs(epochs)

        if isinstance(epochs, (mne.Epochs, mne.epochs.EpochsArray)):
            data = epochs.get_data()
            X = [torch.Tensor(sample) for sample in data]
            Y = one_hot_encoder(epochs.events[:, 2])
            Y = [torch.Tensor(event) for event in Y]
            Z = list(epochs.metadata.iterrows()) if epochs.metadata is not None else [None for _ in range(len(X))]
        elif isinstance(epochs, tuple):
            X = [torch.Tensor(sample) for sample in epochs[0]]
            Y = [torch.Tensor(target) for target in epochs[1]]

            if len(epochs) == 3:
                Z = [metadata for metadata in epochs[2]]
            else:
                Z = [None for _ in range(len(X))]
        else:
            raise ValueError(f'Unsupported type for data samples: {type(epochs)}')

        self.n_samples = len(X)
        self.savepath = savepath
        self.transform = transform
        self.target_transform = target_transform

        check_path(savepath)

        for i, (sample, target, meta) in enumerate(zip(X, Y, Z)):
            torch.save(sample, os.path.join(self.savepath, f'sample_{i}.pt'))
            torch.save(target, os.path.join(self.savepath, f'target_{i}.pt'))

            if meta is not None:
                torch.save(meta, os.path.join(self.savepath, f'meta_{i}.pt'))

    def __getitem__(self, idx):
        """
        Returns a processed data sample and its target with optional metadata from the dataset.

        Args:
            idx: An integer representing the index of the data sample.

        Returns:
            X: A PyTorch Tensor representing the processed input data sample.
            Y: A PyTorch Tensor representing the processed target data.
            Z: A PyTorch Tensor representing the metadata, or None if not available.

        """
        sample_path = os.path.join(self.savepath, f'sample_{idx}.pt')
        target_path = os.path.join(self.savepath, f'target_{idx}.pt')
        meta_path = os.path.join(self.savepath, f'meta_{idx}.pt')

        X = torch.load(sample_path)
        Y = torch.load(target_path)
        Z = torch.load(meta_path) if os.path.exists(meta_path) else None

        if self.transform:
            X = self.transform(X)

        if self.target_transform:
            Y = self.target_transform(Y)

        return X, Y, Z
