import os
from typing import Callable
import mne
import torch
from torch.utils.data import Dataset
import numpy as np
from ..preprocessing.transforms import one_hot_encoder
from ..utils import check_path


class EpochsDataset(Dataset):
    """
    A PyTorch dataset class for working with data from MNE Epochs.
    """
    def __init__(
        self,
        epochs: str | os.PathLike | tuple[np.ndarray, np.ndarray] | mne.Epochs,
        transform: Callable[[torch.Tensor], torch.Tensor] = None,target_transform: Callable[[torch.Tensor], torch.Tensor]  = None,
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
        if os.path.dirname(path) == os.path.dirname(self.savepath):
            raise OSError('Dataset can not be saved in the same directory with its contents')
        torch.save(self, path)

    @staticmethod
    def load(path: str | os.PathLike) -> 'EpochsDataset':
        """Reads epochs dataset in '.pt' format"""
        return torch.load(path)


def read_epochs_dataset(path: str | os.PathLike) -> EpochsDataset:
    """Reads epochs dataset in '.pt' format"""
    return torch.load(path)