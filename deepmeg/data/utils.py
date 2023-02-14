import random
import torch
from torch.utils.data import DataLoader, Dataset


def compute_train_val_indices(n_samples: int, batch_size: int, val_batch_size: int, shuffle: bool = True) -> tuple[list[int], list[int]]:
    """
    Computes the indices for the training and validation sets.

    Args:
        n_samples (int): the total number of samples in the dataset
        batch_size (int): the number of samples in a batch for the training set
        val_batch_size (int): the number of samples in a batch for the validation set
        shuffle (bool): whether to shuffle indices

    Returns:
        Tuple[List[int], List[int]] : a tuple containing two lists of integers representing the indices of the samples in the training and validation sets respectively.
    """
    val_ratio = val_batch_size/(batch_size + val_batch_size)
    all_indices = list(range(n_samples))

    if shuffle:
        random.shuffle(all_indices)

    n_indices = len(all_indices)
    start = 0
    dist = batch_size + val_batch_size
    end = dist
    train_indices, val_indices = list(), list()

    while True:
        slice_ = slice(start, end)
        group = all_indices[slice_]
        actual_batch_size = int(len(group)*val_ratio)
        train_indices += group[actual_batch_size:]
        val_indices += group[:actual_batch_size]

        if end is None:
            break

        start = end
        estimated_end = end + dist
        end = estimated_end if estimated_end <= n_indices else None

    return train_indices, val_indices


def make_train_and_val_loaders(
    dataset: Dataset,
    batch_size: int,
    val_batch_size: int,
    shuffle: bool=True
) -> tuple[DataLoader, DataLoader]:
    """
    Creates PyTorch DataLoaders for the training and validation sets.

    Args:
        dataset (Dataset): the dataset to create the DataLoaders from
        batch_size (int): the number of samples in a batch for the training set
        val_batch_size (int): the number of samples in a batch for the validation set
        shuffle (bool): whether to shuffle the dataset before splitting into train and val sets
    Returns:
        Tuple[DataLoader, DataLoader]: a tuple of PyTorch DataLoader objects for the training and validation sets respectively.
    """
    train_indices, val_indices = compute_train_val_indices(len(dataset), batch_size, val_batch_size, shuffle=shuffle)
    return DataLoader(dataset, batch_size, sampler=train_indices), DataLoader(dataset, val_batch_size, sampler=val_indices)

