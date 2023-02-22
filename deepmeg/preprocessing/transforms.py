import torch
import numpy as np
from typing import TypeVar

dType = TypeVar('dType', np.ndarray, torch.Tensor)


def zscore(x: dType) -> dType:
    return (x - x.mean())/x.std()


def rowwise_zscore(x: dType) -> dType:
    """
    The rowwise_zscore function computes the z-score of each row of a given 2D array of numerical data.

    Args:
    x (np.ndarray | torch.Tensor): The input array of shape (m, n) where m is the number of rows and n is the number of columns.

    Returns:
    np.ndarray | torch.Tensor: An array of the same shape as the input data where each row has been standardized to have zero mean and unit variance.

    Raises:
    ValueError: If the input array has less than two dimensions.

    Example:
    >>> data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> rowwise_zscore(data)
    array([[-1.22474487, 0. , 1.22474487],
    [-1.22474487, 0. , 1.22474487],
    [-1.22474487, 0. , 1.22474487]])
    """
    return (x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, keepdims=True)


def one_hot_encoder(Y: np.ndarray) -> np.ndarray:
    """One-hot encoder

    Args:
        Y (np.ndarray): Data to encode with shape (n_samples, n_classes)

    Returns:
        np.ndarray: encoded data with shape (n_samples,)
    """
    y = list()
    n_classes = len(np.unique(Y))

    for val in Y:
        new_y_value = np.zeros(n_classes)
        new_y_value[val - 1] = 1
        y.append(new_y_value)

    return np.array(y)


def one_hot_decoder(y: np.array) -> np.array:
    """One-hot decoder

    Args:
        Y (np.ndarray): Data to decode with shape (n_samples,)

    Returns:
        np.ndarray: decoded data with shape (n_samples, n_classes)
    """
    y_decoded = list()
    for val in y:
        y_decoded.append(np.where(val == val.max())[0][0])

    return np.array(y_decoded)


def interpolate_sequence(original_sequence: np.ndarray, new_length: int) -> np.ndarray:
    """
    Interpolate a sequence of length 'n' into a sequence of length 'm' (m > n)

    Args:
        original_sequence: A numpy array of shape (n, ) representing the original sequence
        new_length: An integer, the desired length of the interpolated sequence

    Returns:
        A numpy array of shape (m, ) representing the interpolated sequence
    """
    x = np.arange(original_sequence.shape[0])
    f = original_sequence
    x_new = np.linspace(0, x.max(), new_length)
    f_new = np.interp(x_new, x, f)
    return f_new
