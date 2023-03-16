import numpy as np
from typing import Generator
from itertools import product

def __pad_left_right(window_size: int, padding: int | str | tuple[int, int]) -> tuple[int, int]:
    """
    Computes the left and right padding for a convolution operation given the window size and padding.

    Args:
        window_size: the size of the sliding window
        padding: the amount of padding to add to the beginning and end of the array. Can be an integer, string, or tuple of integers.
            If padding is an integer, the same padding is applied to the beginning and end of the array.
            If padding is a string, it must be one of 'VALID', 'SAME', or 'FULL'.
                'VALID' indicates that no padding is applied.
                'SAME' indicates that the padding is chosen so that the output is the same size as the input.
                'FULL' indicates that the padding is equal to the window size minus 1.
            If padding is a tuple of integers, it must contain two integers representing the left and right padding.

    Returns:
        A tuple of two integers representing the left and right padding.
    """
    if isinstance(padding, (int, str)):
        if padding == 'VALID' or padding == 'valid':
            padding_left = 0
            padding_right = 0
        elif padding == 'SAME' or padding == 'same':
            if not window_size%2:
                padding_left = max(window_size // 2, 0)
                padding_right = padding_left - 1
            else:
                padding_left = (window_size - 1) // 2
                padding_right = padding_left
        elif padding == 'FULL' or padding == 'full':
            padding_left = window_size - 1
            padding_right = window_size - 1
        else:
            padding_left, padding_right = padding, padding

        assert padding_left >= 0 and padding_right >= 0, 'Padding can not be less than 0'
        assert padding_left < window_size and padding_right < window_size, 'Padding can not be greater than window_size'

    else:
        padding_left, padding_right = padding

    return padding_left, padding_right


def compute_output_shape(shape: tuple[int, ...], kernel_size: tuple[int, ...], paddings: int | tuple[int, ...] = 0, strides: int | tuple[int, ...] = 1) -> tuple[int, ...]:
    """
    Computes the shape of the output tensor of a convolution operation given the shape of the input tensor, kernel size, padding, and stride.

    Args:
        shape: a tuple of integers representing the shape of the input tensor
        kernel_size: a tuple of integers representing the size of the kernel in each dimension
        paddings: the amount of padding to add to the beginning and end of the input tensor in each dimension (default is 0)
        strides: the number of indices to skip when moving the kernel in each dimension (default is 1)

    Returns:
        A tuple of integers representing the shape of the output tensor.
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,)
    if isinstance(paddings, (int, str)):
        paddings = [paddings for _ in range(len(kernel_size))]
    paddings = [__pad_left_right(window_size, padding) for window_size, padding in zip(kernel_size, paddings)]
    if isinstance(strides, (int, str)):
        strides = [strides for _ in range(len(kernel_size))]

    output_shape = []
    for dim, k, (p_left, p_right), s in zip(shape, kernel_size, paddings, strides):
        output_dim = (dim + (p_left + p_right) - k) // s + 1
        output_shape.append(output_dim)
    return tuple(output_shape + [1 for _ in range(len(shape) - len(kernel_size))])


def conv_indices_generator(
    array_len: int,
    window_size: int,
    padding: int | str | tuple[int, int] = 0,
    stride: int = 1
) -> Generator[int | tuple[int, int], None, None]:
    """
    Generates indices for convolution of an array of a certain length with a sliding window.

    Args:
        array_len: the length of the array that the convolution is being applied to
        window_size: the size of the sliding window
        padding: the amount of padding to add to the beginning and end of the array (default is 0)
        stride: the number of indices to skip when moving the window (default is 1)

    Yields:
        A tuple of two integers representing the indices of the array elements in the current window position.
    """
    assert window_size > 0, 'Sliding window can not be less or equal to 0'
    assert stride > 0, 'Stride can not be less or equal to 0'
    padding_left, padding_right = __pad_left_right(window_size, padding)

    for i in range(0, array_len + padding_right, stride):
        from_ = min(array_len - 1, max(i-padding_left, 0))

        if array_len + padding_right - from_ < window_size:
            break

        to = max(min(array_len - 1, i - padding_left + window_size - 1), 0)

        if from_ != to:
            yield from_, to
        else:
            yield from_


def compute_kernel_coords(kernel_size: int, *coords: slice) -> tuple[slice, ...]:
    """
    Computes the kernel coordinates for a convolution operation given the kernel size and coordinates.

    Args:
        kernel_size: the size of the kernel
        *coords: the coordinates to be transformed. Should be slice objects.

    Returns:
        A tuple of slice objects representing the transformed coordinates.
    """
    out = list()
    for coord in coords:
        if coord == slice(0, None, None):
            out.append(coord)
        else:
            dist = abs(coord.stop - coord.start)
            if coord.start == 0:
                out.append(slice(-dist, None))
            else:
                out.append(slice(None, dist))
    if len(out) == 1:
        return out[0]
    else:
        return tuple(out)


def conviter(
    shape: tuple[int, int],
    kernel_size: tuple[int, ...],
    paddings: int | tuple[int, ...] = 0,
    strides: int | tuple[int, ...] = 1
) -> Generator[slice | tuple[slice], None, None]:
    """
    Generates indices for convolution of an array of a certain shape with a sliding window in n-dimensional space.

    Args:
        shape: a tuple of integers representing the shape of the array that the convolution is being applied to
        kernel_size: a tuple of integers representing the size of the sliding window in each dimension
        paddings: the amount of padding to add to the beginning and end of the array in each dimension (default is 0)
        strides: the number of indices to skip when moving the window in each dimension (default is 1)

    Yields:
        A tuple of integers representing the indices of the array elements in the current window position in n-dimensional space.
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,)
    assert len(kernel_size) <= len(shape), 'It is not possible to perform convolution with a kernel of a higher dimension than the data'

    if isinstance(paddings, (int, str)):
        paddings = [paddings for _ in range(len(kernel_size))]
    assert len(paddings) == len(kernel_size), 'Number of paddings must be consistent with the dimensionality of the kernel'

    if isinstance(strides, (int, str)):
        strides = [strides for _ in range(len(kernel_size))]
    assert len(strides) == len(kernel_size), 'Number of strides must be consistent with the dimensionality of the kernel'

    whole_axis = (0, None)
    plug = [whole_axis for _ in range(len(shape) - len(kernel_size))]
    out_plug = [0 for _ in range(len(plug))]
    conv_indices = [[indices if isinstance(indices, tuple) else (indices,) for indices in conv_indices_generator(*conv_args)] for conv_args in zip(shape, kernel_size, paddings, strides)]

    for indices, out_indices in zip(zip(product(*conv_indices)), zip(product(*[range(len(conv_i)) for conv_i in conv_indices]))):
        conv_out = list(indices[0]) + plug
        out = list(out_indices[0]) + out_plug
        conv_out = tuple(slice(ax[0], ax[-1]+1) if ax[-1] != None else slice(ax[0], ax[-1]) for ax in conv_out)

        if len(conv_out) == 1:
            yield conv_out[0], tuple(out), compute_kernel_coords(kernel_size, conv_out[0])
        else:
            yield conv_out, tuple(out), compute_kernel_coords(kernel_size, *conv_out)


def align_kernel(data: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Aligns the kernel of the convolution with the shape of the data.

    Args:
        data: a numpy array representing the data
        kernel: a numpy array representing the kernel

    Returns:
        A numpy array representing the aligned kernel.
    """
    free_dims = [data.shape[j] for j in range(len(kernel.shape), len(data.shape))]
    new_kernel = kernel.copy()
    if free_dims:
        for free_dim in free_dims:
            new_kernel = np.array([new_kernel for _ in range(free_dim)])
            new_kernel = np.transpose(new_kernel, list(range(len(new_kernel.shape))[1:]) + [0])
    return new_kernel

