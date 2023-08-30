from collections import namedtuple
import pickle
from abc import ABC, abstractmethod
from typing import Any
import os
import numpy as np
from ..interpreters import LFCNNInterpreter
import scipy.signal as sl
import scipy as sp


SpatialParameters = namedtuple('SpatialParameters', 'patterns filters')
SpectralParameters = namedtuple('SpectralParameters', 'range inputs outputs responses patterns filters')
TemporalParameters = namedtuple('TemporalParameters', 'times time_courses time_courses_filtered spectrums spectrums_filtered patterns', defaults=[None])
Predictions = namedtuple('Predictions', 'y_p y_true')


def save(content: Any, path: str | os.PathLike):
    """
    Save an object using pickle serialization.

    Args:
    content (Any): the object to be serialized and saved.
    path (str | os.PathLike): the path and filename where the serialized object will be saved.
    The file extension must be '.pkl'.

    Raises:
    OSError: if the file extension of path is not '.pkl'.

    Returns:
    None
    """

    if path[-4:] != '.pkl':
        raise OSError(f'Pickle file must have extension ".pkl", but it has "{path[-4:]}"')

    pickle.dump(content, open(path, 'wb'))


def read_pkl(path: str | os.PathLike) -> Any:
    """
    Read a pickled object from a file.

    Args:
    path (str or os.PathLike): Path to the pickled file to be read.

    Returns:
    Any: The content of the pickled file.

    Raises:
    FileNotFoundError: If the file specified by the path does not exist.
    OSError: If the file specified by the path is not a valid pickle file.
    """
    with open(
        path,
        'rb'
    ) as file:
        content = pickle.load(
            file
        )
    return content


def compute_morlet_cwt(
    sig: np.ndarray,
    sfreq: float,
    omega_0: float = 5,
    phase: bool = False
) -> np.ndarray:
    """
    Computes the continuous wavelet transform (CWT) of a signal using a Morlet wavelet.

    Args:
        sig (numpy.ndarray): Input signal to be transformed.
        sfreq (float): Sampling frequency of the input signal.
        omega_0 (float, optional): Center frequency of the wavelet. Defaults to 5.
        phase (bool, optional): Flag to control the return type of the transformed data.2
                                If `True`, only the phase angle will be returned.
                                Otherwise, the magnitude squared will be returned. Defaults to False.

    Returns:
        numpy.ndarray: Transformed data of the input signal with a Morlet wavelet.
                        If `phase` is `True`, the function returns a complex array of the same shape as `sig`.
                        Otherwise, it returns a real array of the same shape as `sig`.
    """
    dt = 1/sfreq
    freqs = np.arange(1, sfreq//2)
    widths = omega_0 / ( 2*np.pi * freqs * dt)
    cwtmatr = sl.cwt(sig, lambda M, s: sl.morlet2(M, s, w=omega_0), widths)
    if phase:
        return cwtmatr
    else:
        return np.real(cwtmatr)**2 + np.imag(cwtmatr)**2


def compute_spectrums(interpreter: LFCNNInterpreter) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes spectrum of latent sources activity.

    Args:
        interpreter (LFCNNInterpreter): Instance of the LFCNNInterpreter class.

    Returns:
        tuple: A tuple of two numpy arrays, each with shape (n_sources, n_frequencies, n_times), containing the
        induced activity for the original (unfiltered) and the filtered latent sources.
    """
    time_courses = np.transpose(interpreter.latent_sources, (1, 0, 2))
    time_courses_filtered = np.transpose(interpreter.latent_sources_filtered, (1, 0, 2))
    spectrums = list()
    spectrums_filt = list()

    for tc, tc_filt in zip(time_courses, time_courses_filtered):
        ls_spectrums = list()
        ls_spectrums_filt = list()

        for lc, lc_filt in zip(tc, tc_filt):
            ls_spectrums.append(np.abs(compute_morlet_cwt(lc, interpreter.info['sfreq'], 15)))#, 7.5)))
            ls_spectrums_filt.append(np.abs(compute_morlet_cwt(lc_filt, interpreter.info['sfreq'], 15)))#, 7.5)))

        spectrums.append(np.array(ls_spectrums))#.mean(axis=0))
        spectrums_filt.append(np.array(ls_spectrums_filt))#.mean(axis=0))

    return np.array(spectrums), np.array(spectrums_filt)


class NetworkParameters(ABC):
    """The NetworkParameters class is an abstract base class that defines the basic interface for neural network parameter objects. It includes abstract properties for the spatial, spectral, and temporal parameters of the network, as well as properties for the order and additional information. The class also provides methods for saving the object to a pickle file and for reading a pickle file.

    Attributes:

        None

    Methods:

        save(path: str | os.PathLike) -> None: Saves the object to a pickle file.
        read_pkl(path: str | os.PathLike) -> 'NetworkParameters': Reads a pickle file and returns the contents as a new NetworkParameters object.

    Abstract Properties:

        spatial: An abstract property that returns a SpatialParameters object containing spatial filter and pattern information.
        spectral: An abstract property that returns a SpectralParameters object containing spectral input, output, and response information.
        temporal: An abstract property that returns a TemporalParameters object containing temporal information such as times, time courses, and induced activity.
        order: An abstract property that returns a string indicating the order of the model.
        info: An abstract property that returns a dictionary containing additional information about the model.
    """
    @property
    @abstractmethod
    def spatial(self):
        ...
    @property
    @abstractmethod
    def spectral(self):
        ...
    @property
    @abstractmethod
    def temporal(self):
        ...

    @property
    @abstractmethod
    def order(self):
        ...

    @property
    @abstractmethod
    def info(self):
        ...

    def save(self, path: str | os.PathLike):
        """
        Save an NetworkParameters object using pickle serialization.

        Args:
        path (str | os.PathLike): the path and filename where the serialized object will be saved.
        The file extension must be '.pkl'.

        Raises:
        OSError: if the file extension of path is not '.pkl'.

        Returns:
        None
    """
        save(self, path)

    @staticmethod
    def load(path: str | os.PathLike) -> 'NetworkParameters':
        """
        Read a pickled NetworkParameters object from a file.

        Args:
        path (str or os.PathLike): Path to the pickled file to be read.

        Returns:
        NetworkParameters: The content of the pickled file.

        Raises:
        FileNotFoundError: If the file specified by the path does not exist.
        OSError: If the file specified by the path is not a valid pickle file.
        """
        return read_pkl(path)


class LFCNNParameters(NetworkParameters):
    """A class representing the parameters required for the LFCNN neural network.

    This class is designed to provide the parameters necessary for a specific neural network. The `LFCNNParameters` class
    takes a `LFCNNInterpreter` object, which is used to generate the spatial, spectral, and temporal parameters needed
    for the network.

    Attributes:
        spatial (SpatialParameters): A named tuple that contains the spatial patterns and filters.
        spectral (SpectralParameters): A named tuple that contains the frequency range, filter inputs, outputs,
            responses and patterns.
        temporal (TemporalParameters): A named tuple that contains the times, time courses, time courses filtered,
            induceds, induceds filtered and patterns.
        info (mne.Info): A mne.Info object containing information about the data used in the network.
        order (list[float]): A sequence representing the order according to the loss function used by the network.

    Methods:
        save(path: str | os.PathLike): Saves the `LFCNNParameters` object to a .pkl file.
        read_pkl(path: str | os.PathLike) -> 'LFCNNParameters': Loads the `LFCNNParameters` object from a .pkl file.
    """
    def __init__(self, interpreter: LFCNNInterpreter):
        """Initializes the `LFCNNParameters` object.

        Args:
            interpreter (LFCNNInterpreter): An `LFCNNInterpreter` object that is used to generate the required parameters.
        """
        self._info = interpreter.info
        self._spatial = SpatialParameters(interpreter.spatial_patterns, interpreter.spatial_filters)
        self._spectral = SpectralParameters(
            interpreter.frequency_range,
            interpreter.filter_inputs,
            interpreter.filter_outputs,
            interpreter.filter_responses,
            interpreter.filter_patterns,
            interpreter.filters
        )
        times = np.arange(0, interpreter.latent_sources.shape[-1]/interpreter.info['sfreq'], 1/interpreter.info['sfreq'])
        spectrums, spectrums_filtered = compute_spectrums(interpreter)
        self._temporal = TemporalParameters(
            times,
            interpreter.latent_sources,
            interpreter.latent_sources_filtered,
            spectrums, spectrums_filtered
        )
        self._branchwise_loss = interpreter.branchwise_loss

    @property
    def spatial(self):
        """A property getter method that returns the `SpatialParameters` attribute of the `LFCNNParameters` object.

        Returns:
            SpatialParameters: A named tuple containing the spatial patterns and filters.
        """
        return self._spatial

    @property
    def spectral(self):
        """A property getter method that returns the `SpectralParameters` attribute of the `LFCNNParameters` object.

        Returns:
            SpectralParameters: A named tuple containing the frequency range, filter inputs, outputs, responses and
                patterns.
        """
        return self._spectral

    @property
    def temporal(self):
        """A property getter method that returns the `TemporalParameters` attribute of the `LFCNNParameters` object.

        Returns:
            TemporalParameters: A named tuple containing the times, time courses, time courses filtered, induceds,
                induceds filtered and patterns.
        """
        return self._temporal

    @property
    def info(self):
        """
        Returns a mne.Info object containing the information about the recordings.

        Returns
        """
        return self._info

    @property
    def order(self):
        """
        This property returns the order of the branches of the LFCNN model.

        Returns:
        int: The order of the LFCNN model brancehs.
        """
        return -self._branchwise_loss
