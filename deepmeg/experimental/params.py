from collections import namedtuple
import pickle
from abc import ABC, abstractmethod
from typing import Any
import os
import numpy as np
from .interpreters import SPIRITInterpreter
from ..utils.params import LFCNNParameters, SpatialParameters, SpectralParameters, TemporalParameters, compute_spectrums
import scipy.signal as sl
import scipy as sp


class SPIRITParameters(LFCNNParameters):
    """A class representing the parameters required for the SPIRIT neural network.

    This class is designed to provide the parameters necessary for a specific neural network. The `SPIRITParameters` class
    takes a `SPIRITInterpreter` object, which is used to generate the spatial, spectral, and temporal parameters needed
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
        save(path: str | os.PathLike): Saves the `SPIRITParameters` object to a .pkl file.
        read_pkl(path: str | os.PathLike) -> 'SPIRITParameters': Loads the `SPIRITParameters` object from a .pkl file.
    """
    def __init__(self, interpreter: SPIRITInterpreter):
        self._info = interpreter.info
        self._spatial = SpatialParameters(interpreter.spatial_patterns, interpreter.spatial_filters)
        self._spectral = SpectralParameters(
            interpreter.frequency_range,
            interpreter.filter_inputs,
            interpreter.filter_outputs,
            interpreter.filter_responses,
            interpreter.filter_patterns
        )
        times = np.arange(0, interpreter.latent_sources.shape[-1]/interpreter.info['sfreq'], 1/interpreter.info['sfreq'])
        spectrums, spectrums_filtered = compute_spectrums(interpreter)
        branch_tempwise_estimate = interpreter.temporal_patterns.mean(0)
        interp_cubic = lambda y: sp.interpolate.interp1d(np.linspace(0, times[-1], y.shape[1]), y, 'cubic')(times)

        self._temporal = TemporalParameters(
            times,
            interpreter.latent_sources,
            interpreter.latent_sources_filtered,
            spectrums, spectrums_filtered,
            interp_cubic(branch_tempwise_estimate)
        )
        self._branchwise_loss = interpreter.branchwise_loss
