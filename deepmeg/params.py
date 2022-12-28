from collections import namedtuple
import mneflow
import mneflow as mf
import numpy as np
import tensorflow as tf
import scipy as sp
import scipy.signal as sl
import pickle
from typing import Optional, NoReturn, Any


SpatialParameters = namedtuple('SpatialParameters', 'patterns filters')
TemporalParameters = namedtuple('TemporalParameters', 'franges finputs foutputs fresponces fpatterns')
ComponentsOrder = namedtuple('ComponentsOrder', 'l2 compwise_loss weight output_corr weight_corr')
Predictions = namedtuple('Predictions', 'y_p y_true')
WaveForms = namedtuple('WaveForms', 'evoked evoked_filt induced induced_filt times tcs')
CompressionParameters = namedtuple('CompressionParameters', 'loss_estimate eig_estimate envelopes compression_weights')


def compute_patterns(model, data_path=None, *, output='patterns'):

    if not data_path:
        print("Computing patterns: No path specified, using validation dataset (Default)")
        ds = model.dataset.val
    elif isinstance(data_path, str) or isinstance(data_path, (list, tuple)):
        ds = model.dataset._build_dataset(
            data_path,
            split=False,
            test_batch=None,
            repeat=True
        )
    elif isinstance(data_path, mneflow.data.Dataset):
        if hasattr(data_path, 'test'):
            ds = data_path.test
        else:
            ds = data_path.val
    elif isinstance(data_path, tf.data.Dataset):
        ds = data_path
    else:
        raise AttributeError('Specify dataset or data path.')

    X, y = [row for row in ds.take(1)][0]

    model.out_w_flat = model.fin_fc.w.numpy()
    model.out_weights = np.reshape(
        model.out_w_flat,
        [-1, model.dmx.size, model.out_dim]
    )
    model.out_biases = model.fin_fc.b.numpy()
    model.feature_relevances = model.get_component_relevances(X, y)

    # compute temporal convolution layer outputs for vis_dics
    tc_out = model.pool(model.tconv(model.dmx(X)).numpy())

    # compute data covariance
    X = X - tf.reduce_mean(X, axis=-2, keepdims=True)
    X = tf.transpose(X, [3, 0, 1, 2])
    X = tf.reshape(X, [X.shape[0], -1])
    model.dcov = tf.matmul(X, tf.transpose(X))

    # get spatial extraction fiter weights
    demx = model.dmx.w.numpy()
    model.lat_tcs = np.dot(demx.T, X)

    kern = np.squeeze(model.tconv.filters.numpy()).T

    X = X.numpy().T
    if 'patterns' in output:
        if 'old' in output:
            model.patterns = np.dot(model.dcov, demx)
        else:
            patterns = []
            X_filt = np.zeros_like(X)
            for i_comp in range(kern.shape[0]):
                for i_ch in range(X.shape[1]):
                    x = X[:, i_ch]
                    X_filt[:, i_ch] = np.convolve(x, kern[i_comp, :], mode="same")
                patterns.append(np.cov(X_filt.T) @ demx[:, i_comp])
            model.patterns = np.array(patterns).T
    else:
        model.patterns = demx

    model.lat_tcs_filt = np.dot(demx.T, X_filt)

    del X

    #  Temporal conv stuff
    model.filters = kern.T
    model.tc_out = np.squeeze(tc_out)
    model.corr_to_output = model.get_output_correlations(y)


def compute_temporal_parameters(model, *, fs=None):

    if fs is None:

        if model.dataset.h_params['fs']:
            fs = model.dataset.h_params['fs']
        else:
            print('Sampling frequency not specified, setting to 1.')
            fs = 1.

    out_filters = model.filters
    _, psd = sl.welch(model.lat_tcs, fs=fs, nperseg=fs * 2)
    finputs = psd[:, :-1]
    franges = None
    foutputs = list()
    fresponces = list()
    fpatterns = list()

    for i, flt in enumerate(out_filters.T):
        w, h = (lambda w, h: (w, np.abs(h)))(*sl.freqz(flt, 1, worN=fs))
        foutputs.append(np.real(finputs[i, :] * h * np.conj(h)))
        fpatterns.append(np.abs(finputs[i, :] * h))

        if franges is None:
            franges = w / np.pi * fs / 2
        fresponces.append(h)

    return franges, finputs, foutputs, fresponces, fpatterns


def get_order(order: np.array, *args):
    return order.ravel()


def compute_morlet_cwt(
    sig: np.ndarray,
    t: np.ndarray,
    freqs: np.ndarray,
    omega_0: Optional[float] = 5,
    phase: Optional[bool] = False
) -> np.ndarray:
    dt = t[1] - t[0]
    widths = omega_0 / (2 * np.pi * freqs * dt)
    cwtmatr = sl.cwt(sig, lambda M, s: sl.morlet2(M, s, w=omega_0), widths)
    if phase:
        return cwtmatr
    else:
        return np.real(cwtmatr)**2 + np.imag(cwtmatr)**2


def compute_waveforms(model: mf.models.BaseModel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time_courses = np.squeeze(model.lat_tcs.reshape(
        [model.specs['n_latent'], -1, model.dataset.h_params['n_t']]
    ))
    time_courses_filtered = np.squeeze(model.lat_tcs_filt.reshape(
        [model.specs['n_latent'], -1, model.dataset.h_params['n_t']]
    ))
    times = (1 / float(model.dataset.h_params['fs'])) *\
        np.arange(model.dataset.h_params['n_t'])
    induced = list()
    induced_filt = list()

    for tc, tc_filt in zip(time_courses, time_courses_filtered):
        ls_induced = list()
        ls_induced_filt = list()

        for lc, lc_filt in zip(tc, tc_filt):
            freqs = np.arange(1, model.dataset.h_params['fs']//2)
            ls_induced.append(np.abs(compute_morlet_cwt(lc, times, freqs)))
            ls_induced_filt.append(np.abs(compute_morlet_cwt(lc_filt, times, freqs)))

        induced.append(np.array(ls_induced).mean(axis=0))
        induced_filt.append(np.array(ls_induced_filt).mean(axis=0))

    return np.array(induced), np.array(induced_filt), times, time_courses


def save_parameters(content: Any, path: str, parameters_type: Optional[str] = '') -> NoReturn:

    parameters_type = parameters_type + ' ' if parameters_type else parameters_type
    print(f'Saving {parameters_type} parameters...')

    if path[-4:] != '.pkl':
        raise OSError(f'Pickle file must have extension ".pkl", but it has "{path[-4:]}"')

    pickle.dump(content, open(path, 'wb'))

    print('Successfully saved')


def eigencentrality(matrix: np.ndarray) -> np.ndarray:
    """
    Calculate the eigencentrality of a matrix.

    Parameters
    ----------
    matrix : np.ndarray
        The matrix to calculate the eigencentrality of.

    Returns
    -------
    np.ndarray
        The eigencentrality of the matrix.
    """
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    eigencentrality = eigenvectors[:,0]
    return eigencentrality


def moving_average(data: np.ndarray, kernel_size: int = 20) -> np.ndarray:
    """
    Compute the moving average of a given data array.

    Parameters
    ----------
    data : np.ndarray
        The data array.
    kernel_size : int
        The size of the kernel.

    Returns
    -------
    np.ndarray
        The moving average of the data array.
    """
    kernel = np.ones(kernel_size) / kernel_size
    data_convolved = np.convolve(data, kernel, mode='same')
    return np.concatenate([
        [np.nan for i in range(kernel_size//2)],
        sp.stats.zscore(data_convolved[kernel_size//2:-kernel_size//2]),
        [np.nan for i in range(kernel_size//2)],
    ])


def compute_compression_parameters(model) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the cropping of the latent components time courses.

    Parameters
    ----------
    model : mneflow.models.BaseModel
        The model to compute the cropping for.

    Returns
    -------
    temp_relevance_loss : np.ndarray
        The temporal relevance loss.
    eigencentrality : np.ndarray
        The eigencentrality of compression weights covariance.
    time_courses_env : np.ndarray
        The latent component envelopes.
    compression_weights : np.ndarray
        The compression weights.
    """
    time_courses_filtered = model.lat_tcs_filt
    time_courses_env = np.zeros_like(time_courses_filtered)
    kern = np.squeeze(model.envconv.filters.numpy()).T
    for i_comp in range(kern.shape[0]):
        conv = np.abs(np.convolve(time_courses_filtered[i_comp, :], kern[i_comp, :], mode="same"))
        time_courses_env[i_comp, :] = conv
    compression_weights = np.array([
        pool.weights[0].numpy()
        for pool in model.pool_list
    ])

    return model.temp_relevance_loss,\
        np.array([eigencentrality(np.cov(w)) for w in compression_weights]),\
        time_courses_env,\
        compression_weights
