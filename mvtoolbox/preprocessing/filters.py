import mne as mne
import scipy
import numpy as np
from scipy.stats import zscore
from sklearn.base import TransformerMixin, BaseEstimator
import math
from scipy.signal import butter, lfilter


class BandPassFilter(BaseEstimator, TransformerMixin):
    """Band Pass filtering.
    ----------
    filter_bands : list
        bands to filter signal with
    sample_rate : int
        Signal sample rate
    filter_len : int,
        lenght of the filter. The default is 1001.
    l_trans_bandwidth : TYPE, optional
        DESCRIPTION. The default is 4.
    h_trans_bandwidth : TYPE, optional
        DESCRIPTION. The default is 4.
    """

    def __init__(self, filter_bands, sample_rate, filter_len='1000ms', l_trans_bandwidth=4, h_trans_bandwidth=4):
        self.filter_bands = filter_bands
        self.sample_rate = sample_rate
        self.filter_len = filter_len
        self.l_trans_bandwidth = l_trans_bandwidth
        self.h_trans_bandwidth = h_trans_bandwidth
        self.filters = _calc_band_filters(self.filter_bands,
                                               self.sample_rate,
                                               self.filter_len,
                                               self.l_trans_bandwidth,
                                               self.h_trans_bandwidth)

    def fit(self, X, y=None):
        self.filters = _calc_band_filters(self.filter_bands,
                                               self.sample_rate,
                                               self.filter_len,
                                               self.l_trans_bandwidth,
                                               self.h_trans_bandwidth)
        return self

    def transform(self, X, y=None):
        X_ = _apply_filter(X, self.filters)
        return X_

    def fit_transform(self, X, y=None):
        self.filters = _calc_band_filters(self.filter_bands,
                                               self.sample_rate,
                                               self.filter_len,
                                               self.l_trans_bandwidth,
                                               self.h_trans_bandwidth)
        X_ = _apply_filter(X, self.filters)
        return X_


def _calc_band_filters(f_ranges, sample_rate, filter_len="1000ms", l_trans_bandwidth=4, h_trans_bandwidth=4):
    """
    This function returns for the given frequency band ranges filter coefficients with with length "filter_len"
    Thus the filters can be sequentially used for band power estimation
    Parameters
    ----------
    f_ranges : TYPE
        DESCRIPTION.
    sample_rate : float
        sampling frequency.
    filter_len : int,
        lenght of the filter. The default is 1001.
    l_trans_bandwidth : TYPE, optional
        DESCRIPTION. The default is 4.
    h_trans_bandwidth : TYPE, optional
        DESCRIPTION. The default is 4.
    Returns
    -------
    filter_fun : array
        filter coefficients stored in rows.
    """
    filter_fun = []

    for a, f_range in enumerate(f_ranges):
        h = mne.filter.create_filter(None, sample_rate, l_freq=f_range[0], h_freq=f_range[1],
                                     fir_design='firwin', l_trans_bandwidth=l_trans_bandwidth,
                                     h_trans_bandwidth=h_trans_bandwidth, filter_length=filter_len, verbose=False)

        filter_fun.append(h)

    return np.array(filter_fun)


def _apply_filter(dat_, filter_fun):
    """
    For a given channel, apply previously calculated filters
    Parameters
    ----------
    dat_ : array (ns,)
        segment of data at a given channel and downsample index.
    sample_rate : float
        sampling frequency.
    filter_fun : array
        output of calc_band_filters.
    line_noise : int|float
        (in Hz) the line noise frequency.
    seglengths : list
        list of ints with the leght to which variance is calculated.
        Used only if variance is set to True.
    variance : bool,
        If True, return the variance of the filtered signal, else
        the filtered signal is returned.
    Returns
    -------
    filtered : array
        if variance is set to True: (nfb,) array with the resulted variance
        at each frequency band, where nfb is the number of filter bands used to decompose the signal
        if variance is set to False: (nfb, filter_len) array with the filtered signal
        at each freq band, where nfb is the number of filter bands used to decompose the signal
    """
    filtered = []

    for filt in range(filter_fun.shape[0]):
        for ch in dat_.T:
            filtered.append(scipy.signal.convolve(ch, filter_fun[filt, :], mode='same'))

    return np.array(filtered).T


class NotchFilter(BaseEstimator, TransformerMixin):
    """Notch filtering.
    ----------
    sample_rate : int
        Signal sample rate.
    line_noise : int
        Line noise
    """

    def __init__(self, line_noise, sample_rate):
        self.sample_rate = sample_rate
        self.line_noise = line_noise

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = self._notch_filter(X)
        return X_

    def fit_transform(self, X, y=None):
        X_ = self._notch_filter(X)
        return X_

    def _notch_filter(self, dat_):
        dat_notch_filtered = mne.filter.notch_filter(x=dat_.T, Fs=self.sample_rate, trans_bandwidth=7,
                                                     freqs=np.arange(self.line_noise, 4 * self.line_noise,
                                                                     self.line_noise),
                                                     fir_design='firwin', verbose=False, notch_widths=1,
                                                     filter_length=dat_.shape[0] - 1)
        return dat_notch_filtered.T
