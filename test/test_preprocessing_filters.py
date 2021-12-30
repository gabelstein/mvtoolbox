import numpy as np

from mvtoolbox.preprocessing.filters import BandPassFilter


def test_bandpass_filters():
    """Test BandPassFilters"""
    filters = BandPassFilter(filter_bands=[[4, 8]], sample_rate=1000)


def test_bandpass_filters():
    """Test BandPassFilters"""
    test_data = np.array([6*np.sin(np.linspace(0, 1, 1000))])
    filters = BandPassFilter(filter_bands=[[4, 8]], sample_rate=1000)
    filtered = filters.fit_transform(test_data)