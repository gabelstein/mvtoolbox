import numpy
from scipy.linalg import block_diag
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.covariance import ledoit_wolf, oas, empirical_covariance, fast_mcd
np = numpy

def _lwf(X):
    """Wrapper for sklearn ledoit wolf covariance estimator"""
    C, _ = ledoit_wolf(X.T)
    return C


def _oas(X):
    """Wrapper for sklearn oas covariance estimator"""
    C, _ = oas(X.T)
    return C


def _scm(X):
    """Wrapper for sklearn sample covariance estimator"""
    return empirical_covariance(X.T)


def _mcd(X):
    """Wrapper for sklearn mcd covariance estimator"""
    _, C, _, _ = fast_mcd(X.T)
    return C


def _sch(X):
    """Schaefer-Strimmer covariance estimator
    Shrinkage estimator using method from [1]_:
    .. math::
            \hat{\Sigma} = (1 - \gamma)\Sigma_{scm} + \gamma T
    where :math:`T` is the diagonal target matrix:
    .. math::
            T_{i,j} = \{ \Sigma_{scm}^{ii} \text{if} i = j, 0 \text{otherwise} \}
    Note that the optimal :math:`\gamma` is estimated by the authors' method.
    :param X: Signal matrix, (n_channels, n_times)
    :returns: Schaefer-Strimmer shrinkage covariance matrix, (n_channels, n_channels)
    Notes
    -----
    .. versionadded:: 0.2.8
    References
    ----------
    .. [1] Schafer, J., and K. Strimmer. 2005. A shrinkage approach to
        large-scale covariance estimation and implications for functional
        genomics. Statist. Appl. Genet. Mol. Biol. 4:32.
    """  # noqa
    n_times = X.shape[1]
    X_c = (X.T - X.T.mean(axis=0)).T
    C_scm = 1. / n_times * X_c @ X_c.T

    # Compute optimal gamma, the weigthing between SCM and srinkage estimator
    R = (n_times / ((n_times - 1.) * np.outer(X.std(axis=1), X.std(axis=1))))
    R *= C_scm
    var_R = (X_c ** 2) @ (X_c ** 2).T - 2 * C_scm * (X_c @ X_c.T)
    var_R += n_times * C_scm ** 2
    Xvar = np.outer(X.var(axis=1), X.var(axis=1))
    var_R *= n_times / ((n_times - 1) ** 3 * Xvar)
    R -= np.diag(np.diag(R))
    var_R -= np.diag(np.diag(var_R))
    gamma = max(0, min(1, var_R.sum() / (R ** 2).sum()))

    sigma = (1. - gamma) * (n_times / (n_times - 1.)) * C_scm
    shrinkage = gamma * (n_times / (n_times - 1.)) * np.diag(np.diag(C_scm))
    return sigma + shrinkage


def covariances(X, estimator='cov'):
    """Estimation of covariance matrix.
    Parameters
    ----------
    X : ndarray, shape (n_trials, n_channels, n_times)
        ndarray of trials.
    estimator : {'cov', 'scm', 'lwf', 'oas', 'mcd', 'sch', 'corr'} (default: 'scm')
        covariance matrix estimator:
        * 'cov' for numpy based covariance matrix,
          https://numpy.org/doc/stable/reference/generated/numpy.cov.html
        * 'scm' for sample covariance matrix,
          https://scikit-learn.org/stable/modules/generated/sklearn.covariance.empirical_covariance.html
        * 'lwf' for shrunk Ledoit-Wolf covariance matrix
          https://scikit-learn.org/stable/modules/generated/sklearn.covariance.ledoit_wolf.html
        * 'oas' for oracle approximating shrunk covariance matrix,
          https://scikit-learn.org/stable/modules/generated/sklearn.covariance.OAS.html
        * 'mcd' for minimum covariance determinant matrix,
          https://scikit-learn.org/stable/modules/generated/sklearn.covariance.MinCovDet.html
        * 'sch' for Schaefer-Strimmer covariance,
          http://doi.org/10.2202/1544-6115.1175,
        * 'corr' for correlation coefficient matrix,
          https://numpy.org/doc/stable/reference/generated/numpy.corrcoef.html
    Returns
    -------
    covmats : ndarray, shape (n_trials, n_channels, n_channels)
        ndarray of covariance matrices.
    References
    ----------
    .. [1] https://scikit-learn.org/stable/modules/covariance.html
    """  # noqa
    est = _check_est(estimator)
    n_trials, n_channels, n_times = X.shape
    covmats = np.empty((n_trials, n_channels, n_channels))
    for i in range(n_trials):
        covmats[i, :, :] = est(X[i, :, :])
    return covmats

def _check_est(est):
    """Check if a given estimator is valid"""

    # Check estimator exist and return the correct function
    estimators = {
        'cov': np.cov,
        'scm': _scm,
        'lwf': _lwf,
        'oas': _oas,
        'mcd': _mcd,
        'sch': _sch,
        'corr': np.corrcoef
    }

    if callable(est):
        # All good (cross your fingers)
        pass
    elif est in estimators.keys():
        # Map the corresponding estimator
        est = estimators[est]
    else:
        # raise an error
        raise ValueError(
            """%s is not an valid estimator ! Valid estimators are : %s or a
             callable function""" % (est, (' , ').join(estimators.keys())))
    return est

class BlockCovariances(BaseEstimator, TransformerMixin):
    """Estimation of covariance matrix.
    Perform a simple covariance matrix estimation for each given trial.
    Parameters
    ----------
    estimator : string (default: 'scm')
        covariance matrix estimator. For regularization consider 'lwf' or 'oas'
        For a complete list of estimator, see `utils.covariance`.
    See Also
    --------
    ERPCovariances
    XdawnCovariances
    CospCovariances
    HankelCovariances
    """

    def __init__(self, n_blocks, estimator='scm'):
        """Init."""
        self.estimator = estimator
        self.n_blocks = n_blocks

    def fit(self, X, y=None):
        """Fit.
        Do nothing. For compatibility purpose.
        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_samples)
            ndarray of trials.
        y : ndarray shape (n_trials,)
            labels corresponding to each trial, not used.
        Returns
        -------
        self : Covariances instance
            The Covariances instance.
        """
        return self

    def transform(self, X):
        """Estimate covariance matrices.
        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_samples)
            ndarray of trials.
        Returns
        -------
        covmats : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of covariance matrices for each trials.
        """

        self.block_size = X.shape[1]//self.n_blocks
        est = _check_est(self.estimator)
        n_trials, n_channels, n_times = X.shape

        covmats = np.empty((n_trials, n_channels, n_channels))
        blockcov = np.empty((self.n_blocks, self.block_size, self.block_size))
        for i in range(n_trials):
            for j in range(self.n_blocks):
                blockcov[j] = est(X[i, j*self.block_size:(j+1)*self.block_size, :])
            covmats[i, :, :] = block_diag(*tuple(blockcov))

        return covmats


        # for i in range(n_blocks):
        #     covmats[i] = covariances(X[:, i*self.block_size:(i+1)*self.block_size, :], estimator=self.estimator)
        #
        # #diag_idx = block_diag(*tuple([numpy.ones((self.block_size, self.block_size)) for n in range(Nblocks)]))
        # #block_diags = numpy.array([cov*diag_idx for cov in covmats])
        #
        # return covmats

