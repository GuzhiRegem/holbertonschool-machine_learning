#!/usr/bin/env python3
"""
    module
"""
import sklearn.mixture


def gmm(X, k):
    """ initialize """
    gmm = sklearn.mixture.GaussianMixture(k).fit(X)
    labels = gmm.predict(X)
    return gmm.weights_, gmm.means_, gmm.covariances_, labels, gmm.bic(X)
