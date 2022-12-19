#!/usr/bin/env python3
"""
    module
"""
import sklearn.cluster


def kmeans(X, k):
    """ initialize """
    kmeans = sklearn.cluster.KMeans(k).fit(X)
    return kmeans.cluster_centers_, kmeans.labels_
