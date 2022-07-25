#!/usr/bin/env python3
"""
    module
"""


def add_arrays(arr1, arr2):
    """ add arrays """
    if len(arr1) != len(arr2):
        return None
    out = []
    for x in range(len(arr1)):
        out.append(arr1[x] + arr2[x])
    return out
