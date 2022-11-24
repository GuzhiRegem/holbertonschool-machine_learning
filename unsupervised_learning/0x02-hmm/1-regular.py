#!/usr/bin/env python3
"""
    module
"""
import numpy as np


def regular(P):
    """ markov_chain regular """
    try:
        p = P - np.eye(P.shape[0])
        for ii in range(p.shape[0]):
            p[0,ii] = 1  
        P0 = np.zeros((p.shape[0],1))    
        P0[0] = 1
        return np.matmul(np.linalg.inv(p),P0)
    except Exception as e:
        return None
