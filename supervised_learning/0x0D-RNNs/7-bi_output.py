#!/usr/bin/env python3
"""
    module
"""
import numpy as np


class BidirectionalCell:
    """ bi directional """
    def __init__(self, i, h, o):
        """ init """
        pass

    def forward(self, h_prev, x_t):
        """ forward """
        return None, None
    
    def backward(self, h_next, x_t):
        """ backward """
        return None
    
    def output(self, H):
        """ output """
        return None