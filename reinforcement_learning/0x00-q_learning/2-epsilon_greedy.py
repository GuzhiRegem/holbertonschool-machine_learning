#!/usr/bin/env python3
"""
module
"""
import gym
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """ epsilon_greedy """
    p = np.random.uniform(0, 1)
    if p < epsilon:
        action = np.random.randint(Q.shape[1])
    else:
        action = np.argmax(Q[state, :])
    return action