#!/usr/bin/env python3
"""
module
"""
import gym
import numpy as np


def q_init(env):
    """ q_init """
    out = np.zeros((env.observation_space.n, env.action_space.n))
    return out