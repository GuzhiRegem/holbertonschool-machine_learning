#!/usr/bin/env python3
"""
module
"""
import gym
import numpy as np


def play(env, Q, max_steps=100):
    """ play """
    current_state = env.reset()
    done = False
    env.render()
    for step in range(max_steps):
        action = np.argmax(Q[current_state, :])
        next_state, reward, done, _ = env.step(action)
        env.render()
        if done:
            break
        current_state = next_state
    return reward
