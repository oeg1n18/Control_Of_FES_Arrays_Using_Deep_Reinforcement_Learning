import itertools
from abc import ABC

import numpy as np
import pandas as pd
import tensorflow as tf

import tf_agents
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tensorflow import keras
from tf_agents.environments import utils
from tf_agents.environments import suite_gym
import time
from data_pipeline import pipeline


class hand_env_base:
    def __init__(self):
        self._state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.hand_model = self.hand_model = keras.models.load_model("model_cp.h5")

    def take_action(self, action):
        old_state = np.copy(self._state)
        amplitudes = self._state + action
        inputs = np.zeros((1, 251, 7))
        slope = np.arange(0, 1, 0.02 / 3.5)
        initial = np.zeros(np.arange(0, 0.5, 0.02).shape)
        n_slope = initial.size + slope.size
        slope = np.arange(0, 1, 0.02 / 3.5)
        n_slope = initial.size + slope.size
        steady_state = np.ones(251 - n_slope)
        ramp = np.hstack((initial, slope, steady_state))
        for channel in range(7):
            if amplitudes[channel] > 1.0 or amplitudes[channel] < 0.0:
                return old_state, inputs, True
            inputs[:, :, channel] = ramp * amplitudes[channel]
        return amplitudes, inputs, False

    def generate_outputs(self, ins):
        return self.hand_model(ins)

    def generate_reward(self, angles):
        err = np.sum(np.abs(self.target_angles[0,-1:,:] - angles[0,-1:,:]))
        if err < 0.1:
            return 100.0
        elif err < 0.5:
            return 10.0
        elif err < 1.0:
            return 1.0
        else:
            return 0.0


class hand_env(tf_agents.environments.py_environment.PyEnvironment, hand_env_base):
    def __init__(self):
        super().__init__()
        hand_env_base.__init__(self)
        self._action_spec = array_spec.BoundedArraySpec(shape=(7,), dtype=np.float32, minimum=-0.2, maximum=0.2, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(shape=(7,), dtype=np.float32,
                                                             minimum=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                             maximum=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], name='observation')
        _, self.target_inputs, _ = self.take_action(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.8]))
        self._state = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        self.target_angles = self.generate_outputs(self.target_inputs)

        self.counter = 0
        self.step_limit = 10

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        self.counter = 0
        self._episode_ended = False
        return ts.restart(self._state)

    def _step(self, action):
        self.counter += 1
        self._state, inputs, done = self.take_action(action)
        if done:
            self.counter = 0
            return ts.termination(self._state, reward=np.float32(0.0))
        elif self.counter >= self.step_limit:
            self.counter = 0
            return ts.termination(self._state, reward=np.float32(0.0))
        angles = self.generate_outputs(inputs)
        reward = self.generate_reward(angles)
        return ts.transition(self._state, reward=np.float32(reward), discount=0.99)

