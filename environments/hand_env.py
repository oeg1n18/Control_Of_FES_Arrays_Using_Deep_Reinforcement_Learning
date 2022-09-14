
import numpy as np

import tensorflow as tf
import os
from tensorflow import keras
from gym.spaces import Discrete, Box
from gym import Env


class hand_env_base:
    def __init__(self):
        self.state_max = np.array([3.5, 1.0, 1.0, 7.0, 7.0], dtype=np.float32)
        print("Working Directory: ", os.getcwd())
        self.hand_model = self.hand_model = keras.models.load_model("model_cp.h5")
        self.target_inputs, _ = self.generate_inputs(np.array([0.45, 0.45, 0.80, 0.850], dtype=np.float32))
        self.target_angles = self.generate_outputs(self.target_inputs)

    def generate_inputs(self, input_state):
        state = (input_state + 1.0) / 2.0
        state = np.hstack((np.array([0.5]), state))
        state = np.multiply(state, self.state_max)
        inputs = np.zeros((1, 251, 7))
        if state[0] <= 0.01:
            done = True
            slope = np.arange(0, 1, 0.02 / 3.5)
        else:
            done = False
            slope = np.arange(0, 1, 0.02 / state[0])
        initial = np.zeros(np.arange(0, 0.5, 0.02).shape)
        n_slope = initial.size + slope.size
        if n_slope > 200:
            done = True
            slope = np.arange(0, 1, 0.02 / 3.5)
            n_slope = initial.size + slope.size
        else:
            done = False
        steady_state = np.ones(251 - n_slope)
        ramp = np.hstack((initial, slope, steady_state))

        if 7 > state[3] >= 0:
            inputs[0, :, int(state[3])] = ramp * state[1]
        if 7 > state[4] >= 0:
            inputs[0, :, int(state[4])] = ramp * state[2]
        return inputs[:, :251, :], done

    def generate_outputs(self, ins):
        return self.hand_model(ins)

    def take_action(self, input_state, action, delta=0.5):
        n_state = input_state.size
        old_input_state = input_state
        new_input_state = input_state
        for index, var in enumerate(input_state):
            if int(action) == int(index * 2):
                if int(action) >= 4:
                    var += 2.0/7.0
                else:
                    var += delta
            if int(action) == int(index * 2 + 1):
                if int(action) >= 5:
                    var -= 2.0/7.0
                else:
                    var -= delta
            new_input_state[index] = var
        if np.max(new_input_state) > 1.001 or np.min(new_input_state) < -1.0001:
            return old_input_state, True
        else:
            return new_input_state, False

    def generate_reward(self, angles):
        err = np.sum(np.abs(self.target_angles[0,-1:,:] - angles[0,-1:,:]))
        return np.float32(np.clip((10/((err)**1.2)), 0.0, 300))




class hand_environment(Env, hand_env_base):
    def __init__(self):
        hand_env_base.__init__(self)
        self.observation_space = Box(low=np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32), high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32), shape=(4,))
        self.action_space = Discrete(8)
        self.counter = 0
        self.step_limit = 20

    def step(self, action):
        assert action < 8, "Action must be smaller than 8"
        assert action >= 0, "Action must be larger than 0"
        self.counter += 1
        self._state, done = self.take_action(self._state, action)
        if done:
            self.counter = 0
            return np.copy(self._state), np.float32(0.0), done, {}
        inputs, done = self.generate_inputs(self._state)
        if self.counter >= self.step_limit:
            self.counter = 0
            return np.copy(self._state), np.float32(0.0), True, {}
        angles = self.generate_outputs(inputs)
        reward = self.generate_reward(angles)
        self._state = np.nan_to_num(self._state, neginf=0.0, posinf=0.0)
        reward = np.nan_to_num(reward, neginf=0.0, posinf=0.0)
        return np.copy(self._state), np.float32(reward), done, {}

    def reset(self):
        self._state = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.counter = 0
        self._episode_ended = False
        return np.copy(self._state)

    def render(self):
        print("does not render yet")
        return 0

