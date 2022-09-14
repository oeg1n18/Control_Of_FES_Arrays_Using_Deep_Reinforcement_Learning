import sys
sys.path.insert(0, "/mainfs/lyceum/oeg1n18/RL_FES_ARRAYS/rl_fes_array_control")
from rl_algs.Networks.QNet import CreateQNetwork
from rl_algs.replay_buffers.Trajectory import Trajectory
from rl_algs.agents.TabularQAgent import DoubleTabularQLearningAgent
from rl_algs.replay_buffers.UniformReplayMemory import ReplayMemory
from rl_algs.drivers.value_driver import TabularQDriver
from rl_algs.replay_buffers.Utils import get_data_spec
from rl_algs.observers.metrics import AverageReturnObserver
from rl_algs.observers.metrics import AverageEpisodeLengthObserver
from environments.hand_env import hand_environment
import gym
import numpy as np
import time
import matplotlib.pyplot as plt
np.random.seed(10)

N_TRAIN_STEPS = 5000

env = hand_environment()

agent = DoubleTabularQLearningAgent(env, resolution=10, eps_dec = 0.9/N_TRAIN_STEPS, target_copy_freq=150, lr=0.01)

observers = [AverageReturnObserver(), AverageEpisodeLengthObserver()]

returns = []
max_return = 0
for train_step in range(N_TRAIN_STEPS):
    agent, observers = TabularQDriver(env, agent, 1, observers=observers)
    eps_return = observers[0].result()
    returns.append(eps_return)
    if max_return < eps_return:
        agent.save("saved_agents")
    if train_step % 100 == 0:
        np.savetxt("training_returns", np.array(returns))
    print("Train Step: ", train_step, " AverageReturn: ", observers[0].result(), " AverageLength: ", observers[1].result(), " epsilon: ", agent.epsilon)
