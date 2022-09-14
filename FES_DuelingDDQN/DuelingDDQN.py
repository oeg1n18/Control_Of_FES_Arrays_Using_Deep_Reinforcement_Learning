import sys

sys.path.insert(0, "/mainfs/lyceum/oeg1n18/RL_FES_ARRAYS/rl_fes_array_control")
sys.path.insert(0, "/Users/olivergrainge/UniOnedrive/Uni_year_three/rl_fes_array_control")
from rl_algs.Networks.QNet import CreateQNetwork
from rl_algs.replay_buffers.Trajectory import Trajectory
from rl_algs.agents.DuelingDDQNAgent import DuelingDDQNAgent
from rl_algs.replay_buffers.UniformReplayMemory import ReplayMemory
from rl_algs.drivers.value_driver import driver
from rl_algs.replay_buffers.Utils import get_data_spec
from rl_algs.observers.metrics import AverageReturnObserver
from rl_algs.observers.metrics import AverageEpisodeLengthObserver
from environments.hand_env import hand_environment
import gym
import numpy as np
import time


env = hand_environment()

agent = DuelingDDQNAgent(env, lr=0.001, epsilon=1.0, batch_size=64, epsilon_dec=1/14000, replace=25, df=0.99)

observers = [AverageReturnObserver()]
returns = []

for episode in range(15000):
    agent.replay_buffer, observers = driver(env, agent.policy, agent.replay_buffer, 1, observers=observers)
    experience = agent.replay_buffer.sample_experience()
    agent.train(experience)
    print("Step: ", episode, " epsilon: ", agent.epsilon, " AverageReturn: ", observers[0].result())
    returns.append(observers[0].result())
    np.savetxt("training_returns", np.array(returns))