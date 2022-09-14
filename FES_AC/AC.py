import sys
import os
sys.path.insert(0, "/mainfs/lyceum/oeg1n18/RL_FES_ARRAYS/rl_fes_array_control")
from rl_algs.agents.ActorCriticAgent import ActorCriticAgent
from rl_algs.drivers.value_driver import ac_driver
from rl_algs.Networks.ActorCriticNet import ActorCriticNetwork
from rl_algs.observers.metrics import AverageReturnObserver
from environments.hand_env import hand_environment

import warnings
import numpy as np


np.random.seed(10)

env = hand_environment()

actor_critic = ActorCriticNetwork(env.action_space.n)

agent = ActorCriticAgent(actor_critic, learning_rate=5e-6)

observers = [AverageReturnObserver(buffer_size=30)]

returns = []
max_return = 0.0
for training_step in range(15000):
    agent, observers = ac_driver(env, agent, 1, observers=observers)
    print("Training Step: ", training_step, " Average Return: ", observers[0].result())
    if training_step % 10 == 0:
        returns.append(observers[0].result())
        np.savetxt("AC_training_returns", np.array(returns))
    if training_step % 100 == 0:
        if observers[0].result() > max_return:
            agent.save("saved_agents/AC")


np.savetxt("AC_training_returns", np.array(returns))
