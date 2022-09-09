import sys
sys.path.insert(0, "/mainfs/lyceum/oeg1n18/RL_FES_ARRAYS/rl_fes_array_control")
from rl_algs.Networks.QNet import CreateQNetwork
from rl_algs.replay_buffers.Trajectory import Trajectory
from rl_algs.agents.DDQNAgent import DDQNAgent
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

qnet = CreateQNetwork(env.observation_space, env.action_space, (512, 256), learning_rate=0.005)

data_spec = get_data_spec(env)

replay_memory = ReplayMemory(data_spec, 128, 1000)

agent = DDQNAgent(env.observation_space, env.action_space, qnet)


def random_policy(traj):
    return np.random.randint(env.action_space.n)


replay_memory = driver(env, random_policy, replay_memory, 10)


all_observers = [AverageReturnObserver(), AverageEpisodeLengthObserver()]

returns = []
for step in range(20000):
    agent.epsilon = np.exp(-0.0002*step)
    replay_memory, all_observers = driver(env, agent.collect_policy, replay_memory, 1, observers=all_observers)
    experiences = replay_memory.sample_experience()
    agent.train(experiences)
    if step % 10 == 0:
        print("Average Return: ", all_observers[0].result(), " Average Steps: ", all_observers[1].result(), " Epsilon: ", agent.epsilon)
        returns.append(all_observers[0].result())
        np.savetxt("training_returns", np.array(returns))

np.savetxt("training_returns", np.array(returns))


agent.save("saved_agents/DQN")
