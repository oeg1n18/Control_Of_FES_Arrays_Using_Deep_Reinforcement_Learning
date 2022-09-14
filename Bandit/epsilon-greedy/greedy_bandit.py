import sys
sys.path.insert(0, "/Users/olivergrainge/UniOnedrive/Uni_year_three/rl_fes_array_control/environments")
from environments.bandit_environment import hand_environment
from rl_algs.agents.Bandits.greedy_bandit import greedy_bandit

import matplotlib.pyplot as plt
import numpy as np

N_EPISODES = 1500

env = hand_environment(stim_levels=5)
_ = env.reset()
print(env.action_space)

agent = greedy_bandit(env, epsilon_decay=0.8/N_EPISODES)

rewards = []

for action_step in range(N_EPISODES):
    print("Action step: ", action_step)
    agent.epsilon=np.exp(-0.0025*action_step)
    action = agent.get_action()
    _, reward, _, _ = env.step(action)
    agent.update(action, reward)
    rewards.append(reward)


np.save("training_rewards", np.array(rewards))
rewards = np.array([np.mean(rewards[i:i+50]) for i in range(len(rewards)-50)])
plt.plot(rewards)
np.save("best_state", env.state_space[np.argmax(agent.reward_space)])
plt.show()



