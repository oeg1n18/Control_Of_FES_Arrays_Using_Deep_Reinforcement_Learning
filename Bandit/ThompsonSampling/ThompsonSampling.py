from environments.bandit_environment import hand_environment
from rl_algs.agents.Bandits.ThompsonBandit import ThompsonBandit

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

N_EPISODES = 1500

env = hand_environment(stim_levels=5)
_ = env.reset()
print(env.action_space)

agent = ThompsonBandit(env, priori_mu=0.0001, priori_std=100, posterior_std=1.0)

rewards = []

for action_step in range(N_EPISODES):
    print("Action step: ", action_step)
    action = agent.get_action()
    _, reward, _, _ = env.step(action)
    agent.update_priori(action, reward)
    rewards.append(reward)

agent.plot_best_distributions()
np.save("training_rewards", np.array(rewards))
rewards = np.array([np.mean(rewards[i:i+50]) for i in range(len(rewards)-50)])
plt.plot(rewards)
np.save("best_state", env.state_space[np.argmax([agent.reward_dists[i].mu_0 for i in range(env.action_space.n)])])
plt.show()

