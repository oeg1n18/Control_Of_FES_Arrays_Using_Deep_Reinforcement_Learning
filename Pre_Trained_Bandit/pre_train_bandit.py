from environments.bandit_environment import hand_environment
from rl_algs.agents.Bandits.ThompsonBandit import ThompsonBandit

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

fig_all, (ax1, ax2) = plt.subplots(2)
fig_best, (ax3, ax4) = plt.subplots(1, 2)

env = hand_environment(stim_levels=5)
_ = env.reset()
print(env.action_space)

agent = ThompsonBandit(env, priori_mu=0.0001, priori_std=100, posterior_std=12.0)

for _ in range(2):
    for action in range(env.action_space.n):
        print("Action step: ", action)
        _, reward, _, _ = env.step(action)
        agent.update_priori(action, reward)

ax1 = agent.plot_all_distributions(ax1)
ax3 = agent.plot_best_distributions(ax3)


rewards = []
agent.update_posterior_std(8.0)
for action_step in range(25):
    action = agent.get_action()
    print(agent.reward_dists[action].t_0)
    _, reward, _, _ = env.step(action)
    agent.update_priori(action, reward)
    print("Action Step: ", action_step, " Action: ", action)
    rewards.append(reward)


ax2 = agent.plot_all_distributions(ax2)
ax4 = agent.plot_best_distributions(ax4)
np.save("training_rewards", np.array(rewards))
np.save("best_state", env.state_space[np.argmax([agent.reward_dists[i].mu_0 for i in range(env.action_space.n)])])


ax1.set_xlabel("Reward", fontsize=12)
ax1.set_ylabel("P(R | A)", fontsize=12)
ax1.set_title("All Pre-Trained Action Distributions", fontsize=14)

ax2.set_xlabel("Reward", fontsize=12)
ax2.set_ylabel("P(R | A)", fontsize=12)
ax2.set_title("All Fully-Trained Action Distributions (N=25)", fontsize=14)

ax3.set_xlabel("Reward", fontsize=12)
ax3.set_ylabel("P(R | A)", fontsize=12)
ax3.set_title("Top Pre-Trained Action Distributions", fontsize=14)
ax3.legend()

ax4.set_xlabel("Reward", fontsize=12)
ax4.set_ylabel("P(R | A)", fontsize=12)
ax4.set_title("Top Fully-Trained Action Distributions (N=25)", fontsize=14)
ax4.legend()

fig_all.tight_layout()
fig_best.tight_layout()

plt.show()