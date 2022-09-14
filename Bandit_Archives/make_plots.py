import numpy as np
import matplotlib.pyplot as plt

VAR_RANGE = 50
MEAN_RANGE = 100

constant_epsilon = np.array(np.load("constant_epsilon_greedy/training_rewards.npy"))
exponential_epsilon = np.array(np.load("exponential_decreasing_epsilon/training_rewards.npy"))
linear_epsilon = np.array(np.load("linear_decreasing_epsilon/training_rewards.npy"))

thompson = np.array(np.load("thompson_sampling/training_rewards.npy"))
ucb = np.array(np.load("UCB_bandit/training_rewards.npy"))


fig, (ax1, ax2) = plt.subplots(1,2)

constant_epsilon_mu = np.array([np.mean(constant_epsilon[i:i+MEAN_RANGE]) for i in range(constant_epsilon.size - MEAN_RANGE)])
constant_epsilon_std = np.array([np.std(constant_epsilon[i:i+VAR_RANGE]) for i in range(constant_epsilon.size - MEAN_RANGE)])

exponential_epsilon_mu = np.array([np.mean(exponential_epsilon[i:i+MEAN_RANGE]) for i in range(exponential_epsilon.size - MEAN_RANGE)])
exponential_epsilon_std = np.array([np.std(exponential_epsilon[i:i+VAR_RANGE]) for i in range(exponential_epsilon.size - MEAN_RANGE)])

linear_epsilon_mu = np.array([np.std(linear_epsilon[i:i+MEAN_RANGE]) for i in range(linear_epsilon.size - MEAN_RANGE)])
linear_epsilon_std = np.array([np.std(linear_epsilon[i:i+VAR_RANGE]) for i in range(linear_epsilon.size - MEAN_RANGE)])

thompson_mu = np.array([np.mean(thompson[i:i+MEAN_RANGE]) for i in range(thompson.size - MEAN_RANGE)])
thompson_std = np.array([np.std(thompson[i:i+VAR_RANGE]) for i in range(thompson.size - MEAN_RANGE)])

ucb_mu = np.array([np.mean(ucb[i:i+MEAN_RANGE]) for i in range(ucb.size - MEAN_RANGE)])
ucb_std = np.array([np.std(ucb[i:i+VAR_RANGE]) for i in range(ucb.size - MEAN_RANGE)])



x = np.arange(constant_epsilon_mu.size)

#ax1.fill_between(x, constant_epsilon_mu, y2=constant_epsilon_mu + constant_epsilon_std, color="red", alpha=0.2)
#ax1.fill_between(x, constant_epsilon_mu, y2=constant_epsilon_mu - constant_epsilon_std, color="red", alpha=0.2)
ax1.plot(x, constant_epsilon_mu, label="Constant Epsilon")

#ax1.fill_between(x, exponential_epsilon_mu, y2=exponential_epsilon_mu + exponential_epsilon_std, color="blue", alpha=0.2)
#ax1.fill_between(x, exponential_epsilon_mu, y2=exponential_epsilon_mu - exponential_epsilon_std, color="blue", alpha=0.2)
ax1.plot(x, exponential_epsilon_mu, label="Exponential Epsilon")

#ax1.fill_between(x, linear_epsilon_mu, y2=linear_epsilon_mu + linear_epsilon_std, color="green", alpha=0.2)
#ax1.fill_between(x, linear_epsilon_mu, y2=linear_epsilon_mu - linear_epsilon_std, color="green", alpha=0.2)
ax1.plot(x, linear_epsilon_mu, label="Linear Epsilon")
ax1.legend()
ax1.set_xlabel("Training Round")
ax1.set_ylabel("Reward")
ax1.set_title("Greedy Bandits")

ax2.plot(x, thompson_mu, label="Thompson Sampling")
ax2.plot(x, ucb_mu, label="UCB Sampling")
ax2.set_xlabel("Training Round")
ax2.set_ylabel("Reward")
ax2.legend()
ax2.set_title("Advanced Bandits")

plt.show()
