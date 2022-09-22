import numpy as np
import matplotlib.pyplot as plt
from rl_algs.agents.Bandits.greedy_bandit import greedy_bandit
from environments.bandit_environment import hand_environment
from  scipy.stats import norm

x = np.linspace(0, 10, 1000)
y1 = norm.pdf(x, 4, 0.3)
y2 = norm.pdf(x, 6, 0.3)
y3 = norm.pdf(x, 3, 3)

y5 = np.zeros(x.size)
plt.fill_between(x, y1, y2=y5, label="Action 1", alpha=0.6)
plt.fill_between(x, y2, y2=y5, label="Action 2", alpha=0.6)
plt.fill_between(x, y3, y2=y5, label="Action 3", alpha=0.6)
plt.title("Posterior Distributions")
plt.ylabel(r"$p(\mu \mid r, a)$")
plt.xlabel("Reward")
plt.legend()
plt.show()

x = np.arange(1500)
constant_epsilon = np.ones(1500)*0.3
lin_epsilon = []
exp_epsilon = []
eps = 1.0
for i in range(x.size):
    exp_epsilon.append(np.exp(-0.002*i))
    eps -= 1.0/x.size
    lin_epsilon.append(eps)

fit, ax_eps = plt.subplots()
ax_eps.plot(x, lin_epsilon, label="Linearly Decreasing Epsilon")
ax_eps.plot(x, exp_epsilon, label="Exponential Decreasing Epsilon")
ax_eps.plot(x, constant_epsilon, label="Constant Epsilon")
ax_eps.set_xlabel("Training Episode")
ax_eps.set_ylabel("Random Action Probability")
ax_eps.set_title("Epsilon-Greedy Strategies")
ax_eps.legend()
ax_eps.set_xlim(0, 1500)
plt.show()

raw_rewards = np.load("training_rewards.npy")
rewards = np.array([np.mean(raw_rewards[i:i+50]) for i in range(len(raw_rewards)-50)])
std = np.array([np.std(raw_rewards[i:i+50]) for i in range(len(raw_rewards)-50)])
plt.fill_between(np.arange(rewards.size), rewards, y2=rewards+std, alpha=0.1, color="red")
plt.fill_between(np.arange(rewards.size), rewards, y2=rewards-std, alpha=0.1, color="red")
plt.plot(np.arange(rewards.size), rewards, color="red", label="N=10")


plt.title("Exponentially Decreasing epsilon-Greedy Bandit")
plt.xlabel("Training Step")
plt.ylabel("reward")
plt.legend()
plt.show()

env = hand_environment(stim_levels=10)

agent = greedy_bandit(env)
x = np.arange(251)*0.002
labels = ["angle1", "angle2", "angle3"]
target_labels = ["target_angle1", "target_angle2", "target_angle3"]
colors = ["red", "green", "blue"]
best_state_small = np.load("best_state_small.npy")
fig, ax = plt.subplots()
inputs, _ = env.generate_inputs(np.array(best_state_small))
outputs = env.generate_outputs(inputs)[0]
ax.plot(x, env.target_angles[0, :, 0], color=colors[0], label=target_labels[0])
ax.plot(x, env.target_angles[0, :, 1], color=colors[1], label=target_labels[1])
ax.plot(x, env.target_angles[0, :, 2], color=colors[2], label=target_labels[2])
ax.plot(x, outputs[:, 0], label=labels[0], color=colors[0], linestyle="dashed")
ax.plot(x, outputs[:, 1], label=labels[1], color=colors[1], linestyle="dashed")
ax.plot(x, outputs[:, 2], label=labels[2], color=colors[2], linestyle="dashed")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Angle (rad)")
plt.title("Exponentially Decreasing epsilon-Greedy Bandit")
plt.legend()
print("hello")
plt.show()

