import matplotlib.pyplot as plt
import numpy as np
from rl_algs.agents.TabularQAgent import DoubleTabularQLearningAgent
from environments.hand_env import hand_environment



returns = np.loadtxt("training_returns")
returns = np.array([np.mean(returns[i:i+50]) for i in range(returns.size-50)])
stddev = np.array([np.std(returns[i:i+50]) for i in range(returns.size)])

plt.fill_between(np.arange(returns.size), returns, y2=returns+stddev, alpha=0.2, color="blue")
plt.fill_between(np.arange(returns.size), returns, y2=returns-stddev, alpha=0.2, color="blue")
plt.plot(np.arange(returns.size), returns, color="blue")
plt.title("Actor Critic RL Algorithm")
plt.xlabel("Training Episode")
plt.ylabel("Episode Return")
plt.show()


env = hand_environment()

agent = DoubleTabularQLearningAgent(env, resolution=10, eps_dec = 0.95/1000, target_copy_freq=100, lr=0.01)
agent.load("saved_agents")

done = False
state = env.reset()
states = []
best_state = None
best_reward = 0
while not done:
    action = agent.policy(state, evaluate=True)
    next_state, reward, done, _ = env.step(action)
    if reward > best_reward:
        best_reward = reward
        best_state = state
    states.append(state)
    state = next_state


fig, ax = plt.subplots()
labels = ["angle1", "angle2", "angle3"]
target_labels = ["target_angle1", "target_angle2", "target_angle3"]
colors = ["red", "green", "blue"]
x = np.arange(251)*0.002
inputs, _ = env.generate_inputs(best_state)
outputs = env.generate_outputs(inputs)[0]
ax.plot(x, env.target_angles[0, :, 0], color=colors[0], label=target_labels[0])
ax.plot(x, env.target_angles[0, :, 1], color=colors[1], label=target_labels[1])
ax.plot(x, env.target_angles[0, :, 2], color=colors[2], label=target_labels[2])
ax.plot(x, outputs[:, 0], label=labels[0], color=colors[0], linestyle="dashed")
ax.plot(x, outputs[:, 1], label=labels[1], color=colors[1], linestyle="dashed")
ax.plot(x, outputs[:, 2], label=labels[2], color=colors[2], linestyle="dashed")
plt.legend()
print("hello")
plt.show()




