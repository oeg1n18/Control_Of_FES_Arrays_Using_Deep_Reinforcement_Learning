

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from Networks.ActorCriticNet import ActorCriticNetwork
from agents.ActorCriticAgent import ActorCriticAgent
from environments.hand_env import hand_environment
from replay_buffers.Trajectory import Trajectory

returns = np.loadtxt("AC_training_returns")
returns = returns
varience = np.array([np.std(returns[i:i+10]) for i in range(returns.size)])
returns = np.array([np.mean(returns[i:i+10]) for i in range(returns.size)])


#sns.lineplot(x="episode", y="return", data=smooth(returns))
plt.fill_between(np.arange(returns.size), returns, y2=returns+varience, alpha=0.2, color="blue")
plt.fill_between(np.arange(returns.size), returns, y2=returns-varience, alpha=0.2, color="blue")
plt.plot(np.arange(returns.size), returns, color="blue")
plt.title("Actor Critic RL Algorithm")
plt.xlabel("Training Episode")
plt.ylabel("Episode Return")
plt.show()


env = hand_environment()

actor_critic = ActorCriticNetwork(env.action_space.n)

agent = ActorCriticAgent(actor_critic, learning_rate=1e-5)

agent.load("saved_agents/AC")

done = False
states = []
state = env.reset()
traj = Trajectory(state, None, None, None, False)
while not done:
    action = agent.policy(traj)
    next_state, reward, done, _ = env.step(action)
    states.append(state)
    state = next_state


fig, ax = plt.subplots()
labels = ["angle1", "angle2", "angle3"]
target_labels = ["target_angle1", "target_angle2", "target_angle3"]
colors = ["red", "green", "blue"]
x = np.arange(251)*0.002
inputs, _ = env.generate_inputs(states[-1])
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


