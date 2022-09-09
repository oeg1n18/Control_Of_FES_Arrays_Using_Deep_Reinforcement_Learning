import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import seaborn as sns

from Networks.ActorCriticNet import ActorCriticNetwork
from agents.ActorCriticAgent import ActorCriticAgent
from environments.hand_env import hand_environment
from replay_buffers.Trajectory import Trajectory


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

labels = ["angle1", "angle2", "angle3"]
target_labels = ["target_angle1", "target_angle2", "target_angle3"]
colors = ["red", "green", "blue"]


fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()
x = np.arange(251)*0.002
inputs, _ = env.generate_inputs(states[0])
outputs = env.generate_outputs(inputs)[0]
target_line1, = ax1.plot(x, env.target_angles[0, :, 0], color=colors[0], label=target_labels[0])
target_line2, = ax2.plot(x, env.target_angles[0, :, 1], color=colors[1], label=target_labels[1])
target_line3, = ax3.plot(x, env.target_angles[0, :, 2], color=colors[2], label=target_labels[2])
line1, = ax1.plot(x, outputs[:, 0], label=labels[0], linestyle="dashed", color=colors[0])
line2, = ax2.plot(x, outputs[:, 1], label=labels[1], linestyle="dashed", color=colors[1])
line3, = ax3.plot(x, outputs[:, 2], label=labels[2], linestyle="dashed", color=colors[2])


def animate1(state):
    inputs, _ = env.generate_inputs(states)
    outputs = env.generate_outputs(inputs)[0]
    line1.set_ydata(outputs[:, 0])
    return line1

def animate2(state):
    inputs, _ = env.generate_inputs(states)
    outputs = env.generate_outputs(inputs)[0]
    line2.set_ydata(outputs[:, 1])
    return line2

def animate3(state):
    inputs, _ = env.generate_inputs(states)
    outputs = env.generate_outputs(inputs)[0]
    line3.set_ydata(outputs[:, 2])
    return line3

ani = animation.FuncAnimation(fig1, animate1, interval=200, frames=states)

plt.show()

