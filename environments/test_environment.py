from hand_env import hand_environment
import matplotlib.pyplot as plt
import numpy as np

env = hand_environment()

done = False
state = env.reset()
state, reward, done, _ = env.step(0)
state, reward, done, _ = env.step(0)
state, reward, done, _ = env.step(0)
state, reward, done, _ = env.step(2)
state, reward, done, _ = env.step(2)
state, reward, done, _ = env.step(2)
print("State 1:", state)

print("REWARD 1: ", reward)

fig, ax = plt.subplots()
labels = ["angle1", "angle2", "angle3"]
target_labels = ["target_angle1", "target_angle2", "target_angle3"]
colors = ["red", "green", "blue"]
x = np.arange(251)*0.002
inputs, _ = env.generate_inputs(state)
outputs = env.generate_outputs(inputs)[0]
ax.plot(x, env.target_angles[0, :, 0], color=colors[0], label=target_labels[0])
ax.plot(x, env.target_angles[0, :, 1], color=colors[1], label=target_labels[1])
ax.plot(x, env.target_angles[0, :, 2], color=colors[2], label=target_labels[2])
ax.plot(x, outputs[:, 0], label=labels[0], color=colors[0], linestyle="dashed")
ax.plot(x, outputs[:, 1], label=labels[1], color=colors[1], linestyle="dashed")
ax.plot(x, outputs[:, 2], label=labels[2], color=colors[2], linestyle="dashed")
plt.title("First Print")
plt.legend()

state, reward, done, _ = env.step(6)
state, reward, done, _ = env.step(6)
state, reward, done, _ = env.step(6)
state, reward, done, _ = env.step(4)
state, reward, done, _ = env.step(4)
state, reward, done, _ = env.step(4)



fig, ax = plt.subplots()
labels = ["angle1", "angle2", "angle3"]
target_labels = ["target_angle1", "target_angle2", "target_angle3"]
colors = ["red", "green", "blue"]
x = np.arange(251)*0.002
inputs, _ = env.generate_inputs(state)
outputs = env.generate_outputs(inputs)[0]
ax.plot(x, env.target_angles[0, :, 0], color=colors[0], label=target_labels[0])
ax.plot(x, env.target_angles[0, :, 1], color=colors[1], label=target_labels[1])
ax.plot(x, env.target_angles[0, :, 2], color=colors[2], label=target_labels[2])
ax.plot(x, outputs[:, 0], label=labels[0], color=colors[0], linestyle="dashed")
ax.plot(x, outputs[:, 1], label=labels[1], color=colors[1], linestyle="dashed")
ax.plot(x, outputs[:, 2], label=labels[2], color=colors[2], linestyle="dashed")
plt.title("Second Print")
plt.legend()
print("Reward 2: ", reward)
print("State 2: ", state)
plt.show()


