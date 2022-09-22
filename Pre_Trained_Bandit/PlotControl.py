from environments.bandit_environment import hand_environment
import numpy as np
import matplotlib.pyplot as plt



env = hand_environment()
best_state = np.array(np.load("best_state.npy"))
print(best_state)

fig, ax = plt.subplots()
labels = ["angle1", "angle2", "angle3"]
target_labels = ["target_angle1", "target_angle2", "target_angle3"]
colors = ["red", "green", "blue"]
x = np.arange(251)*0.002
inputs, _ = env.generate_inputs(np.array([0.0, 0.5, 0.3333333, 0.6666667]))
stim_levels = []
for i in range(7):
    label = "Channel " + str(i)
    if np.max(inputs[0, :, i]) > 0.01:
        stim_levels.append(np.max(inputs[0, :, i]*0.15))
    plt.plot(np.arange(251)*0.02, inputs[0, :, i]*0.15, label=label)
plt.title("Ramp Input", fontsize=16)
plt.legend()
plt.ylabel("Duty Cycle (%)", fontsize=12)
plt.xlabel("Time (s)", fontsize=12)
plt.show()


outputs = env.generate_outputs(inputs)[0]
ax.plot(x, env.target_angles[0, :, 0], color=colors[0], label=target_labels[0])
ax.plot(x, env.target_angles[0, :, 1], color=colors[1], label=target_labels[1])
ax.plot(x, env.target_angles[0, :, 2], color=colors[2], label=target_labels[2])
ax.plot(x, outputs[:, 0], label=labels[0], color=colors[0], linestyle="dashed")
ax.plot(x, outputs[:, 1], label=labels[1], color=colors[1], linestyle="dashed")
ax.plot(x, outputs[:, 2], label=labels[2], color=colors[2], linestyle="dashed")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Angle (rad)")
ax.set_title("Fully Trained Thompson Sampling Bayesian Bandit")
plt.legend()
print("hello")
plt.show()