from environments.bandit_environment import hand_environment
from rl_algs.agents.Bandits.UCB_Bandit import UCB_Bandit

import matplotlib.pyplot as plt
import numpy as np

N_EPISODES = 1500

env = hand_environment(stim_levels=5)
_ = env.reset()
print(env.action_space)

agent = UCB_Bandit(env)

rewards = []

for action_step in range(N_EPISODES):
    print("Action step: ", action_step)
    action = agent.get_action()
    print(action)
    _, reward, _, _ = env.step(action)
    agent.update(action, reward)
    rewards.append(reward)


np.save("training_rewards", np.array(rewards))
rewards = np.array([np.mean(rewards[i:i+50]) for i in range(len(rewards)-50)])
plt.plot(rewards)
np.save("best_state", env.state_space[np.argmax(agent.reward_space)])
plt.show()



#env = hand_environment(stim_levels=5)
#_ = env.reset()
#print(env.action_space)

#agent = greedy_bandit(env, epsilon_decay=0.7/N_EPISODES)

#rewards = []

#for action_step in range(N_EPISODES):
#    print("Action step: ", action_step)
#    action = agent.get_action()
#    _, reward, _, _ = env.step(action)
#    agent.update(action, reward)
#    rewards.append(reward)


#np.save("training_rewards_large", np.array(rewards))
#rewards = np.array([np.mean(rewards[i:i+50]) for i in range(len(rewards)-50)])
#np.save("best_state_large", env.state_space[np.argmax(agent.reward_space)])

