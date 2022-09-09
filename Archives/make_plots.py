
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

returns = np.loadtxt("AC_training_returns_3")
returns = returns[:-250]
varience = np.array([np.std(returns[i:i+40]) for i in range(returns.size)])
returns = np.array([np.mean(returns[i:i+40]) for i in range(returns.size)])




#sns.lineplot(x="episode", y="return", data=smooth(returns))
plt.fill_between(np.arange(returns.size), returns, y2=returns+varience, alpha=0.2, color="blue")
plt.fill_between(np.arange(returns.size), returns, y2=returns-varience, alpha=0.2, color="blue")
plt.plot(np.arange(returns.size), returns, color="blue")
plt.title("Actor Critic RL Algorithm")
plt.xlabel("Training Episode")
plt.ylabel("Episode Return")
plt.show()