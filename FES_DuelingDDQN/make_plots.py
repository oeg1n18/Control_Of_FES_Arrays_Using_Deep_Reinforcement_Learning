
import numpy as np
import matplotlib.pyplot as plt

returns = np.loadtxt("training_returns")

plt.plot(returns)
plt.show()