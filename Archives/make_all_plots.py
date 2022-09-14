
import matplotlib.pyplot as plt
import numpy as np

ret = np.loadtxt("TabularQLearning_smalldelta/training_returns")

returns = np.array([np.mean(ret[i:i+1000]) for i in range(ret.size-1000)])
stddev = np.array([np.std(ret[i:i+50]) for i in range(ret.size - 1000)])

returns_l = np.loadtxt("TabularLambdaQLearning_smalldelta/training_returns")
returns_lambda = np.array([np.mean(returns_l[i:i+1000]) for i in range(returns_l.size-1000)])
stddev_lambda = np.array([np.std(returns_l[i:i+50]) for i in range(returns_l.size - 1000)])



plt.fill_between(np.arange(returns.size), returns, y2=returns+stddev, alpha=0.1, color="blue")
plt.fill_between(np.arange(returns.size), returns, y2=returns-stddev, alpha=0.1, color="blue")
plt.plot(np.arange(returns.size), returns, color="blue", label="Double Q Learning")

plt.fill_between(np.arange(returns_lambda.size), returns_lambda, y2=returns_lambda+stddev_lambda, alpha=0.1, color="red")
plt.fill_between(np.arange(returns_lambda.size), returns_lambda, y2=returns_lambda-stddev_lambda, alpha=0.1, color="red")
plt.plot(np.arange(returns_lambda.size), returns_lambda, color="red", label="Double Q Learning with Eligibility")
plt.legend()
plt.title("Environment Rewards")
plt.xlabel("Training Episode")
plt.ylabel("Episode Return")
plt.show()


ret = np.loadtxt("TabularQLearning_smalldelta/training_returns")

returns = np.array([np.mean(ret[i:i+1000]) for i in range(ret.size-1000)])
stddev = np.array([np.std(ret[i:i+50]) for i in range(ret.size - 1000)])

returns_l = np.loadtxt("TabularQLearning_largedelta/training_returns")
returns_lambda = np.array([np.mean(returns_l[i:i+1000]) for i in range(returns_l.size-1000)])
stddev_lambda = np.array([np.std(returns_l[i:i+50]) for i in range(returns_l.size - 1000)])



plt.fill_between(np.arange(returns.size), returns, y2=returns+stddev, alpha=0.1, color="blue")
plt.fill_between(np.arange(returns.size), returns, y2=returns-stddev, alpha=0.1, color="blue")
plt.plot(np.arange(returns.size), returns, color="blue", label="N=10")

plt.fill_between(np.arange(returns_lambda.size), returns_lambda, y2=returns_lambda+stddev_lambda, alpha=0.1, color="red")
plt.fill_between(np.arange(returns_lambda.size), returns_lambda, y2=returns_lambda-stddev_lambda, alpha=0.1, color="red")
plt.plot(np.arange(returns_lambda.size), returns_lambda, color="red", label="N=5")
plt.legend()
plt.title("Environment Rewards")
plt.xlabel("Training Episode")
plt.ylabel("Episode Return")
plt.show()



ret = np.loadtxt("TabularLambdaQLearning_smalldelta/training_returns")
returns = np.array([np.mean(ret[i:i+1000]) for i in range(ret.size-1000)])
stddev = np.array([np.std(ret[i:i+50]) for i in range(ret.size - 1000)])

returns_l = np.loadtxt("TabularLambdaQLearning_largedelta/training_returns")
returns_lambda = np.array([np.mean(returns_l[i:i+1000]) for i in range(returns_l.size-1000)])
stddev_lambda = np.array([np.std(returns_l[i:i+50]) for i in range(returns_l.size - 1000)])



plt.fill_between(np.arange(returns.size), returns, y2=returns+stddev, alpha=0.1, color="blue")
plt.fill_between(np.arange(returns.size), returns, y2=returns-stddev, alpha=0.1, color="blue")
plt.plot(np.arange(returns.size), returns, color="blue", label="N=10")

plt.fill_between(np.arange(returns_lambda.size), returns_lambda, y2=returns_lambda+stddev_lambda, alpha=0.1, color="red")
plt.fill_between(np.arange(returns_lambda.size), returns_lambda, y2=returns_lambda-stddev_lambda, alpha=0.1, color="red")
plt.plot(np.arange(returns_lambda.size), returns_lambda, color="red", label="N=5")
plt.legend()
plt.title("Environment Rewards")
plt.xlabel("Training Episode")
plt.ylabel("Episode Return")
plt.show()

