from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

rewards = pd.read_csv('Rewards.csv')

print(rewards)

n_episodes = len(rewards)

episodes = np.linspace(0,n_episodes, n_episodes)

plt.plot(episodes, rewards)
plt.xlabel('E')
plt.ylabel('R')
plt.show()