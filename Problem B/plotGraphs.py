from matplotlib import pyplot as plt
import numpy as np
from numpy.core.fromnumeric import size
import pandas as pd
from scipy.signal import lfilter


def task1a():

    rewards = pd.read_csv("Problem B\\Task 1\\1a.csv").to_numpy()
    n_episodes = len(rewards)
    episodes = np.linspace(0,n_episodes, n_episodes)

    plt.plot(episodes, rewards)

    plt.xlabel('E', size=24)
    plt.ylabel('R', size=24)
    plt.show()

def task1b():

    rewards = pd.read_csv("Problem B\\Task 1\\1b.csv").to_numpy()
    n_episodes = len(rewards)
    episodes = np.linspace(0,n_episodes, n_episodes)

    plt.plot(episodes, rewards)
    
    plt.xlabel('E', size=24)
    plt.ylabel('R', size=24)
    plt.show()

def task1c():

    rewards = pd.read_csv("Problem B\\Task 1\\1c.csv")
    n_episodes = len(rewards)
    episodes = np.linspace(0,n_episodes, n_episodes)

    plt.plot(episodes, rewards)
    plt.xlabel('E', size=24)
    plt.ylabel('R', size=24)
    plt.show()

def task1d():

    rewards = pd.read_csv("Problem B\\Task 1\\1c.csv")
    n_episodes = len(rewards)
    episodes = np.linspace(0,n_episodes, n_episodes)

    plt.plot(episodes, rewards)
    plt.xlabel('E', size=24)
    plt.ylabel('R', size=24)
    plt.show()

task1a()
#task1b()
#task1c()
#task1d()