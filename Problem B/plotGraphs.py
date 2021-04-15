from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import operator
from sys import platform
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def plot_task(TASK):

    if platform == "linux" or platform == "linux2":
        rewards = pd.read_csv("Problem B/Task 1/" + TASK + ".csv").to_numpy()
    else:
        rewards = pd.read_csv("Problem B\\Task 1\\" + TASK + ".csv").to_numpy()

    n_episodes = len(rewards)
    episodes = np.linspace(0,n_episodes, n_episodes)

    plt.plot(episodes, rewards, label="Rewards")
    
    box_pts = 1

    if TASK == "1a":
        box_pts = 25
    if TASK == "1b":
        box_pts = 50
    if TASK == "1c":
        box_pts = 100
    if TASK == "2a":
        box_pts = 50

    plt.plot(episodes, smooth(rewards.flatten(), box_pts=box_pts), label="Average")

    plt.xlabel('E', size=24)
    plt.ylabel('R', size=24)
    plt.legend()
    plt.show()

plot_task("2a")
