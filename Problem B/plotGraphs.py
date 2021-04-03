from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import operator
from sys import platform
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def plot_task(TASK):

    if platform == "linux" or platform == "linux2":
        rewards = pd.read_csv("Problem B/Task 1/" + TASK + ".csv").to_numpy()
    else:
        rewards = pd.read_csv("Problem B\\Task 1\\" + TASK + ".csv").to_numpy()

    n_episodes = len(rewards)
    episodes = np.linspace(0,n_episodes, n_episodes)

    plt.plot(episodes, rewards, label="Rewards")

    rewards = np.array(rewards).reshape((len(rewards)))

    episodes = episodes[:, np.newaxis]
    rewards = rewards[:, np.newaxis]

    poly_feat = PolynomialFeatures(degree=1)
    
    if TASK == "1a":
        poly_feat = PolynomialFeatures(degree=6)
    if TASK == "1b":
        poly_feat = PolynomialFeatures(degree=4)
    if TASK == "1c":
        poly_feat = PolynomialFeatures(degree=3)
    if TASK == "2a":
        poly_feat = PolynomialFeatures(degree=3)
    
    ep_poly = poly_feat.fit_transform(episodes)

    model = LinearRegression()
    model.fit(ep_poly, rewards)

    reward_poly_pred = model.predict(ep_poly)

    # sort the values of x before line plot
    sort_axis = operator.itemgetter(0)
    sorted_zip = sorted(zip(episodes,reward_poly_pred), key=sort_axis)
    episodes, reward_poly_pred = zip(*sorted_zip)
    plt.plot(episodes, reward_poly_pred, color='r', label="Fit")

    plt.xlabel('E', size=24)
    plt.ylabel('R', size=24)
    plt.legend()
    plt.show()

#plot_task("1a")
#plot_task("1b")
plot_task("1c")

