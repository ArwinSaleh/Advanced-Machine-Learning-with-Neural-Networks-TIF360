from datetime import time
from matplotlib import pyplot as plt
import numpy as np
from numpy.core.fromnumeric import size
import pandas as pd
import operator
from sys import platform

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures


def task1a():

    if platform == "linux" or platform == "linux2":
        rewards = pd.read_csv("Problem B/Task 1/1a.csv").to_numpy()
    else:
        rewards = pd.read_csv("Problem B\\Task 1\\1a.csv").to_numpy()

    n_episodes = len(rewards)
    episodes = np.linspace(0,n_episodes, n_episodes)

    plt.plot(episodes, rewards, label="Rewards")

    rewards = np.array(rewards).reshape((len(rewards)))

    episodes = episodes[:, np.newaxis]
    rewards = rewards[:, np.newaxis]

    poly_feat = PolynomialFeatures(degree=6)
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

def task1b():

    if platform == "linux" or platform == "linux2":
        rewards = pd.read_csv("Problem B/Task 1/1b.csv").to_numpy()
    else:
        rewards = pd.read_csv("Problem B\\Task 1\\1b.csv").to_numpy()

    n_episodes = len(rewards)
    episodes = np.linspace(0,n_episodes, n_episodes)

    plt.plot(episodes, rewards, label="Rewards")

    rewards = np.array(rewards).reshape((len(rewards)))

    episodes = episodes[:, np.newaxis]
    rewards = rewards[:, np.newaxis]

    poly_feat = PolynomialFeatures(degree=4)
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

def task1c():

    if platform == "linux" or platform == "linux2":
        rewards = pd.read_csv("Problem B/Task 1/1c.csv").to_numpy()
    else:
        rewards = pd.read_csv("Problem B\\Task 1\\1c.csv").to_numpy()

    n_episodes = len(rewards)
    episodes = np.linspace(0,n_episodes, n_episodes)

    plt.plot(episodes, rewards, label="Rewards")

    rewards = np.array(rewards).reshape((len(rewards)))

    episodes = episodes[:, np.newaxis]
    rewards = rewards[:, np.newaxis]

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

#task1a()
#task1b()
task1c()
