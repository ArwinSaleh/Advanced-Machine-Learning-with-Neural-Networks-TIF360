import itertools
from os import stat
import numpy as np
import itertools
from numpy.core.fromnumeric import argmax
import pandas as pd


o = np.zeros((1, 5))
p = np.ones((1, 5))

a = np.array([o, p])


states = list(itertools.product(range(0,5), repeat=4))

states.append(itertools.product(range(1,5)))

#print(np.array(states))

lst=[]
arrays = []

for i in range(4):
    arrays.append(range(0, 5))
arrays.append(range(0, 4))

for i in itertools.product(*arrays):
         lst.append(i)

actions = []
action_perm = []
action_perm.append(range(0, 4))
action_perm.append(range(0, 4))
action_perm.append(range(0, 4))


for i in itertools.product(*action_perm):
    actions.append(i)

rewards = pd.read_csv('Q_table_episode_10000.csv')

a = np.random.randint(0, 5, size=(4, 4))

n, m = 4, 4


states = []

x = itertools.product([1, -1], repeat=n * m)
x = np.reshape(list(x), (-1, n, m))
y = itertools.product([range(0, 4)], repeat=1 * m)
y = np.reshape(list(y), (-1, n, 1))
# print(x[0,:,:-1])


print(x)
print(y)
print(len(x))
print(len(npy))
print(np.hstack((x, y)))