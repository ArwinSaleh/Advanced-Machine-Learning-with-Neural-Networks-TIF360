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

a = np.random.randint(0, 5, size=(4, 4))

n, m = 4, 4


states = []

x = itertools.product([1, -1], repeat=n * m)
for board in x:
    for i in range(4):
        #state = (board, i)
        #states.append(state)
        states.append((board, i))


test_board = np.random.randint(0, 2, size=(4,4))
test_board[test_board == 0] = -1

#print(states.index((test_board, 0)))

key = np.ones((4, 4))
key = np.ndarray.flatten(key)
key[key == 1] = -1
print(key)
key = tuple(map(tuple, [key]))[0]

print(key)
print(states[0])

print(states.index((key, 0)))

a = np.array([0, 0, -np.inf, -999, -np.inf])

print(np.max(a))