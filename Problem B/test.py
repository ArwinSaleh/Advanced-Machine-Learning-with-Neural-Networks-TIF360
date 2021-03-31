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
x = np.reshape(list(x), (-1, n, m))
# print(x[0,:,:-1])

for board in x:
    for i in range(4):
        state = (board, i)
        states.append(state)


test_board = np.random.randint(0, 2, size=(4,4))
test_board[test_board == 0] = -1

#print(states.index((test_board, 0)))


print(np.where(state == test_board))
