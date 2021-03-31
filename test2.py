import numpy as np
import itertools

a = np.ones((4, 4))

states = []

n, m = 2, 2
x = itertools.product([1, -1], repeat=n * m)
for board in x:
    for i in range(4):
        #state = (board, i)
        #states.append(state)
        states.append(board)

print(np.array(states))
print(states.index((1, 1, 1, 1)))