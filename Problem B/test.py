import itertools
from os import stat
import numpy as np
import itertools

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

print(np.array(actions))
print(len(actions))