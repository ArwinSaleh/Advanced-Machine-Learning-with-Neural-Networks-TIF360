import numpy as np
import itertools
import torch


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

a = torch.randint(0, 3, size=(4,4))

print(a)
#print(a.max(1)[1].view(1, 1))
print(a.max(1)[1][0])
print(a)