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
arrays = [range(0,5), range(0,5), range(0, 5), range(0, 5), range(1,5)]  

for i in itertools.product(*arrays):
         lst.append(i)


heights = []
heights.append(0)
heights.append(0)
heights.append(0)
heights.append(0)
heights.append(1)


print(lst.index(heights))