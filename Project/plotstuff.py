import numpy as np
import random
import math
import pickle
import matplotlib.pyplot as plt
import os


Y = np.loadtxt('Y.csv')
print(Y.shape[0])
X = pickle.load(open('pkl_data' + os.sep + 'FAST.pkl', "rb"))
X = X[:,-2]
print(X.shape[0])


fig, axs = plt.subplots(1, 1, figsize=(15, 8), facecolor='w', edgecolor='k', sharey=False)
fig.subplots_adjust(hspace=0.25, wspace=0.2)

axs.plot(range(X.shape[0]), X)
axs.plot(range(Y.shape[0]), Y)
# axs.set_xlabel('Episode')
# axs.set_ylabel('Points')
# axs.set_xlim(0, len(rewards))
# axs.set_ylim(min(rewards), max(rewards)*1.0)
# axs.set_title('Task '+task)

plt.show()
