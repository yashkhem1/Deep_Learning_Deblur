import pickle
import sys


with open(sys.argv[1], 'rb') as f:
	loss = pickle.load(f)

# print(loss)

import matplotlib.pyplot as plt
import numpy as np

loss = np.array(loss)
loss = loss[:500]

idx = np.arange(len(loss))

plt.plot(idx, loss)
plt.savefig(sys.argv[2])