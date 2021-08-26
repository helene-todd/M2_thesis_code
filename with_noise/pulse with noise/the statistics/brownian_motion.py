from matplotlib import cm, rcParams
import matplotlib.pyplot as plt
import numpy as np
import math as math
import random as rand

np.random.seed()
T = 1
N = 500
dt = T/N
dW = np.zeros(N)
W = np.zeros(N)

dW[1] = math.sqrt(dt)*np.random.randn()
W[1] = dW[1]

for j in range(2, N):
    dW[j] = math.sqrt(dt)*np.random.randn()
    W[j] = W[j-1] + dW[j]

plt.plot(np.arange(0, T, dt), W)
plt.show()
