from matplotlib import cm, rcParams
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib as matplotlib
from matplotlib import patheffects
import numpy as np
import math as math
import random as rand
import os, sys, csv
import pandas as pd

# matplotlib.pyplot.xkcd(scale=.5, length=100, randomness=2)
# rcParams['path.effects'] = [patheffects.withStroke(linewidth=.5)]

def lif_euler(t, dt, v1, v2, I1, I2):
    if len(ts1) > 0 :
        syn2 = alpha*alpha*(t-ts1)*np.exp(-alpha*(t-ts1))
    else :
        syn2 = 0
    if len(ts2) > 0 :
        syn1 = alpha*alpha*(t-ts2)*np.exp(-alpha*(t-ts2))
    else :
        syn1 = 0
    res = [v1 + dt*(-v1 + gc*(v2-v1) + I1) + -gs*dt*np.sum(syn1), v2 + dt*(-v2 + gc*(v1-v2) + I2) + -gs*dt*np.sum(syn2)]
    print(t, res[0], res[1])
    return res

gc, gs, beta, alpha = 0., 0.1, 0.2, 4
I1, I2 = 1.6, 1.6
Vth, Vr = 1, 0

dt = 0.001
phis1, phis2 = [], []

maxtime = 200

v1_0, v2_0 = 0.5, 0
x1, x2 = [v1_0], [v2_0]
ts1, ts2 = np.array([]), np.array([]) #time spike 1, time spike 2
t = [0]

while t[-1] < maxtime :
    t.append(t[-1]+dt)
    next_values= lif_euler(t[-1], dt, x1[-1], x2[-1], I1, I2) # example of common input

    if (next_values[0] > 1 and (next_values[1] > 1 or next_values[1]+gc*beta > 1)) or (next_values[1] > 1 and (next_values[0] > 1 or next_values[0]+gc*beta > 1)) :
        x2.append(0)
        x1.append(0)
        ts2 = np.append(ts2, t[-1])
        ts1 = np.append(ts1, t[-1])

    elif next_values[1] > 1 :
        x2.append(0)
        ts2 = np.append(ts2, t[-1])
        if next_values[0] + gc*beta > 1 :
            x1.append(0)
        else :
            x1.append(next_values[0]+gc*beta)

    elif next_values[0] > 1 :
        x1.append(0)
        ts1 = np.append(ts1, t[-1])
        if next_values[1] + gc*beta > 1 :
            x2.append(0)
        else :
            x2.append(next_values[1]+gc*beta)

    else :
        x2.append(next_values[1])
        x1.append(next_values[0])

# A spike occurs iff there was a reset
for i in range(1,len(x1)) :
    if abs(x1[i]-x1[i-1]) > (Vth-Vr)/2 and x1[i] < 1 and x1[i-1] < 1:
        x1.insert(i, Vth+0.5)
        x2.insert(i, x2[i])
        t.insert(i, t[i])

for i in range(1,len(x2)) :
    if abs(x2[i]-x2[i-1]) > (Vth-Vr)/2 and x2[i] < 1 and x2[i-1] < 1:
        x2.insert(i, Vth+0.5)
        x1.insert(i, x1[i])
        t.insert(i, t[i])

plt.figure(figsize=(12,3.5))

plt.plot(t, x1, label='$V_{1}$', color='#aa3863')
plt.plot(t, x2, label='$V_{2}$', color='#3b7d86')
plt.xlim(100, 200)

plt.legend(loc='upper right', fontsize=10.5)
plt.xlabel('Time ($10^{-2}$ s)', fontsize=11)
plt.ylabel('Voltage $V_k, k \in \{1,2}$', fontsize=11)
plt.title(f'Example of electrically & chemically coupled neurons, $I={I1}, \gamma={gc}, \\beta={beta}, \kappa={gs}, \\alpha={alpha}$', pad=15, size=14)

plt.tight_layout()
plt.savefig('example_elec_chem.svg')
plt.show()
