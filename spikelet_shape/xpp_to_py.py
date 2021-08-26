from matplotlib import cm, rcParams
import matplotlib.pyplot as plt
import matplotlib as matplotlib
from matplotlib import patheffects
import numpy as np
import math as math
import random as rand
import os
import csv

rcParams.update({'figure.autolayout': True})
#matplotlib.pyplot.xkcd(scale=.5, length=100, randomness=2)
#rcParams['path.effects'] = [patheffects.withStroke(linewidth=.5)]

c = ['#aa3863', '#3b7d86', '#5443a3']

times = []
V1_plot1, V2_plot2, V2_plot3 = [], [], []


Vth = 1
Vr = 0

fig, ax = plt.subplots(1, 3, figsize=(12,3.5))

k = 0
with open('spikelet_beta=0.1.dat', newline='') as file:
    datareader = csv.reader(file, delimiter=' ')
    for row in datareader:
        if float(row[0]) >= 0 and float(row[0]) <= 20 :
            times.append(float(row[0]))
            V1_plot1.append(float(row[1]))
            V2_plot2.append(float(row[2]))

with open('spikelet_beta=0.2.dat', newline='') as file:
    datareader = csv.reader(file, delimiter=' ')
    for row in datareader:
        if float(row[0]) >= 0 and float(row[0]) <= 20 :
            V2_plot3.append(float(row[2]))

# A spike occurs iff there was a reset
for i in range(1,len(V1_plot1)) :
    if abs(V1_plot1[i]-V1_plot1[i-1]) > (Vth-Vr)/2 and V1_plot1[i] <= 1 and V1_plot1[i-1] <= 1:
        V1_plot1.insert(i, Vth+0.5)
        V2_plot2.insert(i, V2_plot2[i])
        V2_plot3.insert(i, V2_plot3[i])
        times.insert(i, times[i])

ax[0].plot(times, V1_plot1, alpha=1, color=c[0], linestyle='-', label='$V_1$') #alpha=0.75
ax[1].plot(times, V2_plot2, alpha=1, color=c[1], linestyle='-', label='$V_2$') #alpha=0.75
ax[2].plot(times, V2_plot3, alpha=1, color=c[1], linestyle='-', label='$V_2$') #alpha=0.75

ax[0].set_xlabel('Time ($10^{-2}$ seconds)', size=12)
ax[1].set_xlabel('Time ($10^{-2}$ seconds)', size=12)
ax[2].set_xlabel('Time ($10^{-2}$ seconds)', size=12)
ax[0].set_ylabel('Voltage $V_{k}, k \in \{1,2\}$', size=12)

ax[0].set_xlim(2.5, 10)
ax[0].set_ylim(-0.1, 1.6)
ax[1].set_xlim(2.5, 10)
ax[1].set_ylim(0.85, 0.95)
ax[2].set_xlim(2.5, 10)
ax[2].set_ylim(0.85, 0.95)

fig.suptitle('Dominantly inhibitory effects', size=16)
ax[0].set_title('Presynaptic spike', size=14)
ax[1].set_title('Post-synaptic effect, $\\beta=0.1$', size=14)
ax[2].set_title('Post-synaptic effect, $\\beta=0.2$', size=14)

ax[0].legend(loc='upper right', fontsize=12)
ax[1].legend(loc='upper right', fontsize=12)
ax[2].legend(loc='upper right', fontsize=12)

plt.tight_layout()
plt.savefig('spikelet_shape.svg')
plt.show()
