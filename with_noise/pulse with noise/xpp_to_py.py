import matplotlib.pyplot as plt
from matplotlib import cm, rcParams
import numpy as np
import math as math
import random as rand
import os
import csv

rcParams.update({'figure.autolayout': True})

c = ['#aa3863', '#3b7d86', '#5443a3']
# blue, red, green, pruple

times = []
V1, V2 = [], []
I = []

Vth = 1
Vr = 0

fig, ax = plt.subplots(2, 1, figsize=(12,5), sharey='row', sharex='col')

with open('data.dat', newline='') as file:
    datareader = csv.reader(file, delimiter=' ')
    for row in datareader:
        if float(row[0]) >= 8 and float(row[0]) <= 20:
            times.append(float(row[0]))
            V1.append(float(row[1]))
            V2.append(float(row[2]))
            I.append(float(row[3]))


ax[0].plot(times, I, alpha=0.75, color=c[2], linestyle='-', label='$I$')
ax[1].plot(times, V1, alpha=0.75, color=c[0], linestyle='-', label='$V_1$')
ax[1].plot(times, V2, alpha=0.75, color=c[1], linestyle='-', label='$V_2$')

# A spike occurs iff there was a reset
spike_times_V1 = [times[i] for i in range(1,len(V1)) if abs(V1[i]-V1[i-1]) > (Vth-Vr)/2]
spike_times_V2 = [times[i] for i in range(1,len(V2)) if abs(V2[i]-V2[i-1]) > (Vth-Vr)/2]

for t in spike_times_V1:
    ax[1].plot([t, t], [Vth, Vth+0.5], alpha=0.75, color=c[0])

for t in spike_times_V2:
    ax[1].plot([t, t], [Vth, Vth+0.5], alpha=0.75, color=c[1])

ax[0].set_ylabel('Current $I$')
ax[1].set_ylabel('Voltage $V_k, k \in \{1,2\}$')
ax[1].set_xlabel('time ($10^{-4}$ seconds)')

ax[0].set_title('From A/S regime to A/S regime, $\gamma=0.1, \\beta=0.2$', size=14)
ax[0].legend(loc='upper right', fontsize=10)
ax[1].legend(loc='upper right', fontsize=10)

plt.savefig('pulse_variation.png', dpi=600)
plt.show()
