from matplotlib import cm, rcParams
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import math as math
import random as rand
import os
import csv

rcParams.update({'figure.autolayout': True})

# Button palette
c = ['#aa3863', '#d97020', '#ef9f07', '#449775', '#3b7d86']

times = []
V1, V2, V3, V4, V5 = [], [], [], [], []
I1, I2, I3, I4, I5 = [], [], [], [], []

Vth = 1
Vr = 0

fig, ax = plt.subplots(2, 1, figsize=(16,6), sharey='row')

with open('diff_I_wo_coupling.dat', newline='') as file:
    datareader = csv.reader(file, delimiter=' ')
    for row in datareader:
        if float(row[0]) <= 45 :
            times.append(float(row[0]))
            V1.append(float(row[1]))
            V2.append(float(row[2]))
            V3.append(float(row[3]))
            V4.append(float(row[4]))
            V5.append(float(row[5]))
            I1.append(float(row[6]))
            I2.append(float(row[7]))
            I3.append(float(row[8]))
            I4.append(float(row[9]))
            I5.append(float(row[10]))

ax[0].plot(times, I1, alpha=0.75, color=c[0], linestyle='-', label='$I_1$')
ax[0].plot(times, I2, alpha=0.75, color=c[1], linestyle='-', label='$I_2$')
ax[0].plot(times, I3, alpha=0.75, color=c[2], linestyle='-', label='$I_3$')
ax[0].plot(times, I4, alpha=0.75, color=c[3], linestyle='-', label='$I_4$')
ax[0].plot(times, I5, alpha=0.75, color=c[4], linestyle='-', label='$I_5$')

ax[1].plot(times, V1, alpha=0.75, color=c[0], linestyle='-', label='$V_1$')
ax[1].plot(times, V2, alpha=0.75, color=c[1], linestyle='-', label='$V_2$')
ax[1].plot(times, V3, alpha=0.75, color=c[2], linestyle='-', label='$V_3$')
ax[1].plot(times, V4, alpha=0.75, color=c[3], linestyle='-', label='$V_4$')
ax[1].plot(times, V5, alpha=0.75, color=c[4], linestyle='-', label='$V_5$')


# A spike occurs iff there was a reset
spike_times_V1 = [times[i] for i in range(1,len(V1)) if abs(V1[i]-V1[i-1]) > (Vth-Vr)/2]
spike_times_V2 = [times[i] for i in range(1,len(V2)) if abs(V2[i]-V2[i-1]) > (Vth-Vr)/2]
spike_times_V3 = [times[i] for i in range(1,len(V3)) if abs(V3[i]-V3[i-1]) > (Vth-Vr)/2]
spike_times_V4 = [times[i] for i in range(1,len(V4)) if abs(V4[i]-V4[i-1]) > (Vth-Vr)/2]
spike_times_V5 = [times[i] for i in range(1,len(V5)) if abs(V5[i]-V5[i-1]) > (Vth-Vr)/2]

for t in spike_times_V1:
    ax[1].plot([t, t], [Vth, Vth+0.5], alpha=0.75, color=c[0])
for t in spike_times_V2:
    ax[1].plot([t, t], [Vth, Vth+0.5], alpha=0.75, color=c[1])
for t in spike_times_V3:
    ax[1].plot([t, t], [Vth, Vth+0.5], alpha=0.75, color=c[2])
for t in spike_times_V4:
    ax[1].plot([t, t], [Vth, Vth+0.5], alpha=0.75, color=c[3])
for t in spike_times_V5:
    ax[1].plot([t, t], [Vth, Vth+0.5], alpha=0.75, color=c[4])

ax[1].set_xlabel('Time ($10^{-2}$ seconds)', size=11)
ax[1].set_ylabel('Voltage $V_k, k \in \{1,..,5\}$', size=11)

ax[0].set_ylabel('Current $I_k, k \in \{1,..,5\}$', size=11)
ax[0].set_ylim(.88)

fig.suptitle('Network of 5 electrically coupled neurons, $\\beta=0.1$ and $\gamma=0.1$', size=15)
ax[0].legend(bbox_to_anchor=(1.0525, 1))
ax[1].legend(bbox_to_anchor=(1, 1))

plt.savefig('5_neurons_different_I_wo_coupling.svg')
#plt.show()
