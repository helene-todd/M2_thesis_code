import matplotlib.pyplot as plt
from matplotlib import cm, rcParams
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import math as math
import random as rand
import os
import csv

rcParams.update({'figure.autolayout': True})

# Button palette
c = ['#aa3863', '#d97020', '#ef9f07', '#449775', '#3b7d86']

times_plot1, times_plot2, times_plot3 = [], [], []
V1_plot1, V1_plot2, V1_plot3 = [], [], []
V2_plot1, V2_plot2, V2_plot3 = [], [], []
V3_plot1, V3_plot2, V3_plot3 = [], [], []
V4_plot1, V4_plot2, V4_plot3 = [], [], []
V5_plot1, V5_plot2, V5_plot3 = [], [], []
I_plot1, I_plot2, I_plot3 = [], [], []

Vth = 1
Vr = 0

fig, ax = plt.subplots(2, 1, figsize=(16,6), sharey='row')

with open('synchro.dat', newline='') as file:
    datareader = csv.reader(file, delimiter=' ')
    for row in datareader:
        if float(row[0]) >= 20 and float(row[0]) <= 52.5  :
            times_plot1.append(float(row[0]))
            V1_plot1.append(float(row[1]))
            V2_plot1.append(float(row[2]))
            V3_plot1.append(float(row[3]))
            V4_plot1.append(float(row[4]))
            V5_plot1.append(float(row[5]))
            I_plot1.append(float(row[6]))

ax[0].plot(times_plot1, I_plot1, alpha=0.75, color='red', linestyle='-', label='Neurons 1 & 2')
ax[0].plot(times_plot1, I_plot1[0]*np.ones(len(times_plot1)), alpha=0.75, color='blue', linestyle='-', label='Neurons 3, 4 & 5')

ax[1].plot(times_plot1, V1_plot1, alpha=0.75, color=c[0], linestyle='-', label='$V_1$')
ax[1].plot(times_plot1, V2_plot1, alpha=0.75, color=c[1], linestyle='-', label='$V_2$')
ax[1].plot(times_plot1, V3_plot1, alpha=0.75, color=c[2], linestyle='-', label='$V_3$')
ax[1].plot(times_plot1, V4_plot1, alpha=0.75, color=c[3], linestyle='-', label='$V_4$')
ax[1].plot(times_plot1, V5_plot1, alpha=0.75, color=c[4], linestyle='-', label='$V_5$')


# A spike occurs iff there was a reset
# Plot 1
spike_times_V1_plot1 = [times_plot1[i] for i in range(1,len(V1_plot1)) if abs(V1_plot1[i]-V1_plot1[i-1]) > (Vth-Vr)/2]
spike_times_V2_plot1 = [times_plot1[i] for i in range(1,len(V2_plot1)) if abs(V2_plot1[i]-V2_plot1[i-1]) > (Vth-Vr)/2]
spike_times_V3_plot1 = [times_plot1[i] for i in range(1,len(V3_plot1)) if abs(V3_plot1[i]-V3_plot1[i-1]) > (Vth-Vr)/2]
spike_times_V4_plot1 = [times_plot1[i] for i in range(1,len(V4_plot1)) if abs(V4_plot1[i]-V4_plot1[i-1]) > (Vth-Vr)/2]
spike_times_V5_plot1 = [times_plot1[i] for i in range(1,len(V5_plot1)) if abs(V5_plot1[i]-V5_plot1[i-1]) > (Vth-Vr)/2]

for t in spike_times_V1_plot1:
    ax[1].plot([t, t], [Vth, Vth+0.5], alpha=0.75, color=c[0])
for t in spike_times_V2_plot1:
    ax[1].plot([t, t], [Vth, Vth+0.5], alpha=0.75, color=c[1])
for t in spike_times_V3_plot1:
    ax[1].plot([t, t], [Vth, Vth+0.5], alpha=0.75, color=c[2])
for t in spike_times_V4_plot1:
    ax[1].plot([t, t], [Vth, Vth+0.5], alpha=0.75, color=c[3])
for t in spike_times_V5_plot1:
    ax[1].plot([t, t], [Vth, Vth+0.5], alpha=0.75, color=c[4])

ax[1].set_xlabel('Time ($10^{-2}$ seconds)', size=11)
ax[1].set_ylabel('Voltage $V_k, k \in \{1,..,5\}$', size=11)

ax[0].set_ylabel('Current $I$', size=11)

fig.suptitle('Network of 5 electrically coupled neurons, $\\beta=0.1$ and $\gamma=0.1$', size=15)
ax[0].legend(loc='upper right')
ax[1].legend(loc='upper right') #bbox_to_anchor=(1, 1)

plt.savefig('synchrony.svg')
plt.show()
