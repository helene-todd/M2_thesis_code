from matplotlib import cm, rcParams
import matplotlib.pyplot as plt
import numpy as np
import math as math
import random as rand
import os
import csv

rcParams.update({'figure.autolayout': True})

c = ['#aa3863', '#3b7d86', '#5443a3']

times_plot1, times_plot2 = [], []
V1_plot1, V2_plot1, V1_plot2, V2_plot2 = [], [], [], []
I_plot1, I_plot2 = [], []

Vth = 1
Vr = 0

fig, ax = plt.subplots(2, 2, figsize=(16,6), sharey='row')

with open('phase_reset.dat', newline='') as file:
    datareader = csv.reader(file, delimiter=' ')
    for row in datareader:
        if float(row[0]) >= 10 and float(row[0]) <= 25  :
            times_plot1.append(float(row[0]))
            V1_plot1.append(float(row[1]))
            V2_plot1.append(float(row[2]))
            I_plot1.append(float(row[3]))

        if float(row[0]) >= 220 and float(row[0]) <= 235  :
            times_plot2.append(float(row[0]))
            V1_plot2.append(float(row[1]))
            V2_plot2.append(float(row[2]))
            I_plot2.append(float(row[3]))


ax[1, 0].plot(times_plot1, V1_plot1, alpha=0.75, color=c[0], linestyle='-')
ax[1, 0].plot(times_plot1, V2_plot1, alpha=0.75, color=c[1], linestyle='-')

ax[0, 0].plot(times_plot1, I_plot1, alpha=0.75, color=c[2], linestyle='-')


ax[1, 1].plot(times_plot2, V1_plot2, alpha=0.75, color=c[0], linestyle='-', label='$V_1$')
ax[1, 1].plot(times_plot2, V2_plot2, alpha=0.75, color=c[1], linestyle='-', label='$V_2$')

ax[0, 1].plot(times_plot2, I_plot2, alpha=0.75, color=c[2], linestyle='-', label='$I$')

# A spike occurs iff there was a reset
spike_times_V1_plot1 = [times_plot1[i] for i in range(1,len(V1_plot1)) if abs(V1_plot1[i]-V1_plot1[i-1]) > (Vth-Vr)/2]
spike_times_V2_plot1 = [times_plot1[i] for i in range(1,len(V2_plot1)) if abs(V2_plot1[i]-V2_plot1[i-1]) > (Vth-Vr)/2]

for t in spike_times_V1_plot1:
    ax[1, 0].plot([t, t], [Vth, Vth+0.5], alpha=0.75, color=c[0])

for t in spike_times_V2_plot1:
    ax[1, 0].plot([t, t], [Vth, Vth+0.5], alpha=0.75, color=c[1])

spike_times_V1_plot2 = [times_plot2[i] for i in range(1,len(V1_plot2)) if abs(V1_plot2[i]-V1_plot2[i-1]) > (Vth-Vr)/2]
spike_times_V2_plot2 = [times_plot2[i] for i in range(1,len(V2_plot2)) if abs(V2_plot2[i]-V2_plot2[i-1]) > (Vth-Vr)/2]

for t in spike_times_V1_plot2:
    ax[1, 1].plot([t, t], [Vth, Vth+0.5], alpha=0.75, color=c[0])

for t in spike_times_V2_plot2:
    ax[1, 1].plot([t, t], [Vth, Vth+0.5], alpha=0.75, color=c[1])

ax[1, 0].set_xlabel('Time ($10^{-2}$ seconds)', size=12)
ax[1, 1].set_xlabel('Time ($10^{-2}$ seconds)', size=12)
ax[1, 0].set_ylabel('Voltage $V_k, k \in \{1,2\}$', size=12)
ax[0, 0].set_ylabel('Current $I$', size=12)

fig.suptitle('From A/S regime to A/S regime', size=14)
ax[0,1].legend(loc='upper right', bbox_to_anchor=(1, 0.95))
ax[1,1].legend(loc='upper right', bbox_to_anchor=(1, 0.95))

plt.savefig('from_as_to_as.svg')
#plt.show()
