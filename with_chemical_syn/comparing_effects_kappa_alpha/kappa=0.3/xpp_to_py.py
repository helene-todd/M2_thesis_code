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

times_plot1, times_plot2 = [], []
V1_plot1, V2_plot1, V1_plot2, V2_plot2 = [], [], [], []
I1_plot1, I2_plot1, I1_plot2, I2_plot2 = [], [], [], []

Vth = 1
Vr = 0

fig, ax = plt.subplots(2, 2, figsize=(16,5), sharey='row', sharex='col')

k = 0
with open('alpha=4.dat', newline='') as file:
    datareader = csv.reader(file, delimiter=' ')
    for row in datareader:
        if float(row[0]) >= 0 and float(row[0]) <= 20 :
            times_plot1.append(float(row[0]))
            V1_plot1.append(float(row[1]))
            V2_plot1.append(float(row[2]))
            I1_plot1.append(float(row[3]))
            I2_plot1.append(float(row[4]))
        if float(row[0]) >= 80 and float(row[0]) <= 100 :
            times_plot2.append(float(row[0]))
            V1_plot2.append(float(row[1]))
            V2_plot2.append(float(row[2]))
            I1_plot2.append(float(row[3]))
            I2_plot2.append(float(row[4]))

# A spike occurs iff there was a reset
for i in range(1,len(V1_plot1)) :
    if abs(V1_plot1[i]-V1_plot1[i-1]) > (Vth-Vr)/2 and V1_plot1[i] <= 1 and V1_plot1[i-1] <= 1:
        V1_plot1.insert(i, Vth+0.5)
        V2_plot1.insert(i, V2_plot1[i])
        I1_plot1.insert(i, I1_plot1[i])
        I2_plot1.insert(i, I2_plot1[i])
        times_plot1.insert(i, times_plot1[i])

    if abs(V2_plot1[i]-V2_plot1[i-1]) > (Vth-Vr)/2 and V2_plot1[i] <= 1 and V2_plot1[i-1] <= 1:
        V2_plot1.insert(i, Vth+0.5)
        V1_plot1.insert(i, V1_plot1[i])
        I1_plot1.insert(i, I1_plot1[i])
        I2_plot1.insert(i, I2_plot1[i])
        times_plot1.insert(i, times_plot1[i])

for i in range(1,len(V1_plot2)) :
    if abs(V1_plot2[i]-V1_plot2[i-1]) > (Vth-Vr)/2 and V1_plot2[i] <= 1 and V1_plot2[i-1] <= 1:
        V1_plot2.insert(i, Vth+0.5)
        V2_plot2.insert(i, V2_plot2[i])
        I1_plot2.insert(i, I1_plot2[i])
        I2_plot2.insert(i, I2_plot2[i])
        times_plot2.insert(i, times_plot2[i])

    if abs(V2_plot2[i]-V2_plot2[i-1]) > (Vth-Vr)/2 and V2_plot2[i] < 1 and V2_plot2[i-1] < 1:
        V2_plot2.insert(i, Vth+0.5)
        V1_plot2.insert(i, V1_plot2[i])
        I1_plot2.insert(i, I1_plot2[i])
        I2_plot2.insert(i, I2_plot2[i])
        times_plot2.insert(i, times_plot2[i])

ax[1, 0].plot(times_plot1, V1_plot1, alpha=1, color=c[0], linestyle='-') #alpha=0.75
ax[1, 0].plot(times_plot1, V2_plot1, alpha=1, color=c[1], linestyle='-') #alpha=0.75

ax[0, 0].plot(times_plot1, I1_plot1, alpha=1, color=c[0], linestyle='-') #alpha=0.75
ax[0, 0].plot(times_plot1, I2_plot1, alpha=1, color=c[1], linestyle='-') #alpha=0.75

ax[1, 1].plot(times_plot2, V1_plot2, alpha=1, color=c[0], linestyle='-', label='$V_1$') #alpha=0.75
ax[1, 1].plot(times_plot2, V2_plot2, alpha=1, color=c[1], linestyle='-', label='$V_2$') #alpha=0.75

ax[0, 1].plot(times_plot2, I1_plot2, alpha=1, color=c[0], linestyle='-', label='$I_{tot,1}$') #alpha=0.75
ax[0, 1].plot(times_plot2, I2_plot2, alpha=1, color=c[1], linestyle='-', label='$I_{tot,2}$') #alpha=0.75

ax[1, 0].set_xlabel('Time ($10^{-2}$ seconds)', size=12)
ax[1, 1].set_xlabel('Time ($10^{-2}$ seconds)', size=12)
ax[1, 0].set_ylabel('Voltage $V_{k}, k \in \{1,2\}$', size=12)
ax[0, 0].set_ylabel('Current $I_{tot, k}, k \in \{1,2\}$', size=12)

ax[0, 0].set_ylim(1.,2.)

fig.suptitle('Coupled neurons, $\gamma=0.2, \\beta=0.1, \kappa=0.3, \\alpha=4$', size=18)
ax[0,1].legend(loc='upper right', bbox_to_anchor=(1, 0.95), fontsize=12)
ax[1,1].legend(loc='upper right', bbox_to_anchor=(1, 0.95), fontsize=12)

plt.tight_layout()
plt.savefig('alpha=4.svg')
plt.show()
