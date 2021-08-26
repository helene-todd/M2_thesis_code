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

#c = ['#aa3863', '#3b7d86', '#5443a3']
c = ['#C06B8B', '#6D9EA5']

times_plot1, times_plot2 = [], []
V1_plot1, V2_plot1,  = [], []
V1_plot2, V2_plot2 = [], []

Vth = 30
Vr = -10

fig, ax = plt.subplots(1, 2, figsize=(14,4), sharey='row')

k = 0
with open('data_dt=10-4.dat', newline='') as file:
    datareader = csv.reader(file, delimiter=' ')
    for row in datareader:
        if float(row[0]) >= 0 and float(row[0]) <= 40 :
            times_plot1.append(float(row[0]))
            V1_plot1.append(float(row[1]))
            V2_plot1.append(float(row[2]))

with open('data_dt=10-5.dat', newline='') as file:
    datareader = csv.reader(file, delimiter=' ')
    for row in datareader:
        if float(row[0]) >= 0 and float(row[0]) <= 20 :
            times_plot2.append(float(row[0]))
            V1_plot2.append(float(row[1]))
            V2_plot2.append(float(row[2]))


# A spike occurs iff there was a reset
for i in range(1,len(V1_plot1)) :
    if abs(V1_plot1[i]-V1_plot1[i-1]) > (Vth-Vr)/2 and V1_plot1[i] <= Vth and V1_plot1[i-1] <= Vth:
        V1_plot1.insert(i, Vth+5)
        V2_plot1.insert(i, V2_plot1[i])
        times_plot1.insert(i, times_plot1[i])

# A spike occurs iff there was a reset
for i in range(1,len(V2_plot1)) :
    if abs(V2_plot1[i]-V2_plot1[i-1]) > (Vth-Vr)/2 and V2_plot1[i] <= Vth and V2_plot1[i-1] <= Vth:
        V2_plot1.insert(i, Vth+5)
        V1_plot1.insert(i, V1_plot1[i])
        times_plot1.insert(i, times_plot1[i])

# A spike occurs iff there was a reset
for i in range(1,len(V1_plot2)) :
    if abs(V1_plot2[i]-V1_plot2[i-1]) > (Vth-Vr)/2 and V1_plot2[i] <= Vth and V1_plot2[i-1] <= Vth:
        V1_plot2.insert(i, Vth+5)
        V2_plot2.insert(i, V2_plot2[i])
        times_plot2.insert(i, times_plot2[i])

# A spike occurs iff there was a reset
for i in range(1,len(V2_plot2)) :
    if abs(V2_plot2[i]-V2_plot2[i-1]) > (Vth-Vr)/2 and V2_plot2[i] <= Vth and V2_plot2[i-1] <= Vth:
        V2_plot2.insert(i, Vth+5)
        V1_plot2.insert(i, V1_plot2[i])
        times_plot2.insert(i, times_plot2[i])

# Rescaling such that Vth = 1 and Vr = 0
V1_plot1, V2_plot1, V1_plot2, V2_plot2 = np.array(V1_plot1), np.array(V2_plot1), np.array(V1_plot2), np.array(V2_plot2)
V1_plot1, V2_plot1, V1_plot2, V2_plot2 = (V1_plot1+10)/30, (V2_plot1+10)/30, (V1_plot2+10)/30, (V2_plot2+10)/30

ax[0].plot(times_plot1, V1_plot1, alpha=1, color=c[0], linestyle='-', label='$V_1$') #alpha=0.75
ax[0].plot(times_plot1, V2_plot1, alpha=1, color=c[1], linestyle='-', label='$V_2$') #alpha=0.75

ax[1].plot(times_plot2, V1_plot2, alpha=1, color=c[0], linestyle='-', label='$V_1$') #alpha=0.75
ax[1].plot(times_plot2, V2_plot2, alpha=1, color=c[1], linestyle='-', label='$V_2$') #alpha=0.75

ax[0].set_xlabel('Time ($10^{-2}$ seconds)', size=12)
ax[1].set_xlabel('Time ($10^{-2}$ seconds)', size=12)

ax[0].set_ylabel('Voltage $V_{k}, k \in \{1,2\}$', size=12)

fig.suptitle('The choice of $dt$ affects synchronization', size=16)
ax[0].set_title('Timestep $dt = 10^{-4}}]$', size=14)
ax[1].set_title('Timestep $dt = 10^{-5}}]$', size=14)

ax[0].legend(loc='upper right', fontsize=12)
ax[1].legend(loc='upper right', fontsize=12)

plt.tight_layout()
plt.savefig('compare_dt.svg')
plt.show()
