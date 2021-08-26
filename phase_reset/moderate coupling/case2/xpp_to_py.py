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
I_plot1, I_plot2 = [], []

Vth = 1
Vr = 0

fig, ax = plt.subplots(2, 1, figsize=(8,6), sharex='col')

k = 0
with open('phase_reset.dat', newline='') as file:
    datareader = csv.reader(file, delimiter=' ')
    for row in datareader:
        if float(row[0]) >= 5 and float(row[0]) <= 20 and k%20 == 0 :
            times_plot1.append(float(row[0]))
            V1_plot1.append(float(row[1]))
            V2_plot1.append(float(row[2]))
            I_plot1.append(float(row[3]))

        k += 1

# A spike occurs iff there was a reset
for i in range(1,len(V1_plot1)) :
    if abs(V1_plot1[i]-V1_plot1[i-1]) > (Vth-Vr)/2 and V1_plot1[i] < 1 and V1_plot1[i-1] < 1:
        V1_plot1.insert(i, Vth+0.5)
        V2_plot1.insert(i, V2_plot1[i])
        I_plot1.insert(i, I_plot1[i])
        times_plot1.insert(i, times_plot1[i])

    if abs(V2_plot1[i]-V2_plot1[i-1]) > (Vth-Vr)/2 and V2_plot1[i] < 1 and V2_plot1[i-1] < 1:
        V2_plot1.insert(i, Vth+0.5)
        V1_plot1.insert(i, V1_plot1[i])
        I_plot1.insert(i, I_plot1[i])
        times_plot1.insert(i, times_plot1[i])

ax[1].plot(times_plot1, V1_plot1, alpha=1, color=c[0], linestyle='-', label='$V_1$') #alpha=0.75
ax[1].plot(times_plot1, V2_plot1, alpha=1, color=c[1], linestyle='-', label='$V_2$') #alpha=0.75

ax[0].plot(times_plot1, I_plot1, alpha=1, color=c[2], linestyle='-', label='$I$') #alpha=0.75

ax[1].set_xlabel('Time ($10^{-2}$ seconds)', size=12)
ax[1].set_ylabel('Voltage $V_k, k \in \{1,2\}$', size=12)
ax[0].set_ylabel('Current $I$', size=12)

fig.suptitle('From A/S regime to A/S regime', size=14)
ax[0].legend(loc='upper right', bbox_to_anchor=(1, 0.95), fontsize=12)
ax[1].legend(loc='upper right', bbox_to_anchor=(1, 0.95), fontsize=12)

plt.tight_layout()
plt.savefig('from_as_to_as_2.svg')
#plt.show()
