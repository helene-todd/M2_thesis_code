from matplotlib import cm, rcParams
import matplotlib.pyplot as plt
import numpy as np
import math as math
import random as rand
import os
import csv

rcParams.update({'figure.autolayout': True})
plt.figure(figsize=(20,4))

c = ['#aa3863', '#3b7d86', '#5443a3']

times = []
V1, V2 = [], []

Vth = 1
Vr = 0

with open('gamma=0.1_many.dat', newline='') as file:
    datareader = csv.reader(file, delimiter=' ')
    for row in datareader:
        if float(row[0]) >= 200 and float(row[0]) <= 300  :
            times.append(float(row[0]))
            V1.append(float(row[1]))
            V2.append(float(row[2]))

plt.plot(times, V1, alpha=0.75, color=c[0], linestyle='-', label='$V_1$')
plt.plot(times, V2, alpha=0.75, color=c[1], linestyle='-', label='$V_2$')
'''
# A spike occurs iff there was a reset
spike_times_V1 = [times[i] for i in range(1,len(V1)) if abs(V1[i]-V1[i-1]) > (Vth-Vr)/2]
spike_times_V2 = [times[i] for i in range(1,len(V2)) if abs(V2[i]-V2[i-1]) > (Vth-Vr)/2]

for t in spike_times_V1:
    plt.plot([t-times[1], t-times[1]], [Vth, Vth+0.5], alpha=0.75, color=c[0])

for t in spike_times_V2:
    plt.plot([t-times[1], t-times[1]], [Vth, Vth+0.5], alpha=0.75, color=c[1])
'''
plt.xlabel('Time ($10^{-2}$ seconds)', size=12)
#plt.xticks([k for k in range(11)])
plt.ylabel('Voltage $V_k, k \in \{1,2\}$', size=12)

plt.legend(loc='upper right', bbox_to_anchor=(1, 0.95))

#plt.savefig('noise.svg')
plt.show()
