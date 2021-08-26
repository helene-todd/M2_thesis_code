from matplotlib import cm, rcParams
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib as matplotlib
import numpy as np
import math as math
import random as rand
import os, sys
import csv
import argparse

#plt.rcParams['axes.xmargin'] = 0
#plt.rcParams['axes.facecolor'] = 'black'

c = ['#aa3863', '#3b7d86']
s = ['-', '--']

fig, ax = plt.subplots(1, 2, figsize=(15,6), sharey='row')

min_val, max_val = 10**0, 10**3

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def row_count(filename):
    with open(filename) as in_file:
        return sum(1 for _ in in_file)

""" gamma = 0.2 """

I = [[]]
phi = [[]]
stability = []

for filename in ['gamma = 0.2/gamma_0.2.dat', 'gamma = 0.2/stable1.dat', 'gamma = 0.2/stable2.dat'] :
    with open(filename, newline='') as file:
        datareader = csv.reader(file, delimiter=' ')

        last_line_nb = row_count(filename)

        last_I = -999
        last_phi = -999
        last_stability = 0

        # seperate into sublists by checking if two consecutive values are duplicates
        for row in datareader:

            # the 2nd condition avoids a list with one value when two consecutive values are duplicates
            if last_I == float(row[0]) and len(I[-1]) > 1 :
                if last_stability != int(row[3]):
                    I[-1].append(last_I)
                    phi[-1].append(last_phi)
                I.append([])
                phi.append([])
                if last_stability != 0 :
                    stability.append(last_stability)

            if last_I != -999 :
                I[-1].append(last_I)
                phi[-1].append(last_phi)

            if last_stability != int(row[3]) and len(I[-1]) > 1:
                I.append([])
                phi.append([])
                if last_stability != 0 :
                    stability.append(last_stability)

            # if at last line, then stop checking for consecutive values and just add the remaining data
            if last_line_nb == datareader.line_num:
                I[-1].append(float(row[0]))
                phi[-1].append(float(row[1]))
                stability.append(int(row[3]))

            last_I = float(row[0])
            last_phi = float(row[1])
            last_stability = int(row[3])


Imin, Imax = 2, 0
for l in range(len(I)) :
    for k in range(len(I[l])) :
        if phi[l][k] not in [0, 1, 0.5] and I[l][k] > Imax :
            Imax = I[l][k]
        if phi[l][k] == 0.5 and stability[l] == 1 and I[l][k] < Imin :
            Imin = I[l][k]

data = np.load('gamma = 0.2/mesh_cycles.npz')

for k in range(len(I)) :
    if stability[k] == 1 :
        ax[0].plot(I[k], phi[k], color='black', linewidth=2, linestyle=s[stability[k]-1], label='stable')
    if stability[k] == 2 :
        ax[0].plot(I[k], phi[k], color='black', linewidth=2, linestyle=s[stability[k]-1], label='unstable')

# regime delimiter to make things more visual
ax[0].set_ylim(-0.05, 1.05)
ax[0].set_xlim(1, 2)

ax[0].set_title('$\gamma$=0.2', fontsize=14)
ax[0].set_xlabel('Current $I$', size=12)
ax[0].set_ylabel('Phase Difference $\phi$', size=12)

# remove duplicate legend
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))

ax[0].pcolormesh(data['I'], data['phi'], data['cycles'], cmap='viridis', shading='smooth', edgecolors=None, norm = colors.SymLogNorm(linthresh=0.03, linscale=0.03, vmin=min_val, vmax=max_val, base=10))

""" gamma = 0.4 """

I = [[]]
phi = [[]]
stability = []

for filename in ['gamma = 0.4/gamma_0.4.dat', 'gamma = 0.4/stable1.dat', 'gamma = 0.4/stable2.dat']:
    with open(filename, newline='') as file:
        datareader = csv.reader(file, delimiter=' ')

        last_line_nb = row_count(filename)

        last_I = -999
        last_phi = -999
        last_stability = 0

        # seperate into sublists by checking if two consecutive values are duplicates
        for row in datareader:

            # the 2nd condition avoids a list with one value when two consecutive values are duplicates
            if last_I == float(row[0]) and len(I[-1]) > 1 :
                if last_stability != int(row[3]):
                    I[-1].append(last_I)
                    phi[-1].append(last_phi)
                I.append([])
                phi.append([])
                if last_stability != 0 :
                    stability.append(last_stability)

            if last_I != -999 :
                I[-1].append(last_I)
                phi[-1].append(last_phi)

            if last_stability != int(row[3]) and len(I[-1]) > 1:
                I.append([])
                phi.append([])
                if last_stability != 0 :
                    stability.append(last_stability)

            # if at last line, then stop checking for consecutive values and just add the remaining data
            if last_line_nb == datareader.line_num:
                I[-1].append(float(row[0]))
                phi[-1].append(float(row[1]))
                stability.append(int(row[3]))

            last_I = float(row[0])
            last_phi = float(row[1])
            last_stability = int(row[3])


Imin, Imax = 2, 0
for l in range(len(I)) :
    for k in range(len(I[l])) :
        if phi[l][k] not in [0, 1, 0.5] and I[l][k] > Imax :
            Imax = I[l][k]
        if phi[l][k] == 0.5 and stability[l] == 1 and I[l][k] < Imin :
            Imin = I[l][k]

data = np.load('gamma = 0.4/mesh_cycles.npz')

for k in range(len(I)) :
    if stability[k] == 1 :
        ax[1].plot(I[k], phi[k], color='black', linewidth=2, linestyle=s[stability[k]-1], label='stable')
    if stability[k] == 2 :
        ax[1].plot(I[k], phi[k], color='black', linewidth=2, linestyle=s[stability[k]-1], label='unstable')

# regime delimiter to make things more visual
ax[1].set_ylim(-0.05, 1.05)
ax[1].set_xlim(1, 2)

ax[1].set_title('$\gamma$=0.4', fontsize=14)
ax[1].set_xlabel('Current $I$', size=12)

# remove duplicate legend
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax[1].legend(by_label.values(), by_label.keys(), loc='upper right', bbox_to_anchor=(1, 0.95))

plt.suptitle('Bifurcation diagram for two coupled neurons, $\\beta$=0.1', size=16)
im = ax[1].pcolormesh(data['I'], data['phi'], data['cycles'], cmap='viridis', shading='smooth', edgecolors=None, norm = colors.SymLogNorm(linthresh=0.03, linscale=0.03, vmin=min_val, vmax=max_val, base=10))
fig.tight_layout()

right =  1.06
fig.subplots_adjust(right=right)
cbar = fig.colorbar(im, ax=ax[:])
cbar.set_label('Number of cycles to converge towards synchrony', labelpad=20, fontsize=12)

plt.savefig('bifs_with_cv_speed.png', dpi=600)
plt.show()
