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
#matplotlib.pyplot.xkcd(scale=.4, length=100, randomness=2)

c = ['#aa3863', '#3b7d86']
s = ['-', '--']

del_line = 'k'

fig, ax = plt.subplots(2, 3, figsize=(16,8), sharey='row', sharex='col')

min_val = 10**0
max_val = 40

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def row_count(filename):
    with open(filename) as in_file:
        return sum(1 for _ in in_file)

""" beta = 0.1, gamma = 0.1 """

I = [[]]
phi = [[]]
stability = []

for filename in ['beta=0.1/gamma = 0.1/gamma_0.1.dat', 'beta=0.1/gamma = 0.1/stable1.dat', 'beta=0.1/gamma = 0.1/stable2.dat'] :
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

data = np.load('beta=0.1/gamma = 0.1/mesh_cycles.npz')

legend_stable, legend_unstable = False, False
for k in range(len(I)) :
    if stability[k] == 1 and legend_stable == False :
        ax[0,0].plot(I[k], phi[k], color='black', linewidth=2, linestyle=s[stability[k]-1], label='stable')
        legend_stable = True
    if stability[k] == 1 and legend_stable == True :
        ax[0,0].plot(I[k], phi[k], color='black', linewidth=2, linestyle=s[stability[k]-1])
    if stability[k] == 2 and legend_unstable == False :
        ax[0,0].plot(I[k], phi[k], color='black', linewidth=2, linestyle=s[stability[k]-1], label='unstable')
        legend_unstable = True
    if stability[k] == 2 and legend_unstable == True :
        ax[0,0].plot(I[k], phi[k], color='black', linewidth=2, linestyle=s[stability[k]-1])

ax[0,0].legend(loc='upper right', bbox_to_anchor=(1, 0.95), fontsize=10)

# regime delimiter to make things more visual
ax[0,0].set_ylim(-0.05, 1.05)
ax[0,0].set_xlim(1, 2)

ax[0,0].set_title('$\gamma$=0.1, $\\beta$=0.1', fontsize=13)
ax[0,0].set_ylabel('Phase Difference $\phi$', size=12)

ax[0,0].pcolormesh(data['I'], data['phi'], data['cycles'], cmap='Spectral', shading='smooth', edgecolors=None, vmin=min_val, vmax=max_val)

# the delimiter line
#data = np.load('beta=0.1/gamma = 0.1/line.npz')
#ax[0,0].plot(data['I'], data['phi'], color=del_line, linestyle='-.', alpha=0.4)
#ax[0,0].plot([min(data['I']), min(data['I'])], [0, 1], color=del_line, linestyle='-.', alpha=0.5)

""" beta = 0.1, gamma = 0.2 """

I = [[]]
phi = [[]]
stability = []

for filename in ['beta=0.1/gamma = 0.2/gamma_0.2.dat', 'beta=0.1/gamma = 0.2/stable1.dat', 'beta=0.1/gamma = 0.2/stable2.dat']:
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

data = np.load('beta=0.1/gamma = 0.2/mesh_cycles.npz')

for k in range(len(I)) :
    if stability[k] == 1 :
        ax[0,1].plot(I[k], phi[k], color='black', linewidth=2, linestyle=s[stability[k]-1], label='stable')
    if stability[k] == 2 :
        ax[0,1].plot(I[k], phi[k], color='black', linewidth=2, linestyle=s[stability[k]-1], label='unstable')

# regime delimiter to make things more visual
ax[0,1].set_ylim(-0.05, 1.05)
ax[0,1].set_xlim(1, 2)

ax[0,1].set_title('$\gamma$=0.2, $\\beta$=0.1', fontsize=13)
ax[0,1].pcolormesh(data['I'], data['phi'], data['cycles'], cmap='Spectral', shading='smooth', edgecolors=None, vmin=min_val, vmax=max_val)


""" beta = 0.1, gamma = 0.4 """

I = [[]]
phi = [[]]
stability = []

for filename in ['beta=0.1/gamma = 0.4/gamma_0.4.dat', 'beta=0.1/gamma = 0.4/stable1.dat', 'beta=0.1/gamma = 0.4/stable2.dat']:
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

data = np.load('beta=0.1/gamma = 0.4/mesh_cycles.npz')

for k in range(len(I)) :
    if stability[k] == 1 :
        ax[0,2].plot(I[k], phi[k], color='black', linewidth=2, linestyle=s[stability[k]-1], label='stable')
    if stability[k] == 2 :
        ax[0,2].plot(I[k], phi[k], color='black', linewidth=2, linestyle=s[stability[k]-1], label='unstable')

# regime delimiter to make things more visual
ax[0,2].set_ylim(-0.05, 1.05)
ax[0,2].set_xlim(1, 2)

ax[0,2].set_title('$\gamma$=0.4, $\\beta$=0.1', fontsize=13)
ax[0,2].pcolormesh(data['I'], data['phi'], data['cycles'], cmap='Spectral', shading='smooth', edgecolors=None, vmin=min_val, vmax=max_val)

# the delimiter line
#data = np.load('beta=0.1/gamma = 0.4/line.npz')
#ax[0,1].plot(data['I'], data['phi'], color=del_line, linestyle='-.', alpha=0.4)
#ax[0,1].plot([min(data['I']), min(data['I'])], [0, 1], color=del_line, linestyle='-.', alpha=0.5)

""" beta = 0.2, gamma = 0.1 """

I = [[]]
phi = [[]]
stability = []

for filename in ['beta=0.2/gamma = 0.1/gamma_0.1.dat', 'beta=0.2/gamma = 0.1/stable1.dat', 'beta=0.2/gamma = 0.1/stable2.dat'] :
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



data = np.load('beta=0.2/gamma = 0.1/mesh_cycles.npz')

for k in range(len(I)) :
    if stability[k] == 1 :
        ax[1,0].plot(I[k], phi[k], color='black', linewidth=2, linestyle=s[stability[k]-1], label='stable')
    if stability[k] == 2 :
        ax[1,0].plot(I[k], phi[k], color='black', linewidth=2, linestyle=s[stability[k]-1], label='unstable')

# regime delimiter to make things more visual
ax[1,0].set_ylim(-0.05, 1.05)
ax[1,0].set_xlim(1, 2)

ax[1,0].set_title('$\gamma$=0.1, $\\beta$=0.2', fontsize=13)
ax[1,0].set_xlabel('Current $I$', size=12)
ax[1,0].set_ylabel('Phase Difference $\phi$', size=12)

ax[1,0].pcolormesh(data['I'], data['phi'], data['cycles'], cmap='Spectral', shading='smooth', edgecolors=None, vmin=min_val, vmax=max_val)

# the delimiter line
#data = np.load('beta=0.2/gamma = 0.1/line.npz')
#ax[1,0].plot(data['I'], data['phi'], color=del_line, linestyle='-.', alpha=0.4)
#ax[1,0].plot([min(data['I']), min(data['I'])], [0, 1], color=del_line, linestyle='-.', alpha=0.5)

""" beta = 0.2, gamma = 0.2 """

I = [[]]
phi = [[]]
stability = []

for filename in ['beta=0.2/gamma = 0.2/gamma_0.2.dat', 'beta=0.2/gamma = 0.2/stable1.dat', 'beta=0.2/gamma = 0.2/stable2.dat']:
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

data = np.load('beta=0.2/gamma = 0.2/mesh_cycles.npz')

for k in range(len(I)) :
    if stability[k] == 1 :
        ax[1,1].plot(I[k], phi[k], color='black', linewidth=2, linestyle=s[stability[k]-1], label='stable')
    if stability[k] == 2 :
        ax[1,1].plot(I[k], phi[k], color='black', linewidth=2, linestyle=s[stability[k]-1], label='unstable')

# regime delimiter to make things more visual
ax[1,1].set_ylim(-0.05, 1.05)
ax[1,1].set_xlim(1, 2)

ax[1,1].set_title('$\gamma$=0.2, $\\beta$=0.2', fontsize=13)
ax[1,1].set_xlabel('Current $I$', size=12)
ax[1,1].pcolormesh(data['I'], data['phi'], data['cycles'], cmap='Spectral', shading='smooth', edgecolors=None, vmin=min_val, vmax=max_val)


""" beta = 0.2, gamma = 0.4 """

I = [[]]
phi = [[]]
stability = []

for filename in ['beta=0.2/gamma = 0.4/gamma_0.4.dat', 'beta=0.2/gamma = 0.4/stable1.dat', 'beta=0.2/gamma = 0.4/stable2.dat']:
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

data = np.load('beta=0.2/gamma = 0.4/mesh_cycles.npz')

for k in range(len(I)) :
    if stability[k] == 1 :
        ax[1,2].plot(I[k], phi[k], color='black', linewidth=2, linestyle=s[stability[k]-1], label='stable')
    if stability[k] == 2 :
        ax[1,2].plot(I[k], phi[k], color='black', linewidth=2, linestyle=s[stability[k]-1], label='unstable')

# regime delimiter to make things more visual
ax[1,2].set_ylim(-0.05, 1.05)
ax[1,2].set_xlim(1, 2)

ax[1,2].set_title('$\gamma$=0.4, $\\beta$=0.2', fontsize=13)
ax[1,2].set_xlabel('Current $I$', size=12)

im = ax[1,2].pcolormesh(data['I'], data['phi'], data['cycles'], cmap='Spectral', shading='smooth', edgecolors=None, vmin=min_val, vmax=max_val)

# the delimiter line
#data = np.load('beta=0.2/gamma = 0.4/line.npz')
#ax[1,1].plot(data['I'], data['phi'], color=del_line, linestyle='-.', alpha=0.4)
#ax[1,1].plot([min(data['I']), min(data['I'])], [0, 1], color=del_line, linestyle='-.', alpha=0.5)

""" General Settings """

plt.suptitle('Bifurcation diagrams for moderately coupled neurons, with convergence speed', size=18)
fig.tight_layout()

right =  0.95
fig.subplots_adjust(right=right)
cbar = fig.colorbar(im, ax=ax[:])

print(cbar.ax.get_yticklabels()[:-1])
cbar.ax.set_yticklabels(['5', '10', '15', '20','25', '30', '35', '>40'])
cbar.set_label('Time (in $10^{-2}$ s) to converge towards synchrony', labelpad=20, fontsize=15)

plt.savefig('comparing_bifs_cv_speed6.png', dpi=600)
plt.show()
