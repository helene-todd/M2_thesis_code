from matplotlib import cm, rcParams
import matplotlib.pyplot as plt
import matplotlib as matplotlib
import numpy as np
import math as math
import random as rand
import os
import csv
import argparse

# RUN python prettyplot.py stable1.dat stable2.dat gamma\=0.1_beta\=0.1.dat gamma\=0.1_beta\=0.2.dat gamma\=0.2_beta\=0.1.dat gamma\=0.2_beta\=0.2.dat

plt.rcParams['axes.xmargin'] = 0

p = argparse.ArgumentParser()
p.add_argument('files', type=str, nargs='*')
args = p.parse_args()

matplotlib.rc('xtick', labelsize=11)
matplotlib.rc('ytick', labelsize=11)

fig, ax = plt.subplots(2, 2, figsize=(10,8), sharey='row', sharex='col')

c = ['#aa3863', '#3b7d86']
s = ['-', '--']

def row_count(file):
    with open(file) as in_file:
        return sum(1 for _ in in_file)

def dat_to_plot(file, i, j):
    I, phi, stability = [[]], [[]], []

    with open(file, newline='') as filename:

        if 'stable' in file : stableLegend, unstableLegend = True, True # Avoid duplicate legend labels
        else : stableLegend, unstableLegend = False, False

        datareader = csv.reader(filename, delimiter=' ')
        last_line_nb = row_count(file)

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

    if file != 'stable1.dat' and file != 'stable2.dat':
        Imin, Imax = 2, 0
        for l in range(len(I)) :
            for k in range(len(I[l])) :
                if phi[l][k] not in [0, 1, 0.5] and I[l][k] > Imax :
                    Imax = I[l][k]
                if phi[l][k] == 0.5 and stability[l] == 1 and I[l][k] < Imin :
                    Imin = I[l][k]

        # regime delimiter to make things more visual
        ax[i, j].axvspan(Imin, Imax, facecolor='0.2', alpha=0.1)

    # Avoid puplicate legend entries
    for k in range(len(I)) :
        if stability[k] == 1 and stableLegend == True :
            ax[i, j].plot(I[k], phi[k], color='black', linestyle=s[stability[k]-1])
            print(i, j, 'no legend st', stableLegend)

        if stability[k] == 1 and stableLegend == False :
            ax[i, j].plot(I[k], phi[k], color='black', linestyle=s[stability[k]-1], label='stable')
            stableLegend = True
            print(i, j, 'legend st', stableLegend)

        if stability[k] == 2 and unstableLegend == True :
            ax[i, j].plot(I[k], phi[k], color='black', linestyle=s[stability[k]-1])
            print(i, j, 'no legend un', unstableLegend)

        if stability[k] == 2 and unstableLegend == False :
            ax[i, j].plot(I[k], phi[k], color='black', linestyle=s[stability[k]-1], label='unstable')
            unstableLegend = True
            print(i, j, 'legend un', unstableLegend)

if 'stable1.dat' not in args.files or 'stable2.dat' not in args.files:
    print('Must include stable1.dat and stable2.dat in args')
    quit()

files = []
values_gamma = []
values_beta = []
for file in args.files :
    if file != 'stable1.dat' and file != 'stable2.dat':
        files.append(file)
        file = file.replace('_', ' ').replace('=', ' ').replace('.dat', '').split(' ')
        print(file)
        values_gamma.append(file[1])
        values_beta.append(file[3])

i, j = 0, 0
k = 0

print(len(files))
for i in range(int(len(files)/2)):
    for j in range(int(len(files)/2)):
        dat_to_plot(files[k], i, j)
        dat_to_plot('stable1.dat', i, j)
        dat_to_plot('stable2.dat', i, j)
        ax[i, j].set_title(f'$\gamma={values_gamma[k]}, \\beta={values_beta[k]}$', fontsize=13)
        k += 1

ax[1, 0].set_xlabel('Current $I$', size=12)
ax[1, 1].set_xlabel('Current $I$', size=12)

ax[0, 0].set_ylabel('Phase Difference $\phi$', size=12)
ax[1, 0].set_ylabel('Phase Difference $\phi$', size=12)

ax[0, 1].legend(bbox_to_anchor=(0.995,0.95))

fig.suptitle('Bifurcation diagrams for moderately two coupled neurons', size=15)

fig.tight_layout()
fig.subplots_adjust(hspace=0.25, top=0.9, wspace=0.25)
# subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)

plt.savefig('allinone2.svg')
plt.show()
