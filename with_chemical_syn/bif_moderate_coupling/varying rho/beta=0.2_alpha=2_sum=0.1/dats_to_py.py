from matplotlib import cm, rcParams
import matplotlib as matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math as math
import random as rand
import os
import csv
import argparse

# TO DO : Rewrite this code to make it more readable.
# USAGE : Run in terminal "python dats_to_py.py gamma=0_kappa=0.1.dat gamma=0.02_kappa=0.08.dat gamma=0.05_kappa=0.05.dat gamma=0.08_kappa=0.02.dat  gamma=0.1_kappa=0.dat line.dat stable1.dat stable2.dat".

plt.rcParams['axes.xmargin'] = 0
plt.figure(figsize=(8,6))
#matplotlib.pyplot.xkcd(scale=.4, length=100, randomness=2)

matplotlib.rc('xtick', labelsize=11)
matplotlib.rc('ytick', labelsize=11)

p = argparse.ArgumentParser()
p.add_argument('files', type=str, nargs='*')
args = p.parse_args()

def row_count(filename):
    with open(filename) as in_file:
        return sum(1 for _ in in_file)

c = ['#aa3863', '#d97020', '#ef9f07', '#449775', '#3b7d86']
s = ['-', '--']

I = [[]]
phi = [[]]
stability = []

for filename in args.files :
    with open(filename, newline='') as file:
        datareader = csv.reader(file, delimiter=' ')

        last_line_nb = row_count(filename)

        last_I = -999
        last_phi = -999
        last_stability = 0

        # seperate by checking if two consecutive values are duplicates
        for row in datareader:

            # this last condition avoids a list with one value when two consecutive values are duplicates
            if last_I == float(row[0]) and len(I[-1]) > 1 :
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

rho = [0, 0.25, 0.5, 0.75, 1]
p = 0
for k in range(len(I)) :
    if phi[k][1] != 0.5 and phi[k][1] != 1 and phi[k][1] != 0 :
        plt.plot(I[k], phi[k], color=c[int(p/2)], linestyle=s[1], label=f'$\\rho = {rho[int(p/2)]}$')
        p += 1

    else :
        if stability[k] == 1 :
            plt.plot(I[k], phi[k], color='k', linestyle=s[0]) #linestyle=s[stability[k]-1]
        if stability[k] == 2 :
            plt.plot(I[k], phi[k], color='k', linestyle=s[1])

plt.title('Bifurcation diagram for two coupled neurons, various $\\rho$', fontsize=15, pad=15)
plt.xlabel('Current $I$', fontsize=14, labelpad=10)
plt.ylabel('Phase Difference $\phi$', fontsize=14)

# remove duplicate legend
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='upper right', bbox_to_anchor=(1, 0.95), prop={'size': 11})

plt.tight_layout()
plt.savefig('rho_range.svg')
plt.show()
