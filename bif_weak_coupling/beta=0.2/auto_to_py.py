from matplotlib import cm, rcParams
import matplotlib.pyplot as plt
import numpy as np
import math as math
import random as rand
import os
import csv
import argparse

# TO DO : Rewrite this code to make it more readable.
# USAGE : Run in terminal  "python auto_to_py.py bif_points_1.dat bif_points_2.dat bif_points_3.dat"

plt.rcParams['axes.xmargin'] = 0

p = argparse.ArgumentParser()
p.add_argument('files', type=str, nargs='*')
args = p.parse_args()

def row_count(filename):
    with open(filename) as in_file:
        return sum(1 for _ in in_file)

c = ['#B5EAD7', '#C7CEEA']
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

# regime delimiter to make things more visual
plt.axvspan(1, 1.25925, facecolor='0.2', alpha=0.1)

for k in range(len(I)) :
    if stability[k] == 1 :
        plt.plot(I[k], phi[k], color='black', linestyle=s[stability[k]-1], label='stable')
    if stability[k] == 2 :
        plt.plot(I[k], phi[k], color='black', linestyle=s[stability[k]-1], label='unstable')

plt.title('Bifurcation diagram for two weakly coupled neurons, $\\beta =0.2$', fontsize=11)
plt.xlabel('Current $I$', fontsize=10.5)
plt.ylabel('Phase Difference $\phi$', fontsize=10.5)

# remove duplicate legend
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='upper right', bbox_to_anchor=(1, 0.95))

plt.savefig('bif_diagram_beta=0.2.svg')
#plt.show()
