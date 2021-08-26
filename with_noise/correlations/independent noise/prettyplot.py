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

fig, ax = plt.subplots(2, 4, figsize=(16,6), sharex='col')

c = ['#aa3863', '#d97020', '#ef9f07', '#449775', '#3b7d86', '#5443a3']

""" Figure 8=1 """
with open('ds 1/data_correlations.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    row1 = next(reader)
    row2 = next(reader)
    corr = row2[4:]
    T = row2[1] # just in case we want it displayed

corr = [float(i) for i in corr]

t = np.linspace(-0.5, 0.5, len(corr))
ax[0, 0].plot(t, corr, marker='o', linestyle='-', markersize=8, color=c[-3], alpha=0.8, linewidth=1.2)
ax[0, 0].set_title('$s_1=s_2=0$')
ax[0, 0].set_xticks([-0.5, -0.25, 0, 0.25, 0.5])
ax[0, 0].set_ylabel('cross-correlation', size=11)

""" Figure 2 """
with open('ds 2/data_correlations.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    row1 = next(reader)
    row2 = next(reader)
    corr = row2[4:]
    T = row2[1] # just in case we want it displayed

corr = [float(i) for i in corr]

t = np.linspace(-0.5, 0.5, len(corr))
ax[0, 1].plot(t, corr, marker='o', linestyle='-', markersize=8, color=c[-3], alpha=0.8, linewidth=1.2)
ax[0, 1].set_title('$s_1=s_2=0.005$')
ax[0, 1].set_xticks([-0.5, -0.25, 0, 0.25, 0.5])

""" Figure 3 """
with open('ds 3/data_correlations.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    row1 = next(reader)
    row2 = next(reader)
    corr = row2[4:]
    T = row2[1] # just in case we want it displayed

corr = [float(i) for i in corr]

t = np.linspace(-0.5, 0.5, len(corr))
ax[0, 2].plot(t, corr, marker='o', linestyle='-', markersize=8, color=c[-3], alpha=0.8, linewidth=1.2)
ax[0, 2].set_title('$s_1=s_2=0.01$')
ax[0, 2].set_xticks([-0.5, -0.25, 0, 0.25, 0.5])

""" Figure 4 """
with open('ds 4/data_correlations.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    row1 = next(reader)
    row2 = next(reader)
    corr = row2[4:]
    T = row2[1] # just in case we want it displayed

corr = [float(i) for i in corr]

t = np.linspace(-0.5, 0.5, len(corr))
ax[0, 3].plot(t, corr, marker='o', linestyle='-', markersize=8, color=c[-3], alpha=0.8, linewidth=1.2)
ax[0, 3].set_title('$s_1=s_2=0.015$')
ax[0, 3].set_xticks([-0.5, -0.25, 0, 0.25, 0.5])

""" Figure 5 """
with open('ds 5/data_correlations.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    row1 = next(reader)
    row2 = next(reader)
    corr = row2[4:]
    T = row2[1] # just in case we want it displayed

corr = [float(i) for i in corr]

t = np.linspace(-0.5, 0.5, len(corr))
ax[1, 0].plot(t, corr, marker='o', linestyle='-', markersize=8, color=c[-3], alpha=0.8, linewidth=1.2)
ax[1, 0].set_title('$s_1=s_2=0.02$')
ax[1, 0].set_xticks([-0.5, -0.25, 0, 0.25, 0.5])
ax[1, 0].set_xlabel('$\\tau/T$', size=11)
ax[1, 0].set_ylabel('cross-correlation', size=11)

""" Figure 6 """
with open('ds 6/data_correlations.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    row1 = next(reader)
    row2 = next(reader)
    corr = row2[4:]
    T = row2[1] # just in case we want it displayed

corr = [float(i) for i in corr]

t = np.linspace(-0.5, 0.5, len(corr))
ax[1, 1].plot(t, corr, marker='o', linestyle='-', markersize=8, color=c[-3], alpha=0.8, linewidth=1.2)
ax[1, 1].set_title('$s_1=s_2=0.025$')
ax[1, 1].set_xticks([-0.5, -0.25, 0, 0.25, 0.5])
ax[1, 1].set_xlabel('$\\tau/T$', size=11)

""" Figure 7 """
with open('ds 7/data_correlations.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    row1 = next(reader)
    row2 = next(reader)
    corr = row2[4:]
    T = row2[1] # just in case we want it displayed

corr = [float(i) for i in corr]

t = np.linspace(-0.5, 0.5, len(corr))
ax[1, 2].plot(t, corr, marker='o', linestyle='-', markersize=8, color=c[-3], alpha=0.8, linewidth=1.2)
ax[1, 2].set_title('$s_1=s_2=0.03$')
ax[1, 2].set_xticks([-0.5, -0.25, 0, 0.25, 0.5])
ax[1, 2].set_xlabel('$\\tau/T$', size=11)

""" Figure 8 """
with open('ds 8/data_correlations.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    row1 = next(reader)
    row2 = next(reader)
    corr = row2[4:]
    T = row2[1] # just in case we want it displayed

corr = [float(i) for i in corr]

t = np.linspace(-0.5, 0.5, len(corr))
ax[1, 3].plot(t, corr, marker='o', linestyle='-', markersize=8, color=c[-3], alpha=0.8, linewidth=1.2)
ax[1, 3].set_title('$s_1=s_2=0.035$')
ax[1, 3].set_xticks([-0.5, -0.25, 0, 0.25, 0.5])
ax[1, 3].set_xlabel('$\\tau/T$', size=11)


plt.suptitle(f'Two electrically coupled neurons with uncorrelated inputs', size=16)
fig.tight_layout()
plt.savefig('crosscorrs.svg')
plt.show()
