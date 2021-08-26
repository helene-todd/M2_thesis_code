from matplotlib import cm, rcParams
import matplotlib.pyplot as plt
from shapely import geometry
import numpy as np
import pandas as pd
import math as math
import random as rand
import os
import csv
import argparse

# Run in terminal : python plotting_corr [path to .csv file with correlation data from computing_corr.py]

c = ['#aa3863', '#d97020', '#ef9f07', '#449775', '#3b7d86', '#5443a3']
rcParams.update({'figure.autolayout': True})

p = argparse.ArgumentParser()
p.add_argument('files', type=str, nargs='*')
args = p.parse_args()

print(args.files[0])

filename = args.files[0].rsplit('/', 1)[-1].replace('_correlations.csv', '')
path = args.files[0].rsplit('/', 1)[:-1][0]+str('/')
params_path = args.files[0].replace(args.files[0].rsplit('/')[-1], '')

params = open(os.path.join(params_path, 'parameters.txt'), 'r')
print(params)
lines = params.readlines()

I, gamma, beta, s1, s2 = 0, 0, 0, 0, 0
for line in lines:
    el = line.replace(' ', '').replace('\n', '').split('=')
    if el[0] == 'I':
        I = el[1]
    if el[0] == 'gamma':
        gamma = el[1]
    if el[0] == 'beta':
        beta = el[1]
    if el[0] == 's1':
        s1 = el[1]
    if el[0] == 's2':
        s2 = el[1]

params.close()

with open(args.files[0], newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    row1 = next(reader)
    row2 = next(reader)
    corr = row2[4:]
    T = row2[1] # just in case we want it displayed

corr = [float(i) for i in corr]

t = np.linspace(-0.5, 0.5, len(corr))
plt.plot(t, corr, marker='o', linestyle='-', markersize=8, color=c[-3], alpha=0.5, linewidth=1.2)
plt.xticks([-0.5, -0.25, 0, 0.25, 0.5])
plt.xlabel('$\\tau/T$', size=12)
plt.ylabel('cross-correlation', size=12)


title = f'Two electrically coupled neurons with uncorrelated inputs \n $I = {I} + {s1}W_1, I_2 = {I} + {s2}W_2$'
plt.title(title, size=13.5)

plt.savefig(f'{path+filename}_correlation.svg')
plt.show()
