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

# Run in terminal : python computing_corr [path to .dat file]

rcParams.update({'figure.autolayout': True})

p = argparse.ArgumentParser()
p.add_argument('file', type=str, help='.dat file to read data from.')
p.add_argument('nb_bins', type=int, default=20,
                    help='number of bins between two spikes (default is 20) - must be an even number.')
args = p.parse_args()

print('File to analyze data from :', args.file)

if args.nb_bins % 2 != 0 :
    print('Must choose an even number of bins. Aborting.')
    quit()
else :
    print('Number of bins between spikes :', args.nb_bins)

'''
if len(args.file) > 1 :
    print(args.)
    if args.file[1] % 2 != 0 :
        print('Number of bins must be an even number. Aborting.')
        quit()
'''
times = []
V1, V2 = [], []

Vth = 1
Vr = 0

# We begin by finding the average period T of V1
# in order to decide on the value of delta
# so that there are approx 20 time bins between each spike

times = []
V1, V2 = [], []
with open(args.file, newline='') as file:
    datareader = csv.reader(file, delimiter=' ')
    for row in datareader:
        times.append(float(row[0]))
        V1.append(float(row[1]))
        V2.append(float(row[2]))

spike_times_V1 = [times[i] for i in range(1,len(V1)) if abs(V1[i]-V1[i-1]) > (Vth-Vr)/2]
spike_times_V2 = [times[i] for i in range(1,len(V2)) if abs(V2[i]-V2[i-1]) > (Vth-Vr)/2]

T = 0
for k in range(1, len(spike_times_V1)):
    T += spike_times_V1[k]-spike_times_V1[k-1]
T = T/(len(spike_times_V1)-1)

delta = round(T/args.nb_bins, 2)
N = math.ceil(max(times)/delta)

print(T)
print(delta)

# We put the data in time bins of width delta
# to allow for correlation analysis
# and then put this data in a DataFrame
# to allow an easier manipulation

spikes_V1, spikes_V2 = np.zeros(N), np.zeros(N)
for i in spike_times_V1 :
    spikes_V1[math.floor(i/delta)] += 1
for i in spike_times_V2 :
    spikes_V2[math.floor(i/delta)] += 1

time_bins = np.array([delta*k for k in range(0, N)])

df = pd.DataFrame({'time': time_bins, 'V1': spikes_V1, 'V2': spikes_V2}, dtype=float)

kmin, kmax = -int(args.nb_bins/2), int(args.nb_bins/2) +1
for k in range(kmin,kmax) :
    if k > 0 :
        df[f'V1(t)V2(t+{k/args.nb_bins}T)'] = ''
    if k < 0 :
        df[f'V1(t)V2(t{k/args.nb_bins}T)'] = ''
    else :
        df[f'V1(t)V2(t)'] = ''

for k in range(kmin,kmax) :
    t = abs(k*delta)
    if k > 0 :
        df[f'V1(t)V2(t+{k/args.nb_bins}T)'] = df['V1']*df.V2.shift(k)
    if k < 0 :
        df[f'V1(t)V2(t{k/args.nb_bins}T)'] = df['V1']*df.V2.shift(k)
    else :
        df[f'V1(t)V2(t)'] = df['V1']*df['V2']

# Now we get S1, S2 etc.
S1 = df['V1'].div(delta).sum()/len(df['V1'])
S2 = df['V2'].div(delta).sum()/len(df['V2'])

s_df = pd.DataFrame({'T' : T, '<S1(t)>' : S1, '<S2(t)>' : S2}, index=[0], dtype=float)

for k in range(kmin,kmax) :
    if k > 0 :
        s_df[f'<S1(t)S2(t+{k/args.nb_bins}T)>'] = df[f'V1(t)V2(t+{k/args.nb_bins}T)'].div(delta**2).sum()/(len(df[f'V1(t)V2(t+{k/args.nb_bins}T)'])-df[f'V1(t)V2(t+{k/args.nb_bins}T)'].isna().sum())
    if k < 0 :
        s_df[f'<S1(t)S2(t{k/args.nb_bins}T)>'] = df[f'V1(t)V2(t{k/args.nb_bins}T)'].div(delta**2).sum()/(len(df[f'V1(t)V2(t{k/args.nb_bins}T)'])-df[f'V1(t)V2(t{k/args.nb_bins}T)'].isna().sum())
    else :
        s_df[f'<S1(t)S2(t)>'] = df[f'V1(t)V2(t)'].div(delta**2).sum()/(len(df[f'V1(t)V2(t)'])-df[f'V1(t)V2(t)'].isna().sum())

filename = args.file.rsplit('/', 1)[-1].replace('.dat', '')
path = args.file.rsplit('/', 1)[:-1][0]+str('/')

df.to_csv(f'{path+filename}_spikes.csv')
s_df.to_csv(f'{path+filename}_correlations.csv')

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(s_df)
