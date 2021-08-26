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

# TO DO : Rewrite this code to make it more readable.
# USAGE : Run in terminal  "python dat_to_py.py gamma_0.4.dat stable1.dat stable2.dat"

plt.rcParams['axes.xmargin'] = 0
#plt.rcParams['axes.facecolor'] = 'black'
#matplotlib.pyplot.xkcd(scale=.4, length=100, randomness=2)
fig = plt.figure(figsize=(10, 7))

p = argparse.ArgumentParser()
p.add_argument('files', type=str, nargs=3)
p.add_argument('mesh', type=str, nargs='*')
args = p.parse_args()

def row_count(filename):
    with open(filename) as in_file:
        return sum(1 for _ in in_file)

c = ['#aa3863', '#3b7d86']
s = ['-', '--']


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

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


Imin, Imax = 2, 0
for l in range(len(I)) :
    for k in range(len(I[l])) :
        if phi[l][k] not in [0, 1, 0.5] and I[l][k] > Imax :
            Imax = I[l][k]
        if phi[l][k] == 0.5 and stability[l] == 1 and I[l][k] < Imin :
            Imin = I[l][k]

if len(args.mesh) > 0 :
    data = np.load('mesh_cycles.npz')
    #clev = np.arange(data['diffs'].min(),data['cycles'].max(),10**(-4))
    #h = plt.contourf(data['I'], data['phi'], data['cycles'], clev, cmap="viridis")
    #cmap = plt.get_cmap('viridis')
    #new_cmap = truncate_colormap(cmap, 0., 1)
    min_val = data['cycles'][np.invert(np.isnan(data['cycles']))].min()
    max_val = data['cycles'][np.invert(np.isnan(data['cycles']))].max()
    #plt.pcolormesh(data['I'], data['phi'], data['cycles'], cmap='viridis', shading='smooth', edgecolors=None, norm = colors.PowerNorm(gamma=0.1, vmin=min_val,vmax=max_val))
    max_val = 14364.0
    max_val = 10**2
    im = plt.pcolormesh(data['I'], data['phi'], data['cycles'], cmap='viridis', shading='smooth', edgecolors=None, norm = colors.SymLogNorm(linthresh=0.03, linscale=0.03, vmin=min_val, vmax=max_val, base=10))
    #plt.colorbar()
    cbar = fig.colorbar(im)
    cbar.ax.set_yticklabels([t.get_text() for t in cbar.ax.get_yticklabels()[:-1]]+['$>$'+cbar.ax.get_yticklabels()[-1].get_text()])
    cbar.set_label('Number of cycles to converge towards synchrony', labelpad=20, fontsize=12)

else :
    plt.axvspan(Imin, Imax, facecolor='0.2', alpha=0.1)

for k in range(len(I)) :
    if stability[k] == 1 :
        plt.plot(I[k], phi[k], color='black', linewidth=2, linestyle=s[stability[k]-1], label='stable')
    if stability[k] == 2 :
        plt.plot(I[k], phi[k], color='black', linewidth=2, linestyle=s[stability[k]-1], label='unstable')


# the delimiter line
data = np.load('line.npz')

middle = 0
for k in range(1,len(data['I'])) :
    if data['I'][k-1] < data['I'][k] :
        middle = k
        break

#print(data['I'][:middle])
#print(data['I'][middle:])
#print(data['phi'][:middle])
#print(data['phi'][middle:])

x = np.append(data['I'][:middle], 1)
y1 = np.zeros(len(x))
y2 = np.append(data['phi'][:middle], 0.5)
matplotlib.pyplot.fill_between(x, y1, y2=y2, hatch='//', facecolor='#C1E1C1', alpha=0.6)

x = np.append(1, data['I'][middle:])
print(x)
y1 = np.ones(len(x))
y2 = np.append(0.5, data['phi'][middle:])
print(y2)
matplotlib.pyplot.fill_between(x, y1, y2=y2, hatch='//', facecolor='#C1E1C1', alpha=0.6, label='non-physical')

#matplotlib.pyplot.fill_between([1]+data['I'][:middle], np.zeros(len(data['phi'][:middle])), y2=[0.5]+data['phi'][:middle], hatch='//', facecolor='#C1E1C1', alpha=1)
#matplotlib.pyplot.fill_between(data['I'][middle:], np.ones(len(data['phi'][middle:])), y2= data['phi'][middle:], hatch='//', facecolor='#C1E1C1', alpha=1, label='non-physical')

# plt.plot(data['I'], data['phi'], color='#808080', linestyle='-.', alpha=0.5)
# plt.plot([min(data['I']), min(data['I'])], [0, 1], color='#808080', linestyle='-.', alpha=0.5)

# plt.plot([Imax, Imax], [0, 1], color='k', linestyle=':')

# regime delimiter to make things more visual
plt.ylim(-0.05, 1.05)
plt.xlim(1, 2)

plt.title('Bifurcation diagram of two coupled neurons, $\gamma$=0.4, $\\beta$=0.1', fontsize=17, pad=15)
plt.xlabel('Current $I$', size=14)
plt.ylabel('Phase Difference $\phi$', size=14)

# remove duplicate legend
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='upper right', bbox_to_anchor=(1, 0.95), fontsize=11)

plt.savefig('bif_with_cv_speed.svg')
plt.show()
