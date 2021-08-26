from matplotlib import cm, rcParams
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib as matplotlib
from matplotlib import patheffects
import numpy as np
import math as math
import random as rand
import os, sys, csv
import pandas as pd

#matplotlib.pyplot.xkcd(scale=.5, length=100, randomness=2)
#rcParams['path.effects'] = [patheffects.withStroke(linewidth=.5)]

dW1, dW2, dW3 = 0, 0 ,0
np.random.seed() #42

def lif_euler(dt, v1, v2, I1, I2):
    return [v1 + dt*(-v1 + gamma*(v2-v1) + I1) , v2 + dt*(-v2 + gamma*(v1-v2) + I2) ]

def lif_euler_stoch(dt, v1, v2, I1, I2, s1, s2, s3):
    global dW1, dW2, dW3
    dW1 = s1*math.sqrt(dt)*np.random.randn()
    dW2 = s2*math.sqrt(dt)*np.random.randn()
    dW3 = s3*math.sqrt(dt)*np.random.randn()
    return [v1 + dt*(-v1 + gamma*(v2-v1) + I1) + dW1 + dW3, v2 + dt*(-v2 + gamma*(v1-v2) + I2) + dW2 + dW3]

gamma, beta = 0.1, 0.1
Vth, Vr = 1, 0

dt = 0.001
nb_iterations = 1
phis1, phis2 = [], []
maxtime = 40

for k in range(nb_iterations) :

    #v1_0, v2_0 = 0.7611728117817528, 0.1654125684129333 # Used XPPAUT to find ideal initial conditions s.t. we begin in antiphase with I = 1.4
    v1_0, v2_0 = 0.3764002759711251, 0.8546679415731656
    x1, x2 = [v1_0], [v2_0]
    t = [0]

    nb_spikes = 0
    I_baseline = 1.2
    I1 = [I_baseline]
    I2 = [I_baseline]

    while t[-1] < maxtime :
        t.append(t[-1]+dt)

        next_values= lif_euler_stoch(dt, x1[-1], x2[-1], I1[-1], I2[-1], 0., 0., 0.02) # example of common input
        I1.append(I_baseline + (dW1+dW3)/dt)
        I2.append(I_baseline + (dW2+dW3)/dt)

        if next_values[0] > 1 :
            x1.append(0)
            nb_spikes += 1
            if next_values[1] + gamma*beta > 1 :
                x2.append(0)
            else :
                x2.append(next_values[1]+gamma*beta)

        elif next_values[1] > 1 :
            x2.append(0)
            if next_values[0] + gamma*beta > 1 :
                x1.append(0)
            else :
                x1.append(next_values[0]+gamma*beta)

        else :
            x1.append(next_values[0])
            x2.append(next_values[1])

    # A spike occurs iff there was a reset
    for i in range(1,len(x1)) :
        if abs(x1[i]-x1[i-1]) > (Vth-Vr)/2 and x1[i] < 1 and x1[i-1] < 1:
            x1.insert(i, Vth+0.5)
            x2.insert(i, x2[i])
            I1.insert(i, I1[i])
            I2.insert(i, I2[i])
            t.insert(i, t[i])

    for i in range(1,len(x2)) :
        if abs(x2[i]-x2[i-1]) > (Vth-Vr)/2 and x2[i] < 1 and x2[i-1] < 1:
            x2.insert(i, Vth+0.5)
            x1.insert(i, x1[i])
            I1.insert(i, I1[i])
            I2.insert(i, I2[i])
            t.insert(i, t[i])

    fig, ax = plt.subplots(2, 1, figsize=(12,5), sharey='row', sharex='col')
    ax[1].plot(t, x1, label='$V_{1}$', color='#aa3863')
    ax[1].plot(t, x2, label='$V_{2}$', color='#3b7d86')

    if I1 == I2 :
        ax[0].plot(t, I1, label='$I$', color='#ef9f07', alpha=0.8)
    else:
        ax[0].plot(t, I1, label='$I_1$', color='#aa3863')
        ax[0].plot(t, I2, label='$I_2$', color='#3b7d86')

    ax[0].legend(loc='upper right', fontsize=10)
    ax[1].legend(loc='upper right', fontsize=10)

    #ax[0].set_title('Noisy input current trial, $\sigma=0.0025, I_{base}=1.5, \gamma=0.4, \\beta=0.1$')
    ax[0].set_title('Noisy input current trial, correlated stochastic input, $I_{mean}$=1.2, $\gamma$=0.1, $\\beta$=0.1', size=14)
    ax[1].set_xlabel('Time ($10^-2$ s)')

    plt.savefig('trial_example_dependent.svg')
    plt.show()
