from matplotlib import cm, rcParams
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib as matplotlib
import numpy as np
import math as math
import random as rand
import os, sys, csv
import pandas as pd

#matplotlib.pyplot.xkcd(scale=.5, length=100, randomness=2)
c = ['#aa3863', '#d97020', '#ef9f07', '#449775', '#3b7d86', '#5443a3']

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

def correlations(sigma1, sigma2, sigma3, nb_iterations=10) :
    phis = []

    for k in range(nb_iterations) :
        #v1_0, v2_0 = 0.7611728117817528, 0.1654125684129333 # Used XPPAUT to find ideal initial conditions s.t. we begin in antiphase with I = 1.4
        v1_0, v2_0 = 0.3764002759711251, 0.8546679415731656
        x1, x2 = [v1_0], [v2_0]
        t = [0]

        nb_spikes = 0
        I_baseline = 1.5
        I1, I2 = [I_baseline], [I_baseline]
        pulse_start, pulse_duration = 0, 0.2
        begin_pulse = True

        while t[-1] < maxtime :
            t.append(t[-1]+dt)

            if nb_spikes == 10 and begin_pulse :
                pulse_start = t[-1]
                begin_pulse = False

            if nb_spikes >= 10 and t[-1] < pulse_start + pulse_duration :
                next_values= lif_euler_stoch(dt, x1[-1], x2[-1], I1[-1], I2[-1], sigma1, sigma2, sigma3)
                I1.append(I_baseline + (dW1+dW3)/dt)
                I2.append(I_baseline + (dW2+dW3)/dt)

            else :
                I1.append(I_baseline)
                I2.append(I_baseline)
                next_values = lif_euler(dt, x1[-1], x2[-1], I1[-1], I2[-1])

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

        # Spike times
        spike_times, k = [], 0
        for i in range(1, len(t)) :
            if abs(x1[i]-x1[i-1]) > (Vth-Vr)/2 and t[i] >= Dtime :
                spike_times.append(t[i])
                k = i
                break
        for i in range(k, len(t)) :
            if abs(x2[i]-x2[i-1]) > (Vth-Vr)/2 :
                spike_times.append(t[i])
                k = i
                break
        for i in range(k, len(t)) :
            if abs(x1[i+1]-x1[i]) > (Vth-Vr)/2 :
                spike_times.append(t[i])
                break

        phis.append((spike_times[2] - spike_times[1])/(spike_times[2] - spike_times[0]))

        # Plot trials
        fig, ax = plt.subplots(2, 1, figsize=(12,5), sharey='row')
        ax[1].plot(t, x1, label='$V_{1}$', color='#aa3863')
        ax[1].plot(t, x2, label='$V_{2}$', color='#3b7d86')
        ax[0].plot(t, I1, label='$I_1$', color='#5443a3')
        ax[0].plot(t, I2, label='$I_2$', color='#5443a3')

        ax[0].legend(loc='upper right', fontsize=10)
        ax[1].legend(loc='upper right', fontsize=10)

        ax[0].set_title('Noisy input current trial, $\sigma=0.25, I_{base}=1.5, \gamma=0.4, \\beta=0.1$', size=14)

        plt.savefig('trial_example_pulse.svg')
        plt.show()


    phis = np.array(phis) % 1
    print("phis ", phis)
    return phis

gamma, beta = 0.4, 0.1
Vth, Vr = 1, 0

dt = 0.001
Dtime = 40
maxtime = 45

sigma = [[0.1, 0.1, 0.], [0.15, 0.15, 0.], [0.2, 0.2, 0.], [0.25, 0.25, 0.], [0.3, 0.3, 0.], [0.4, 0.4, 0.]]

#phis1 = correlations(sigma[0][0], sigma[0][1], sigma[0][2])
#phis2 = correlations(sigma[1][0], sigma[1][1], sigma[1][2])
#phis3 = correlations(sigma[2][0], sigma[2][1], sigma[2][2])
phis4 = correlations(sigma[3][0], sigma[3][1], sigma[3][2])
#phis5 = correlations(sigma[4][0], sigma[4][1], sigma[4][2])
#phis6 = correlations(sigma[5][0], sigma[5][1], sigma[5][2])
"""
# Generate data on phase differences
phis1 = pd.Series(phis1)
phis2 = pd.Series(phis2)
phis3 = pd.Series(phfontis3)
phis4 = pd.Series(phis4)
phis5 = pd.Series(phis5)
phi65 = pd.Series(phis6)

fig, ax = plt.subplots(2, 3, figsize=(10, 5), sharex='col', sharey='row')
plt.ylim(0, 1000)

fig.suptitle('Distribution of phase difference, uncorrelated')
ax[1, 0].set_xlabel('Phase Difference $\phi$')
ax[1, 1].set_xlabel('Phase Difference $\phi$')
ax[1, 2].set_xlabel('Phase Difference $\phi$')
ax[0, 0].set_ylabel(f'$\phi$ (t={Dtime})')
ax[1, 0].set_ylabel(f'$\phi$ (t={Dtime})')

ax[0, 0].set_ylim(0, 1000)
ax[1, 0].set_ylim(0, 1000)

ax[0, 0].hist(phis1, bins=np.linspace(0, 1, 22), alpha=0.5, edgecolor='k', color=c[0], rwidth=0.9, label=f'$\sigma_1$={sigma[0][0]},\n$\sigma_2$={sigma[0][1]},\n$\sigma_3$={sigma[0][2]}.') #, grid=False, edgecolor='black', alpha=0.5, align='left', rwidth=0.9, color='#aa3863', label='$\sigma$=0.15') # color='#bfe6bd' # int(math.sqrt(nb_iterations))
ax[0, 0].legend(loc='upper right', prop={'size': 8})

ax[0, 1].hist(phis2, bins=np.linspace(0, 1, 22), alpha=0.5, edgecolor='k', color=c[1], rwidth=0.9, label=f'$\sigma_1$={sigma[1][0]},\n$\sigma_2$={sigma[1][1]},\n$\sigma_3$={sigma[1][2]}.') #, grid=False, edgecolor='black', alpha=0.5, align='left', rwidth=0.9, color='#3b7d86', label='$\sigma$=0.05') # color='#dcacd2' # '#607c8e'
ax[0, 1].legend(loc='upper right', prop={'size': 8})

ax[0, 2].hist(phis3, bins=np.linspace(0, 1, 22), alpha=0.5, edgecolor='k', color=c[2], rwidth=0.9, label=f'$\sigma_1$={sigma[2][0]},\n$\sigma_2$={sigma[2][1]},\n$\sigma_3$={sigma[2][2]}.')
ax[0, 2].legend(loc='upper right', prop={'size': 8})

ax[1, 0].hist(phis4, bins=np.linspace(0, 1, 22), alpha=0.5, edgecolor='k', color=c[3], rwidth=0.9, label=f'$\sigma_1$={sigma[3][0]},\n$\sigma_2$={sigma[3][1]},\n$\sigma_3$={sigma[3][2]}.')
ax[1, 0].legend(loc='upper right', prop={'size': 8})

ax[1, 1].hist(phis5, bins=np.linspace(0, 1, 22), alpha=0.5, edgecolor='k', color=c[4], rwidth=0.9, label=f'$\sigma_1$={sigma[4][0]},\n$\sigma_2$={sigma[4][1]},\n$\sigma_3$={sigma[4][2]}.')
ax[1, 1].legend(loc='upper right', prop={'size': 8})

ax[1, 2].hist(phis6, bins=np.linspace(0, 1, 22), alpha=0.5, edgecolor='k', color=c[5], rwidth=0.9, label=f'$\sigma_1$={sigma[5][0]},\n$\sigma_2$={sigma[5][1]},\n$\sigma_3$={sigma[5][2]}.')
ax[1, 2].legend(loc='upper right', prop={'size': 8})

plt.tight_layout()
plt.savefig('uncorrelated.svg')
plt.show()
"""
"""
#plt.title('Phase diff. distribution of LIF neurons with noisy input current, uncorrelated')
plt.title('Distribution of phase diff., uncorrelated')
plt.xlabel('Phase Difference $\phi$')
plt.ylabel(f'$\phi$ (t={Dtime})')
plt.legend()
plt.tight_layout()
"""
#plt.savefig('histogram_dep.png', dpi=600)
