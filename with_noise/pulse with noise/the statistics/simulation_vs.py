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
# red, orange, yellow, green, blue, purple

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

def correlations(sigma1, sigma2, sigma3, nb_iterations=1000) :
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

        """
        # Plot trials
        fig, ax = plt.subplots(2, 1, figsize=(12,5), sharey='row')
        ax[1].plot(t, x1, label='$V_{1}$', color='#aa3863')
        ax[1].plot(t, x2, label='$V_{2}$', color='#3b7d86')
        ax[0].plot(t, I1, label='$I_1$')
        ax[0].plot(t, I2, label='$I_2$')

        ax[0].legend(loc='upper right')
        ax[1].legend(loc='upper right')

        ax[0].set_title('Noisy input current trial, $\sigma=0.0025, I_{base}=1.5, \gamma=0.4, \\beta=0.1$')

        #plt.savefig('trial_example_.png', dpi=600)
        plt.show()
        """

    phis = np.array(phis) % 1
    print("phis ", phis)
    return phis

gamma, beta = 0.4, 0.1
Vth, Vr = 1, 0

dt = 0.001
Dtime = 75
maxtime = 80

# CORRELATED
sigma_corr = [[0., 0., 0.1], [0., 0., 0.15], [0., 0., 0.2], [0., 0., 0.25], [0., 0., 0.3], [0., 0., 0.4]]

phis1_corr = correlations(sigma_corr[0][0], sigma_corr[0][1], sigma_corr[0][2])
phis2_corr = correlations(sigma_corr[1][0], sigma_corr[1][1], sigma_corr[1][2])
phis3_corr = correlations(sigma_corr[2][0], sigma_corr[2][1], sigma_corr[2][2])
phis4_corr = correlations(sigma_corr[3][0], sigma_corr[3][1], sigma_corr[3][2])
phis5_corr = correlations(sigma_corr[4][0], sigma_corr[4][1], sigma_corr[4][2])
phis6_corr = correlations(sigma_corr[5][0], sigma_corr[5][1], sigma_corr[5][2])

# Generate data on phase differences
phis1_corr = pd.Series(phis1_corr)
phis2_corr = pd.Series(phis2_corr)
phis3_corr = pd.Series(phis3_corr)
phis4_corr = pd.Series(phis4_corr)
phis5_corr = pd.Series(phis5_corr)
phi65_corr = pd.Series(phis6_corr)

# UNCORRELATED
sigma_uncorr = [[0.1, 0.1, 0.], [0.15, 0.15, 0.], [0.2, 0.2, 0.], [0.25, 0.25, 0.], [0.3, 0.3, 0.], [0.4, 0.4, 0.]]

phis1_uncorr = correlations(sigma_uncorr[0][0], sigma_uncorr[0][1], sigma_uncorr[0][2])
phis2_uncorr = correlations(sigma_uncorr[1][0], sigma_uncorr[1][1], sigma_uncorr[1][2])
phis3_uncorr = correlations(sigma_uncorr[2][0], sigma_uncorr[2][1], sigma_uncorr[2][2])
phis4_uncorr = correlations(sigma_uncorr[3][0], sigma_uncorr[3][1], sigma_uncorr[3][2])
phis5_uncorr = correlations(sigma_uncorr[4][0], sigma_uncorr[4][1], sigma_uncorr[4][2])
phis6_uncorr = correlations(sigma_uncorr[5][0], sigma_uncorr[5][1], sigma_uncorr[5][2])

# Generate data on phase differences
phis1_uncorr = pd.Series(phis1_uncorr)
phis2_uncorr = pd.Series(phis2_uncorr)
phis3_uncorr = pd.Series(phis3_uncorr)
phis4_uncorr = pd.Series(phis4_uncorr)
phis5_uncorr = pd.Series(phis5_uncorr)
phi65_uncorr = pd.Series(phis6_uncorr)

fig, ax = plt.subplots(2, 3, figsize=(10, 5), sharex='col', sharey='row')
plt.ylim(0, 1000)

fig.suptitle('Distribution of phase difference, correlated vs uncorrelated')
ax[1, 0].set_xlabel('Phase Difference $\phi$')
ax[1, 1].set_xlabel('Phase Difference $\phi$')
ax[1, 2].set_xlabel('Phase Difference $\phi$')
ax[0, 0].set_ylabel(f'$\phi$ (t={Dtime})')
ax[1, 0].set_ylabel(f'$\phi$ (t={Dtime})')

ax[0, 0].set_ylim(0, 1000)
ax[1, 0].set_ylim(0, 1000)

text_plot1 = f'$\sigma_1$={sigma_uncorr[0][0]}, $\sigma_2$={sigma_uncorr[0][1]} and $\sigma_3$={sigma_corr[0][2]}.'
ax[0, 0].hist(phis1_corr, bins=np.linspace(0, 1, 22), alpha=0.5, edgecolor='k', color=c[2], rwidth=0.9, label='correlated')
ax[0, 0].hist(phis1_uncorr, bins=np.linspace(0, 1, 22), alpha=0.5, edgecolor='k', color=c[-3], rwidth=0.9, label='uncorrelated')
ax[0, 0].legend(loc='upper right', prop={'size': 8})
ax[0, 0].set_title(text_plot1, size=10)

text_plot2 = f'$\sigma_1$={sigma_uncorr[1][0]}, $\sigma_2$={sigma_uncorr[1][1]} and $\sigma_3$={sigma_corr[1][2]}.'
ax[0, 1].hist(phis2_corr, bins=np.linspace(0, 1, 22), alpha=0.5, edgecolor='k', color=c[2], rwidth=0.9, label='correlated')
ax[0, 1].hist(phis2_uncorr, bins=np.linspace(0, 1, 22), alpha=0.5, edgecolor='k', color=c[-3], rwidth=0.9, label='uncorrelated')
ax[0, 1].legend(loc='upper right', prop={'size': 8})
ax[0, 1].set_title(text_plot2, size=10)

text_plot3 = f'$\sigma_1$={sigma_uncorr[2][0]}, $\sigma_2$={sigma_uncorr[2][1]} and $\sigma_3$={sigma_corr[2][2]}.'
ax[0, 2].hist(phis3_corr, bins=np.linspace(0, 1, 22), alpha=0.5, edgecolor='k', color=c[2], rwidth=0.9, label='correlated')
ax[0, 2].hist(phis3_uncorr, bins=np.linspace(0, 1, 22), alpha=0.5, edgecolor='k', color=c[-3], rwidth=0.9, label='uncorrelated')
ax[0, 2].legend(loc='upper right', prop={'size': 8})
ax[0, 2].set_title(text_plot3, size=10)

text_plot4 = f'$\sigma_1$={sigma_uncorr[3][0]}, $\sigma_2$={sigma_uncorr[3][1]} and $\sigma_3$={sigma_corr[3][2]}.'
ax[1, 0].hist(phis4_corr, bins=np.linspace(0, 1, 22), alpha=0.5, edgecolor='k', color=c[2], rwidth=0.9, label='correlated')
ax[1, 0].hist(phis4_uncorr, bins=np.linspace(0, 1, 22), alpha=0.5, edgecolor='k', color=c[-3], rwidth=0.9, label='uncorrelated')
ax[1, 0].legend(loc='upper right', prop={'size': 8})
ax[1, 0].set_title(text_plot4, size=10)

text_plot5 = f'$\sigma_1$={sigma_uncorr[4][0]}, $\sigma_2$={sigma_uncorr[4][1]} and $\sigma_3$={sigma_corr[4][2]}.'
ax[1, 1].hist(phis5_corr, bins=np.linspace(0, 1, 22), alpha=0.5, edgecolor='k', color=c[2], rwidth=0.9, label='correlated')
ax[1, 1].hist(phis5_uncorr, bins=np.linspace(0, 1, 22), alpha=0.5, edgecolor='k', color=c[-3], rwidth=0.9, label='uncorrelated')
ax[1, 1].legend(loc='upper right', prop={'size': 8})
ax[1, 1].set_title(text_plot5, size=10)

text_plot6 = f'$\sigma_1$={sigma_uncorr[5][0]}, $\sigma_2$={sigma_uncorr[5][1]} and $\sigma_3$={sigma_corr[5][2]}.'
ax[1, 2].hist(phis6_corr, bins=np.linspace(0, 1, 22), alpha=0.5, edgecolor='k', color=c[2], rwidth=0.9, label='correlated')
ax[1, 2].hist(phis6_uncorr, bins=np.linspace(0, 1, 22), alpha=0.5, edgecolor='k', color=c[-3], rwidth=0.9, label='uncorrelated')
ax[1, 2].legend(loc='upper right', prop={'size': 8})
ax[1, 2].set_title(text_plot6, size=10)

plt.tight_layout()
#plt.savefig('un_vs_correlated.svg')
plt.show()
