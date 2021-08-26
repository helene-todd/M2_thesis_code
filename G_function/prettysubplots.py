from matplotlib import cm, rcParams
import matplotlib as matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import math as math
import random as rand

matplotlib.pyplot.xkcd(scale=.7, length=100, randomness=2) # Comment for something more serious

c = ['#aa3863', '#d97020', '#ef9f07', '#449775', '#3b7d86', '#581845']
matplotlib.rc('xtick', labelsize=11)
matplotlib.rc('ytick', labelsize=11)
#rcParams.update({'figure.autolayout': True})

fig, ax = plt.subplots(2, 3, figsize=(18,8.5), sharex=True)

def T(I):
    return math.log(I/(I-1))

def G(phi, I, gamma):
    if phi != 0 and phi != 1:
        return gamma*(2/T(I))*(phi*math.sinh((1-phi)*T(I)) - (1-phi)*math.sinh(phi*T(I))) + gamma*(beta/(I*T(I)*T(I)))*(math.exp(phi*T(I)) - math.exp((1-phi)*T(I)))
    else :
        return 0

vector_phi = np.linspace(0,1,1000)
zero_line = np.zeros(len(vector_phi))
gamma = 1

""" Varying I """

""" Subfigure 3 """
beta = 0.1
I = [1.6, 2]

ax[0, 2].plot(vector_phi, zero_line, linestyle='--', color='k')

k = 0
for current in I :
    vector_G = []
    for el in vector_phi:
        vector_G.append(G(el, current, gamma))
    vector_G = np.array(vector_G)
    ax[0, 2].plot(vector_phi, vector_G, label=f'$I$ = {current}', color = c[-1+(k-1)])
    k += 1

    # Adding dots to show stable / unstable states
    zero_crossings = np.where(np.diff(np.sign(vector_G-zero_line)))[0]
    for i in zero_crossings :
        if vector_G[i+1]-vector_G[i] > 0 :
            ax[0, 2].scatter(vector_phi[i], 0, s=40, facecolors='none', edgecolors='k', zorder=10)
        elif vector_G[i+1]-vector_G[i] < 0 :
            ax[0, 2].scatter(vector_phi[i], 0, s=40, color='k', zorder=10)

ax[0, 2].set_title(f'$\\beta$ = {beta}', size=16)
ax[0, 2].legend(loc = 'upper left', prop={'size': 11})

ticks = ax[0, 2].get_yticks()
ax[0, 2].set_yticks([ticks[0], 0, ticks[-1]])

""" Subfigure 1 """
beta = 0.
I = [1.1, 1.2, 1.3, 1.4]

ax[0, 0].plot(vector_phi, zero_line, linestyle='--', color='k')

k = 0
for current in I :
    vector_G = []
    for el in vector_phi:
        vector_G.append(G(el, current, gamma))
    vector_G = np.array(vector_G)
    ax[0, 0].plot(vector_phi, vector_G, label=f'$I$ = {current}', color = c[k])
    k += 1

    # Adding dots to show stable / unstable states
    zero_crossings = np.where(np.diff(np.sign(vector_G-zero_line)))[0]
    for i in zero_crossings :
        if vector_G[i+1]-vector_G[i] > 0 :
            ax[0, 0].scatter(vector_phi[i], 0, s=40, facecolors='none', edgecolors='k', zorder=10)
        elif vector_G[i+1]-vector_G[i] < 0 :
            ax[0, 0].scatter(vector_phi[i], 0, s=40, color='k', zorder=10)

ax[0, 0].set_ylabel('$G(\phi)$', size=14)

if beta == 0 : # Makes title more aesthetic
    beta = '0'

ax[0, 0].set_title(f'$\\beta$ = {beta}', size=16)
ax[0, 0].legend(prop={'size': 11})

ticks = ax[0, 0].get_yticks()
ax[0, 0].set_yticks([ticks[0], 0, ticks[-1]])

""" Subfigure 2 """
beta = 0.1
I = [1.1, 1.2, 1.3, 1.4]

ax[0, 1].plot(vector_phi, zero_line, linestyle='--', color='k')

k = 0
for current in I :
    vector_G = []
    for el in vector_phi:
        vector_G.append(G(el, current, gamma))
    vector_G = np.array(vector_G)
    ax[0, 1].plot(vector_phi, vector_G, label=f'$I$ = {current}', color = c[k])
    k += 1

    # Adding dots to show stable / unstable states
    zero_crossings = np.where(np.diff(np.sign(vector_G-zero_line)))[0]
    for i in zero_crossings :
        if vector_G[i+1]-vector_G[i] > 0 :
            ax[0, 1].scatter(vector_phi[i], 0, s=40, facecolors='none', edgecolors='k', zorder=10)
        elif vector_G[i+1]-vector_G[i] < 0 :
            ax[0, 1].scatter(vector_phi[i], 0, s=40, color='k', zorder=10)

if beta == 0 : # Makes title more aesthetic
    beta = '0'

ax[0, 1].set_title(f'$\\beta$ = {beta}', size=16)

ticks = ax[0, 1].get_yticks()
ax[0, 1].set_yticks([ticks[0], 0, ticks[-1]])


""" Varying beta """

""" Subfigure 6 """
beta = 0.8
I = 1.05

ax[1, 2].plot(vector_phi, zero_line, linestyle='--', color='k')

vector_G = []
for el in vector_phi:
    vector_G.append(G(el, I, gamma))
vector_G = np.array(vector_G)
ax[1, 2].plot(vector_phi, vector_G, label=f'$\\beta$ = {beta}', color = c[-2])

# Adding dots to show stable / unstable states
zero_crossings = np.where(np.diff(np.sign(vector_G-zero_line)))[0]
for i in zero_crossings :
    if vector_G[i+1]-vector_G[i] > 0 :
        ax[1, 2].scatter(vector_phi[i], 0, s=40, facecolors='none', edgecolors='k', zorder=10)
    elif vector_G[i+1]-vector_G[i] < 0 :
        ax[1, 2].scatter(vector_phi[i], 0, s=40, color='k', zorder=10)


ax[1, 2].set_xlabel('$\phi$', size=14)
ax[1, 2].set_title(f'$I$ = {I}', size=16)
ax[1, 2].legend(prop={'size': 11})

ticks = ax[1, 2].get_yticks()
ax[1, 2].set_yticks([ticks[0], 0, ticks[-1]])


""" Subfigure 5 """
betas = [0.1, 0.2, 0.3, 0.4]
I = 1.6

ax[1, 0].plot(vector_phi, zero_line, linestyle='--', color='k')

k = 0
for beta in betas :
    vector_G = []
    for el in vector_phi:
        vector_G.append(G(el, I, gamma))
    vector_G = np.array(vector_G)
    ax[1, 0].plot(vector_phi, vector_G, label=f'$\\beta$ = {beta}', color = c[k])
    k += 1

    # Adding dots to show stable / unstable states
    zero_crossings = np.where(np.diff(np.sign(vector_G-zero_line)))[0]
    for i in zero_crossings :
        if vector_G[i+1]-vector_G[i] > 0 :
            ax[1, 0].scatter(vector_phi[i], 0, s=40, facecolors='none', edgecolors='k', zorder=10)
        elif vector_G[i+1]-vector_G[i] < 0 :
            ax[1, 0].scatter(vector_phi[i], 0, s=40, color='k', zorder=10)

ax[1, 0].set_xlabel('$\phi$', size=14)
ax[1, 0].set_ylabel('$G(\phi)$', size=14)

ax[1, 0].set_title(f'$I$ = {I}', size=16)
ticks = ax[1, 0].get_yticks()
ax[1, 0].set_yticks([ticks[0], 0, ticks[-1]])
ax[1, 0].legend(loc='upper left', prop={'size': 11})


""" Subfigure 4 """
betas = [0.1, 0.2, 0.3, 0.4]
I = 1.05

ax[1, 1].plot(vector_phi, zero_line, linestyle='--', color='k')

k = 0
for beta in betas :
    vector_G = []
    for el in vector_phi:
        vector_G.append(G(el, I, gamma))
    vector_G = np.array(vector_G)
    ax[1, 1].plot(vector_phi, vector_G, label=f'$\\beta$ = {beta}', color = c[k])
    k += 1

    # Adding dots to show stable / unstable states
    zero_crossings = np.where(np.diff(np.sign(vector_G-zero_line)))[0]
    for i in zero_crossings :
        if vector_G[i+1]-vector_G[i] > 0 :
            ax[1, 1].scatter(vector_phi[i], 0, s=40, facecolors='none', edgecolors='k', zorder=10)
        elif vector_G[i+1]-vector_G[i] < 0 :
            ax[1, 1].scatter(vector_phi[i], 0, s=40, color='k', zorder=10)


ax[1, 1].set_xlabel('$\phi$', size=14)
ax[1, 1].set_title(f'$I$ = {I}', size=16)

ticks = ax[1, 1].get_yticks()
ax[1, 1].set_yticks([ticks[0], 0, ticks[-1]])
fig.suptitle('$G$-function for different values of $\\beta$ and $I$', size=20)

fig.tight_layout()
fig.subplots_adjust(hspace=0.25, top=0.88)

plt.savefig('allinone.svg')
plt.show()
