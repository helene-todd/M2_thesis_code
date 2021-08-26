from matplotlib import cm, rcParams
import matplotlib.pyplot as plt
import numpy as np
import math as math
import random as rand

""" G(phi) function in Rinzel & Lewis' article (2003) under weak coupling """
""" This is under weak coupling theory, although one can note that gamma only serves to scale the function """

c = ['#aa3863', '#d97020', '#ef9f07', '#449775', '#3b7d86']
rcParams.update({'figure.autolayout': True})

def T(I):
    return math.log(I/(I-1))

def G(phi, I, gamma):
    if phi != 0 and phi != 1:
        return gamma*(2/T(I))*(phi*math.sinh((1-phi)*T(I)) - (1-phi)*math.sinh(phi*T(I))) + gamma*(beta/(I*T(I)*T(I)))*(math.exp(phi*T(I)) - math.exp((1-phi)*T(I)))
    else :
        return 0

""" Varying Gamma """
""" Not very useful as we consider weak coupling, i.e. gamma close to 0 """
"""
gamma = [0.4, 0.3, 0.2, 0.1, 0.01]
beta = 0.1
I = 1.8

plt.figure(figsize=(8,5))
vector_phi = np.linspace(0,1,1000)
zero_line = np.zeros(len(vector_phi))
plt.plot(vector_phi, zero_line, color='black', linestyle='--')

k = 0
for g in gamma :
    vector_G = []
    for el in vector_phi:
        vector_G.append(G(el, I, g))
    vector_G = np.array(vector_G)
    plt.plot(vector_phi, vector_G, label=f'$\gamma = {g}$', color = c[k])
    k += 1

plt.xlabel('$\phi$', size=14)
plt.ylabel('$G(\phi)$', size=14)
plt.title(f'G function for $I={I}, \\beta={beta}$')

zero_crossings = np.where(np.diff(np.sign(vector_G-zero_line)))[0]
print(zero_crossings)

plt.legend(loc='upper left')
plt.savefig(f'G_function_range_gammas_I={I}.png', dpi=600)
plt.show()
plt.close()
"""
""" Varying I """

gamma = 0.001
beta = 0.1
I = [1.6]

plt.figure(figsize=(8,5))
vector_phi = np.linspace(0,1,1000)
zero_line = np.zeros(len(vector_phi))
plt.plot(vector_phi, zero_line, linestyle='--', color='k')

k = 0
for current in I :
    vector_G = []
    for el in vector_phi:
        vector_G.append(G(el, current, gamma))
    vector_G = np.array(vector_G)
    plt.plot(vector_phi, vector_G, label=f'$I = {current}$', color = c[k])
    k += 1

    # Adding dots to show stable / unstable states
    zero_crossings = np.where(np.diff(np.sign(vector_G-zero_line)))[0]
    print(zero_crossings)
    for i in zero_crossings :
        if vector_G[i+1]-vector_G[i] > 0 :
            plt.scatter(vector_phi[i], 0, s=40, facecolors='none', edgecolors='k', zorder=10)
        elif vector_G[i+1]-vector_G[i] < 0 :
            plt.scatter(vector_phi[i], 0, s=40, color='k', zorder=10)

plt.xlabel('$\phi$', size=14)
plt.ylabel('$G(\phi)$', size=14)

if beta == 0 : # Makes title more aesthetic
    beta = '0'

if len(I) == 1 :
    name = f'I_beta={beta}.svg'
    title = f'G-function for $I = {I[0]}$, with $\\beta = {beta}, \gamma = {gamma}$'
else :
    name = f'range_I_beta={beta}.svg'
    title = f'G-function for different values of $I$, with $\\beta = {beta}, \gamma = {gamma}$'

plt.title(title, size=16)
plt.legend()
plt.savefig(name)
plt.show()

""" Varying beta """

gamma = 0.001
betas = [0.8]
I = 1.05

plt.figure(figsize=(8,5))
vector_phi = np.linspace(0,1,1000)
zero_line = np.zeros(len(vector_phi))
plt.plot(vector_phi, zero_line, linestyle='--', color='k')

k = 0
for beta in betas :
    vector_G = []
    for el in vector_phi:
        vector_G.append(G(el, I, gamma))
    vector_G = np.array(vector_G)
    plt.plot(vector_phi, vector_G, label=f'$\\beta = {beta}$', color = c[k])
    k += 1

    # Adding dots to show stable / unstable states
    zero_crossings = np.where(np.diff(np.sign(vector_G-zero_line)))[0]
    print(zero_crossings)
    for i in zero_crossings :
        if vector_G[i+1]-vector_G[i] > 0 :
            plt.scatter(vector_phi[i], 0, s=40, facecolors='none', edgecolors='k', zorder=10)
        elif vector_G[i+1]-vector_G[i] < 0 :
            plt.scatter(vector_phi[i], 0, s=40, color='k', zorder=10)


plt.xlabel('$\phi$', size=14)
plt.ylabel('$G(\phi)$', size=14)

if len(betas) == 1 :
    name = f'beta_I={I}.svg'
    title = f'G-function for $\\beta = {beta}$, with $I = {I}, \gamma = {gamma}$'
else :
    name = f'range_beta_I={I}.svg'
    title = f'G-function for different values of $\\beta$, with $I = {I}, \gamma = {gamma}$'

plt.title(title, size=16)
plt.legend()
plt.savefig(name)
plt.show()
