from scipy.optimize import fsolve
from matplotlib import cm, rcParams
from shapely import geometry
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math, csv, os
import warnings as warnings

""" ToDo : check if this is equivalent to the G-function for weak coupling """

c = ['#aa3863', '#d97020', '#ef9f07', '#449775', '#3b7d86']
rcParams.update({'figure.autolayout': True})

# equation of u1 that must be verified in the system with phi != 0, 1
def u1(phi, T, gamma, beta, I) :
    res = (np.exp((1-phi)*T)*(2-2*I) + 2*I)/(1 + np.exp(-2*gamma*(1-phi)*T)) - gamma*beta
    return res

# equation of u1 that must be verified in the system with phi != 0, 1
def u2(phi, T, gamma, beta, I) :
    res = (np.exp(phi*T)*(2-2*I) + 2*I)/(1 + np.exp(-2*gamma*phi*T)) - gamma*beta
    return res

def tosolve(i, phi, T, gamma, beta, I) :
    return [F_function(i[0], i[1], gamma, beta, I), theo_u2(phi, T, gamma, beta, I) - u2(i[0], i[1], gamma, beta, I)]

def F(T, phi, I) :
    res = -np.exp(-(1+2*gamma)*phi*T)*u2(phi,T, gamma, beta, I) - u1(phi,T, gamma, beta, I) +1 - np.exp(-(1+2*gamma)*phi*T)*gamma*beta
    #res = np.exp(-phi*T)*((u2(phi, T, gamma, beta, I)*gamma*beta)/2 - I - np.exp(-2*gamma*phi*T)*((u2(phi, T, gamma, beta, I)+gamma*beta)/2)) + I - u1(phi, T, gamma, beta, I)
    return res

min_I = 2
with open('gamma_0.4.dat', newline='') as file:
    datareader = csv.reader(file, delimiter=' ')
    for row in datareader:
        if float(row[1]) == 0.5 and float(row[0]) < min_I :
            min_I = float(row[0])

# Only makes sense if I > I_low i.e. when forks stop existing (see bifurcation diagram code or figures) !!!
nb_values = 501
I = np.linspace(min_I, 2, 501) # currents I
phi = np.linspace(0, 1, nb_values) # phis

gamma = 0.4
beta = 0.1

diffs = []
cycles = []

limits_I, limits_phi = [], []

warnings.simplefilter('error')
for k in range(len(phi)) :
    continuer = True

    temp = {}
    try:
        T = fsolve(F, np.ones(len(I)), args=(phi[k], I))
    except RuntimeWarning:
        try:
            T = fsolve(F, 5*np.ones(len(I)), args=(phi[k], I))
        except RuntimeWarning:
            continuer = False

    if continuer == True :
        T[T < 0] = 0
        for i in range(len(I)):

            if u1(phi[k], T[i], gamma, beta, I[i]) + gamma*beta >= 1 or u2(phi[k], T[i], gamma, beta, I[i]) + gamma*beta >= 1 :
                if phi[k] != 0 and phi[k] != 1 :
                    temp[I[i]] = phi[k]

        if len(temp) != 0 :
            limits_I.append(max(temp.keys()))
            limits_phi.append(temp[limits_I[-1]])

print(limits_I)
print(limits_phi)

plt.xlim(1, 2)
plt.ylim(0, 1)
plt.plot(limits_I, limits_phi)

np.savez('line.npz', I = limits_I, phi = limits_phi)
plt.show()
