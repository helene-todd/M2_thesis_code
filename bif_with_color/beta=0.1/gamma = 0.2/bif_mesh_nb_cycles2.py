from scipy.optimize import fsolve
from matplotlib import cm, rcParams
from shapely import geometry
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math, csv, os

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

# next theoretical value of u2
def theo_u2(phi, T, gamma, beta, I) :
    return math.exp(-T+phi*T)*((u1(phi, T, gamma, beta, I)+gamma*beta)/2 - I - math.exp(-2*gamma*(T-phi*T))*(u1(phi, T, gamma, beta, I)+gamma*beta)/2) + I

def if_eq(t, phi, Tm1, gamma, beta, I) :
    return math.exp(-t)*((theo_u2(phi, Tm1, gamma, beta, I)+gamma*beta)/2 - I + math.exp(-2*gamma*t)*(theo_u2(phi, Tm1, gamma, beta, I)+gamma*beta)/2) + I - 1

# next theoretical value of u1
def theo_u1(phi, Tm1, gamma, beta, I) :
    T = fsolve(if_eq, 1, args=(phi, Tm1, gamma, beta, I))
    return math.exp(-T)*((theo_u2(phi, Tm1, gamma, beta, I)+gamma*beta)/2 - I - math.exp(-2*gamma*T)*(theo_u2(phi, Tm1, gamma, beta, I)+gamma*beta)/2) + I

def F_function(phi, T, gamma, beta, I) :
    return -math.exp(-(1+2*gamma)*phi*T)*u2(phi,T, gamma, beta, I) - u1(phi,T, gamma, beta, I) +1 - math.exp(-(1+2*gamma)*phi*T)*gamma*beta

def tosolve(i, phi, T, gamma, beta, I) :
    return [F_function(i[0], i[1], gamma, beta, I), theo_u2(phi, T, gamma, beta, I) - u2(i[0], i[1], gamma, beta, I)]

def F(T, phi, I) :
    res = -np.exp(-(1+2*gamma)*phi*T)*u2(phi,T, gamma, beta, I) - u1(phi,T, gamma, beta, I) +1 - np.exp(-(1+2*gamma)*phi*T)*gamma*beta
    return res

min_I = 2
with open('gamma_0.2.dat', newline='') as file:
    datareader = csv.reader(file, delimiter=' ')
    for row in datareader:
        if float(row[1]) == 0.5 and float(row[0]) < min_I :
            min_I = float(row[0])

# Only makes sense if I > I_low i.e. when forks stop existing (see bifurcation diagram code or figures) !!!
nb_values = 101
I = np.linspace(min_I, 2, nb_values) # currents I
phi = np.linspace(0, 1, nb_values) # phis

gamma = 0.2
beta = 0.1

diffs = []
cycles = []

#epsilon = 10**(-2)

for k in range(len(phi)) :

    T = fsolve(F, np.ones(nb_values), args=(phi[k]*np.ones(len(phi)), I))
    T[T < 0] = T[np.where(T > 0)[0][0]]
    print(phi[k])

    diff = []
    nb_cycles = []
    print('---')

    for i in range(len(I)):

        if u1(phi[k], T[k], gamma, beta, I[i]) + gamma*beta >= 1 or u2(phi[k], T[k], gamma, beta, I[i]) + gamma*beta >= 1 :
            nb_cycles.append(0)

        elif theo_u1(phi[k], T[k], gamma, beta, I[i]) + gamma*beta >= 1 or theo_u2(phi[k], T[k], gamma, beta, I[i]) + gamma*beta >= 1 or phi[k] == 0 or phi[k] == 1:
            if phi[k] <= 0.5 :
                nb_cycles.append(1)
            elif phi[k] > 0.5 :
                nb_cycles.append(1)

        else :

            try:
                [phi_nxt, T_nxt] = fsolve(tosolve, [phi[k], T[k]], args=(phi[k], T[k], gamma, beta, I[i]))
            except OverflowError:
                nb_cycles.append(0)
                phi_nxt = -1

            if (phi[k] < 0.5 and phi_nxt-phi[k] < 0 and phi_nxt != -1) or (phi[k] > 0.5 and phi_nxt-phi[k] > 0 and phi_nxt != -1) :
                nb_cycles.append(0)
                phi_curr, T_curr = phi[k], T[k]
                #print('---')

                while phi_nxt >= 0 and phi_nxt <= 1 and phi_nxt != phi_curr:
                    #print("phi curr", phi_curr, T_curr)
                    #print("phi nxt", phi_nxt, T_nxt)
                    if  phi_nxt >= 0 and phi_nxt <= 1 :
                        nb_cycles[-1] += 1

                    phi_curr, T_curr = phi_nxt, T_nxt

                    try:
                        [phi_nxt, T_nxt] = fsolve(tosolve, [phi_curr, T_curr], args=(phi_curr, T_curr, gamma, beta, I[i]))
                    except OverflowError:
                        nb_cycles[-1] += 1
                        break

            elif phi_nxt != -1 :
                nb_cycles.append(np.nan)

    cycles.append(nb_cycles)

cycles = np.array(cycles)
with np.printoptions(threshold=np.inf):
    print('cycles ', cycles)
    print('phi ', phi)
    print('currents ', I)

h = plt.contourf(I, phi, cycles, cmap="viridis")
plt.show()

np.savez('mesh_cycles.npz', I = I, phi = phi, cycles = cycles)
