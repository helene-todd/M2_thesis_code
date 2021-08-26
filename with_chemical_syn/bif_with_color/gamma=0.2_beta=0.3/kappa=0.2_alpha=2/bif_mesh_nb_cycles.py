from scipy.optimize import fsolve
from matplotlib import cm, rcParams
from shapely import geometry
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math, csv, os

c = ['#aa3863', '#d97020', '#ef9f07', '#449775', '#3b7d86']
rcParams.update({'figure.autolayout': True})

def A(phi, T, gamma, beta, kappa, alpha):
    if isinstance(T, np.ndarray) :
        term = - (kappa*alpha*alpha*np.exp(-alpha*T)/((1-np.exp(-alpha*T))*(1-alpha)**2)) * ((1-alpha)*T-1-np.exp(-phi*T*(1-alpha))*((1-alpha)*(1-phi)*T-1)) \
        - (kappa*alpha*alpha*np.exp(-2*alpha*T)*T/((1-np.exp(-alpha*T))**2 * (1-alpha))) * (1-np.exp(-phi*T*(1-alpha)))
    else :
        term = - (kappa*alpha*alpha*math.exp(-alpha*T)/((1-math.exp(-alpha*T))*(1-alpha)**2)) * ((1-alpha)*T-1-math.exp(-phi*T*(1-alpha))*((1-alpha)*(1-phi)*T-1)) \
        - (kappa*alpha*alpha*math.exp(-2*alpha*T)*T/((1-math.exp(-alpha*T))**2 * (1-alpha))) * (1-math.exp(-phi*T*(1-alpha)))
    return term

def B(phi, T, gamma, beta, kappa, alpha):
    if isinstance(phi, np.ndarray) :
        term = - (kappa*alpha*alpha*np.exp(-phi*T)/((1-np.exp(-alpha*T))*(1-alpha)**2)) * (np.exp(phi*T*(1-alpha))*((1-alpha)*phi*T-1)+1) \
        - (kappa*alpha*alpha*np.exp(-T*(alpha+phi))*T/((1-np.exp(-alpha*T))**2 * (1-alpha))) * (np.exp(phi*T*(1-alpha))-1)
    else :
        term = - (kappa*alpha*alpha*math.exp(-phi*T)/((1-math.exp(-alpha*T))*(1-alpha)**2)) * (math.exp(phi*T*(1-alpha))*((1-alpha)*phi*T-1)+1) \
        - (kappa*alpha*alpha*math.exp(-T*(alpha+phi))*T/((1-math.exp(-alpha*T))**2 * (1-alpha))) * (math.exp(phi*T*(1-alpha))-1)
    return term

def C(phi, T, gamma, beta, kappa, alpha):
    if isinstance(phi, np.ndarray) :
        term = - (kappa*alpha*alpha*np.exp(-alpha*T)/((1-np.exp(-alpha*T))*(1+2*gamma-alpha)**2)) * ((1+2*gamma-alpha)*T -1 -np.exp(-(1+2*gamma-alpha)*phi*T)*((1+2*gamma-alpha)*(1-phi)*T-1)) \
        - (kappa*alpha*alpha*np.exp(-2*alpha*T)*T/((1-np.exp(-alpha*T))**2 * (1+2*gamma-alpha))) * (1-np.exp(-phi*T*(1+2*gamma-alpha)))
    else :
        term = - (kappa*alpha*alpha*math.exp(-alpha*T)/((1-math.exp(-alpha*T))*(1+2*gamma-alpha)**2)) * ((1+2*gamma-alpha)*T -1 -math.exp(-(1+2*gamma-alpha)*phi*T)*((1+2*gamma-alpha)*(1-phi)*T-1)) \
        - (kappa*alpha*alpha*math.exp(-2*alpha*T)*T/((1-math.exp(-alpha*T))**2 * (1+2*gamma-alpha))) * (1-math.exp(-phi*T*(1+2*gamma-alpha)))
    return term

def D(phi, T, gamma, beta, kappa, alpha):
    if isinstance(phi, np.ndarray) :
        term = - (kappa*alpha*alpha*np.exp(-(1+2*gamma)*phi*T)/((1-np.exp(-alpha*T))*(1+2*gamma-alpha)**2)) * (np.exp(phi*T*(1+2*gamma-alpha))*((1+2*gamma-alpha)*phi*T-1) +1) \
        - (kappa*alpha*alpha*np.exp(-((1+2*gamma)*phi+alpha)*T)*T/((1-np.exp(-alpha*T))**2 * (1+2*gamma-alpha))) * (np.exp(phi*T*(1+2*gamma-alpha))-1)
    else :
        term = - (kappa*alpha*alpha*math.exp(-(1+2*gamma)*phi*T)/((1-math.exp(-alpha*T))*(1+2*gamma-alpha)**2)) * (math.exp(phi*T*(1+2*gamma-alpha))*((1+2*gamma-alpha)*phi*T-1) +1) \
        - (kappa*alpha*alpha*math.exp(-((1+2*gamma)*phi+alpha)*T)*T/((1-math.exp(-alpha*T))**2 * (1+2*gamma-alpha))) * (math.exp(phi*T*(1+2*gamma-alpha))-1)
    return term

# equation of u1 that must be verified in the system with phi != 0, 1
def u1(phi, T, gamma, beta, kappa, alpha, I) :
    if isinstance(phi, np.ndarray) :
        res = (np.exp((1-phi)*T)*(2*(1-I)-B(1-phi,T,gamma,beta,kappa,alpha)-A(1-phi,T,gamma,beta,kappa,alpha)-D(1-phi,T,gamma,beta,kappa,alpha)+C(1-phi,T,gamma,beta,kappa,alpha)) + 2*I)/(1 + np.exp(-2*gamma*(1-phi)*T)) - gamma*beta
    else :
        res = (math.exp((1-phi)*T)*(2*(1-I)-B(1-phi,T,gamma,beta,kappa,alpha)-A(1-phi,T,gamma,beta,kappa,alpha)-D(1-phi,T,gamma,beta,kappa,alpha)+C(1-phi,T,gamma,beta,kappa,alpha)) + 2*I)/(1 + math.exp(-2*gamma*(1-phi)*T)) - gamma*beta
    return res

# equation of u1 that must be verified in the system with phi != 0, 1
def u2(phi, T, gamma, beta, kappa, alpha, I) :
    if isinstance(phi, np.ndarray) :
        res = (np.exp(phi*T)*(2*(1-I)-A(phi,T,gamma,beta,kappa,alpha)-B(phi,T,gamma,beta,kappa,alpha)+C(phi,T,gamma,beta,kappa,alpha)-D(phi,T,gamma,beta,kappa,alpha)) + 2*I)/(1 + np.exp(-2*gamma*phi*T)) - gamma*beta
    else :
        res = (math.exp(phi*T)*(2*(1-I)-A(phi,T,gamma,beta,kappa,alpha)-B(phi,T,gamma,beta,kappa,alpha)+C(phi,T,gamma,beta,kappa,alpha)-D(phi,T,gamma,beta,kappa,alpha)) + 2*I)/(1 + math.exp(-2*gamma*phi*T)) - gamma*beta
    return res

# next theoretical value of u2
def theo_u2(phi, T, gamma, beta, kappa, alpha, I) :
    return math.exp(-T+phi*T)*((u1(phi, T,gamma,beta,kappa,alpha,I)+gamma*beta)/2 - I - math.exp(-2*gamma*(T-phi*T))*(u1(phi,T,gamma,beta,kappa,alpha,I)+gamma*beta)/2) + I + (-D(1-phi,T,gamma,beta,kappa,alpha)+C(1-phi,T,gamma,beta,kappa,alpha)+B(1-phi,T,gamma,beta,kappa,alpha)+A(1-phi,T,gamma,beta,kappa,alpha))/2

def if_eq(t, phi, Tm1, gamma, beta, kappa, alpha, I) :
    return math.exp(-t)*((theo_u2(phi,Tm1,gamma,beta,kappa,alpha,I)+gamma*beta)/2 - I + math.exp(-2*gamma*t)*(theo_u2(phi,Tm1,gamma,beta,kappa,alpha,I)+gamma*beta)/2) + I  + (A(phi,Tm1,gamma,beta,kappa,alpha)+B(phi,Tm1,gamma,beta,kappa,alpha)-C(phi,Tm1,gamma,beta,kappa,alpha)+D(phi,Tm1,gamma,beta,kappa,alpha))/2 - 1

# next theoretical value of u1
def theo_u1(phi, Tm1, gamma, beta, kappa, alpha, I) :
    T_spike = fsolve(if_eq, Tm1, args=(phi, Tm1, gamma, beta, kappa, alpha, I))
    res = math.exp(-T_spike)*((theo_u2(phi, Tm1, gamma, beta, kappa, alpha, I)+gamma*beta)/2 - I - math.exp(-2*gamma*T_spike)*(theo_u2(phi, Tm1, gamma, beta, kappa, alpha, I)+gamma*beta)/2) + I +(A(phi,T_spike,gamma,beta,kappa,alpha)+B(phi,T_spike,gamma,beta,kappa,alpha)+C(phi,T_spike,gamma,beta,kappa,alpha)-D(phi,T_spike,gamma,beta,kappa,alpha))/2
    return res

def F_function(phi, T, gamma, beta, kappa, alpha, I) :
    return -math.exp(-(1+2*gamma)*phi*T)*(u2(phi,T, gamma, beta, kappa, alpha, I)+gamma*beta) -D(phi,T,gamma,beta,kappa,alpha)+C(phi,T,gamma,beta,kappa,alpha) - u1(phi,T, gamma, beta, kappa, alpha, I) + 1

# Find phi such that v1(phi*T)-v2(phi*T)= u1-1
# Find T such that analytically computed next u2(phi_curr, T_curr) is equal to formula of u2 (in order to retieve T)
def tosolve(i, phi, T, gamma, beta, kappa, alpha, I) :
    return [F_function(i[0], i[1], gamma, beta, kappa, alpha, I), theo_u2(phi, T, gamma, beta, kappa, alpha, I) - u2(i[0], i[1], gamma, beta, kappa, alpha, I)]

def F(T, phi, I) :
    if isinstance(phi, np.ndarray) :
        res = -np.exp(-(1+2*gamma)*phi*T)*(u2(phi,T, gamma, beta, kappa, alpha, I)+gamma*beta) +C(phi,T,gamma,beta,kappa,alpha)-D(phi,T,gamma,beta,kappa,alpha) - u1(phi,T, gamma, beta, kappa, alpha, I) +1
    else :
        res = -math.exp(-(1+2*gamma)*phi*T)*(u2(phi,T, gamma, beta, kappa, alpha, I)+gamma*beta) +C(phi,T,gamma,beta,kappa,alpha)-D(phi,T,gamma,beta,kappa,alpha) - u1(phi,T, gamma, beta, kappa, alpha, I) +1
    return res

nb_values = 201
currents = np.linspace(1, 2, 201) # currents I
phi = np.linspace(0, 1, nb_values)# phis

gamma = 0.2
kappa = 0.2
beta = 0.3
alpha = 2

diffs = []
cycles = []

for I in currents :
    print('------')
    print(I)

    T = fsolve(F, np.ones(nb_values), args=(phi, I))

    T[T < 0] = T[np.where(T > 0)[0][0]]
    if I < 1.1 :
        T[T == 1] = 0

    #print('phi : ', phi[k],' T : ', T[k])

    diff = []
    nb_cycles = []

    for k in range(len(T)):
        continuer = True

        try :
            u1(phi[k], T[k], gamma, beta, kappa, alpha, I)
            u2(phi[k], T[k], gamma, beta, kappa, alpha, I)
        except ZeroDivisionError :
            nb_cycles.append(np.nan)
            continuer = False

        if continuer == True :

            if (u1(phi[k], T[k], gamma, beta, kappa, alpha, I) + gamma*beta >= 1 or u2(phi[k], T[k], gamma, beta, kappa, alpha, I) + gamma*beta >= 1) :
                nb_cycles.append(np.nan)

            elif theo_u1(phi[k], T[k], gamma, beta, kappa, alpha, I) + gamma*beta >= 1 or theo_u2(phi[k], T[k], gamma, beta, kappa, alpha, I) + gamma*beta >= 1 :
                if phi[k] <= 0.5 :
                    nb_cycles.append(1) #ok
                elif phi[k] > 0.5 :
                    nb_cycles.append(1) #ok

            elif phi[k] == 0 or phi[k] == 1 :
                nb_cycles.append(np.nan)

            else :
                try :
                    [phi_nxt, T_nxt] = fsolve(tosolve, [phi[k], T[k]], args=(phi[k], T[k], gamma, beta, kappa, alpha, I))
                except OverflowError :
                    nb_cycles[-1] += 1 #ok

                # If fsolve did not do its job properly (occurs for a few values)
                res = tosolve([phi_nxt, T_nxt], phi[k], T[k], gamma, beta, kappa, alpha, I)
                if abs(res[0]) > 10**(-4) or abs(res[1]) > 10**(-4) :
                    [phi_nxt, T_nxt] = fsolve(tosolve, [phi[k]-0.2, T[k]+0.1], args=(phi[k], T[k], gamma, beta, kappa, alpha, I))

                # Attempting to find a better solution by changing x0; this seems to work
                res = tosolve([phi_nxt, T_nxt], phi[k], T[k], gamma, beta, kappa, alpha, I)
                if abs(res[0]) > 10**(-4) or abs(res[1]) > 10**(-4) :
                    nb_cycles.append(np.nan)

                elif (phi[k] < 0.5 and phi_nxt-phi[k] < 0) or (phi[k] > 0.5 and phi_nxt-phi[k] > 0) :
                    nb_cycles.append(1) #ok
                    phi_curr, T_curr = phi[k], T[k]

                    while phi_nxt >= 0 and phi_nxt <= 1 and phi_nxt != phi_curr and ((phi[k] < 0.5 and phi_nxt-phi[k] < 0) or (phi[k] > 0.5 and phi_nxt-phi[k] > 0)):

                        #print("phi curr", phi_curr, T_curr)
                        #print("phi nxt", phi_nxt, T_nxt)

                        if  phi_nxt >= 0 and phi_nxt <= 1 :
                            nb_cycles[-1] += 1 #maybe

                        if nb_cycles[-1] >= 10**2 :
                            break

                        phi_curr, T_curr = phi_nxt, T_nxt

                        try:
                            [phi_nxt, T_nxt] = fsolve(tosolve, [phi_curr, T_curr], args=(phi_curr, T_curr, gamma, beta, kappa, alpha, I))
                        except OverflowError:
                            nb_cycles[-1] += 1 #ok
                            break
                else :
                    nb_cycles.append(np.nan)

    cycles.append(nb_cycles)

cycles = np.array(cycles)
with np.printoptions(threshold=np.inf):
    #print('nb_cycles ', nb_cycles)
    print('cycles ', cycles)
    print('cycles shape', cycles.shape)
    print('phi ', phi)
    print('currents ', currents)

# clev = np.arange(diffs.min(),diffs.max(),.001)
#plt.plot(phi, nb_cycles)
h = plt.contourf(currents, phi, np.transpose(cycles), cmap="viridis")
plt.show()

np.savez('mesh_cycles.npz', I = currents, phi = phi, cycles = np.transpose(cycles))
