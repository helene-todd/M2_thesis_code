from scipy.optimize import fsolve
from matplotlib import cm, rcParams
import matplotlib as matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import math as math
import random as rand

c = ['#aa3863', '#d97020', '#ef9f07', '#449775', '#3b7d86', '#581845']
matplotlib.rc('xtick', labelsize=11)
matplotlib.rc('ytick', labelsize=11)

#rcParams.update({'figure.autolayout': True})
matplotlib.pyplot.xkcd(scale=.7, length=100, randomness=2)

fig, ax = plt.subplots(2, 3, figsize=(18,8), sharex=True)

def findIntersection(contour1,contour2):
    p1 = contour1.collections[0].get_paths()[0]
    v1 = p1.vertices
    p2 = contour2.collections[0].get_paths()[0]
    v2 = p2.vertices

    if len(v1)> 1 and len(v2)>1 :
        poly1 = geometry.LineString(v1)
        poly2 = geometry.LineString(v2)
        intersection = poly1.intersection(poly2)
        return intersection

    else :
        return None

# equation of u1 that must be verified in the system with phi != 0, 1
def u1(phi, T, gamma, beta) :
    if isinstance(phi, np.ndarray) :
        res = (np.exp((1-phi)*T)*(2-2*I) + 2*I)/(1 + np.exp(-2*gamma*(1-phi)*T)) - gamma*beta
    else :
        res = (math.exp((1-phi)*T)*(2-2*I) + 2*I)/(1 + math.exp(-2*gamma*(1-phi)*T)) - gamma*beta
    return res

# equation of u1 that must be verified in the system with phi != 0, 1
def u2(phi, T, gamma, beta) :
    if isinstance(phi, np.ndarray) :
        res = (np.exp(phi*T)*(2-2*I) + 2*I)/(1 + np.exp(-2*gamma*phi*T)) - gamma*beta
    else :
        res = (math.exp(phi*T)*(2-2*I) + 2*I)/(1 + math.exp(-2*gamma*phi*T)) - gamma*beta
    return res

# next theoretical value of u2
def theo_u2(phi, T, gamma, beta) :
    return math.exp(-T+phi*T)*((u1(phi, T, gamma, beta)+gamma*beta)/2 - I - math.exp(-2*gamma*(T-phi*T))*(u1(phi, T, gamma, beta)+gamma*beta)/2) + I

def if_eq(t, phi, Tm1, gamma, beta) :
    return math.exp(-t)*((theo_u2(phi, Tm1, gamma, beta)+gamma*beta)/2 - I + math.exp(-2*gamma*t)*(theo_u2(phi, Tm1, gamma, beta)+gamma*beta)/2) + I - 1

# next theoretical value of u1
# works better than relying on fsolve for when beta*gamma induces instant synchronization
def theo_u1(phi, Tm1, gamma, beta) :
    T = fsolve(if_eq, 1, args=(phi, Tm1, gamma, beta))
    return math.exp(-T)*((theo_u2(phi, Tm1, gamma, beta)+gamma*beta)/2 - I - math.exp(-2*gamma*T)*(theo_u2(phi, Tm1, gamma, beta)+gamma*beta)/2) + I

def I_function(T, gamma, beta) :
    return -1/2*(beta*gamma*(math.exp((T*gamma + 1/2*T)) + math.exp((1/2*T))) - 2*math.exp((T*gamma + T)) + math.exp((T*gamma + 1/2*T)) - math.exp((1/2*T)))/(math.exp((T*gamma + T)) - math.exp((T*gamma + 1/2*T)) + math.exp((1/2*T)) - 1)

def F_function(phi, T, gamma, beta) :
    return -math.exp(-(1+2*gamma)*phi*T)*u2(phi,T, gamma, beta) - u1(phi,T, gamma, beta) +1 - math.exp(-(1+2*gamma)*phi*T)*gamma*beta

def G_function(phi, T, gamma, beta) :
    return 2*I*(1-math.exp(-(1-phi)*T)) + u1(phi,T, gamma, beta)*math.exp(-(1-phi)*T) -1 -u2(phi,T, gamma, beta) +gamma*beta*math.exp(-(1-phi)*T)

def tosolve(i, phi, T, gamma, beta) :
    return [F_function(i[0], i[1], gamma, beta), theo_u2(phi, T, gamma, beta) - u2(i[0], i[1], gamma, beta)]


""" line 0 : Varying beta """

I = 1.1
gamma = 0.1
betas = [0.1, 0.15, 0.2, 0.25]

all_diffs = []
all_phis = []

for beta in betas :

    vector_x = []
    vector_g = []

    xrange = np.linspace(0, 1, 1001) # phi
    yrange = np.linspace(0, 20, 1001) # T
    X, Y = np.meshgrid(xrange,yrange)

    phi = []
    T = []

    F = - np.exp(-(1+2*gamma)*X*Y)*u2(X,Y, gamma, beta) - u1(X,Y, gamma, beta) +1 - np.exp(-(1+2*gamma)*X*Y)*gamma*beta
    G = 2*I*(1-np.exp(-(1-X)*Y)) + u1(X,Y, gamma, beta)*np.exp(-(1-X)*Y) -1 -u2(X,Y, gamma, beta) +gamma*beta*np.exp(-(1-X)*Y)

    c1 = plt.contour(X, Y, F , [0], colors='blue')
    c2 = plt.contour(X, Y, G , [0], colors='red')

    # Since it's symmetric at phi=0.5, we only need values for one side !
    if len(c1.collections[0].get_paths()) == 2 :
        p1 = [c1.collections[0].get_paths()[0], c1.collections[0].get_paths()[1]]
        v1 = np.concatenate( (p1[0].vertices, p1[1].vertices), axis = 0)
    else :
        p1 = c1.collections[0].get_paths()[0]
        v1 = p1.vertices

    x1 = v1[:,0]
    y1 = v1[:,1]

    # We dont need these values here since x2 = 1-x1 so F <=> G
    p2 = c2.collections[0].get_paths()[0]
    v2 = p2.vertices
    x2 = v2[:,0]
    y2 = v2[:,1]

    diff, phis = [], []
    for k in range(len(x1)):
      print(x1[k], y1[k], u1(x1[k], y1[k], gamma, beta) + gamma*beta, u2(x1[k], y1[k], gamma, beta) + gamma*beta)
      print(x1[k], y1[k], theo_u1(x1[k], y1[k], gamma, beta) + gamma*beta, theo_u2(x1[k], y1[k], gamma, beta) + gamma*beta)
      if u1(x1[k], y1[k], gamma, beta) + gamma*beta >= 1 or u2(x1[k], y1[k], gamma, beta) + gamma*beta >= 1 :
          phis.append(x1[k])
          diff.append(0)
          print('phi cannot exist : immediate synchrony')
      elif theo_u1(x1[k], y1[k], gamma, beta) + gamma*beta >= 1 or theo_u2(x1[k], y1[k], gamma, beta) + gamma*beta >= 1 or x1[k] == 0 or x1[k] == 1:
          phis.append(x1[k])
          if x1[k] < 0.5 :
              diff.append(-x1[k])
          elif x1[k] > 0.5 :
              diff.append(1-x1[k])
      else :
          phis.append(x1[k])
          [x1n, y1n] = fsolve(tosolve, [x1[k], y1[k]], args=(x1[k], y1[k], gamma, beta))
          print('new phi : ', x1n, ' new T : ', y1n)
          if x1n <= 0 :
              diff.append(-x1[k])
          elif x1n >= 1 :
              diff.append(1-x1[k])
          else :
              diff.append(x1n-x1[k])
      print()

    all_phis.append(phis)
    all_diffs.append(diff)

ax[0, 0].plot([0, 1], [0, 0], linestyle='--', color='k')

for k in range(len(all_phis)):
    ax[0, 0].plot(all_phis[k], all_diffs[k], color=c[k], label=f'$\\beta$ = {betas[k]}')

""" line 0 : Varying Gamma """

I = 1.3
gammas = [0.1, 0.15, 0.2, 0.25]
beta = 0.1

all_diffs = []
all_phis = []

for gamma in gammas :

    vector_x = []
    vector_g = []

    xrange = np.linspace(0, 1, 1001) # phi
    yrange = np.linspace(0, 20, 1001) # T
    X, Y = np.meshgrid(xrange,yrange)

    phi = []
    T = []

    F = - np.exp(-(1+2*gamma)*X*Y)*u2(X,Y, gamma, beta) - u1(X,Y, gamma, beta) +1 - np.exp(-(1+2*gamma)*X*Y)*gamma*beta
    G = 2*I*(1-np.exp(-(1-X)*Y)) + u1(X,Y, gamma, beta)*np.exp(-(1-X)*Y) -1 -u2(X,Y, gamma, beta) +gamma*beta*np.exp(-(1-X)*Y)

    c1 = plt.contour(X, Y, F , [0], colors='blue')
    c2 = plt.contour(X, Y, G , [0], colors='red')

    # Since it's symmetric at phi=0.5, we only need values for one side !
    if len(c1.collections[0].get_paths()) == 2 :
        p1 = [c1.collections[0].get_paths()[0], c1.collections[0].get_paths()[1]]
        v1 = np.concatenate( (p1[0].vertices, p1[1].vertices), axis = 0)
    else :
        p1 = c1.collections[0].get_paths()[0]
        v1 = p1.vertices

    x1 = v1[:,0]
    y1 = v1[:,1]

    # We dont need these values here since x2 = 1-x1 so F <=> G
    p2 = c2.collections[0].get_paths()[0]
    v2 = p2.vertices
    x2 = v2[:,0]
    y2 = v2[:,1]

    diff, phis = [], []
    for k in range(len(x1)):
      print(x1[k], y1[k], u1(x1[k], y1[k], gamma, beta) + gamma*beta, u2(x1[k], y1[k], gamma, beta) + gamma*beta)
      print(x1[k], y1[k], theo_u1(x1[k], y1[k], gamma, beta) + gamma*beta, theo_u2(x1[k], y1[k], gamma, beta) + gamma*beta)
      if u1(x1[k], y1[k], gamma, beta) + gamma*beta >= 1 or u2(x1[k], y1[k], gamma, beta) + gamma*beta >= 1 :
          phis.append(x1[k])
          diff.append(0)
          print('phi cannot exist : immediate synchrony')
      elif theo_u1(x1[k], y1[k], gamma, beta) + gamma*beta >= 1 or theo_u2(x1[k], y1[k], gamma, beta) + gamma*beta >= 1 or x1[k] == 0 or x1[k] == 1:
          phis.append(x1[k])
          if x1[k] < 0.5 :
              diff.append(-x1[k])
          elif x1[k] > 0.5 :
              diff.append(1-x1[k])
      else :
          phis.append(x1[k])
          [x1n, y1n] = fsolve(tosolve, [x1[k], y1[k]], args=(x1[k], y1[k], gamma, beta))
          print('new phi : ', x1n, ' new T : ', y1n)
          if x1n <= 0 :
              diff.append(-x1[k])
          elif x1n >= 1 :
              diff.append(1-x1[k])
          else :
              diff.append(x1n-x1[k])
      print()

    all_phis.append(phis)
    all_diffs.append(diff)

ax[0, 1].plot([0, 1], [0, 0], linestyle='--', color='k')

# with this, we can plot phi_n+1 = F(phi_n) and maybe even number of cycles for synchrony
for k in range(len(all_phis)):
    ax[0, 1].plot(all_phis[k], all_diffs[k], color=c[k], label=f'$\gamma$ = {gammas[k]}')

""" line 0 : Varying I """

currents = [1.15, 1.2, 1.25, 1.3]
gamma = 0.1
beta = 0.1

all_diffs = []
all_phis = []

for I in currents :

    vector_x = []
    vector_g = []

    xrange = np.linspace(0, 1, 1001) # phi
    yrange = np.linspace(0, 100, 1001) # T
    X, Y = np.meshgrid(xrange,yrange)

    phi = []
    T = []

    F = - np.exp(-(1+2*gamma)*X*Y)*u2(X,Y, gamma, beta) - u1(X,Y, gamma, beta) +1 - np.exp(-(1+2*gamma)*X*Y)*gamma*beta
    G = 2*I*(1-np.exp(-(1-X)*Y)) + u1(X,Y, gamma, beta)*np.exp(-(1-X)*Y) -1 -u2(X,Y, gamma, beta) +gamma*beta*np.exp(-(1-X)*Y)

    c1 = plt.contour(X, Y, F , [0], colors='blue')
    c2 = plt.contour(X, Y, G , [0], colors='red')

    # Since it's symmetric at phi=0.5, we only need values for one side !
    if len(c1.collections[0].get_paths()) == 2 :
        p1 = [c1.collections[0].get_paths()[0], c1.collections[0].get_paths()[1]]
        v1 = np.concatenate( (p1[0].vertices, p1[1].vertices), axis = 0)
    else :
        p1 = c1.collections[0].get_paths()[0]
        v1 = p1.vertices

    x1 = v1[:,0]
    y1 = v1[:,1]

    # We dont need these values here since x2 = 1-x1 so F <=> G
    p2 = c2.collections[0].get_paths()[0]
    v2 = p2.vertices
    x2 = v2[:,0]
    y2 = v2[:,1]

    diff, phis = [], []
    for k in range(len(x1)):
      print(x1[k], y1[k], u1(x1[k], y1[k], gamma, beta) + gamma*beta, u2(x1[k], y1[k], gamma, beta) + gamma*beta)
      print(x1[k], y1[k], theo_u1(x1[k], y1[k], gamma, beta) + gamma*beta, theo_u2(x1[k], y1[k], gamma, beta) + gamma*beta)
      if u1(x1[k], y1[k], gamma, beta) + gamma*beta >= 1 or u2(x1[k], y1[k], gamma, beta) + gamma*beta >= 1 :
          phis.append(x1[k])
          diff.append(0)
          print('phi cannot exist : immediate synchrony')
      elif theo_u1(x1[k], y1[k], gamma, beta) + gamma*beta >= 1 or theo_u2(x1[k], y1[k], gamma, beta) + gamma*beta >= 1 or x1[k] == 0 or x1[k] == 1:
          phis.append(x1[k])
          if x1[k] < 0.5 :
              diff.append(-x1[k])
          elif x1[k] > 0.5 :
              diff.append(1-x1[k])
      else :
          phis.append(x1[k])
          [x1n, y1n] = fsolve(tosolve, [x1[k], y1[k]], args=(x1[k], y1[k], gamma, beta))
          print('new phi : ', x1n, ' new T : ', y1n)
          if x1n <= 0 :
              diff.append(-x1[k])
          elif x1n >= 1 :
              diff.append(1-x1[k])
          else :
              diff.append(x1n-x1[k])
      print()

    all_phis.append(phis)
    all_diffs.append(diff)

ax[0, 2].clear() # Clearing the implicit function plots
ax[0, 2].plot([0, 1], [0, 0], linestyle='--', color='k')

# with this, we can plot phi_n+1 = F(phi_n) and maybe even number of cycles for synchrony
for k in range(len(all_phis)):
    ax[0, 2].plot(all_phis[k], all_diffs[k], color=c[k], label=f'$I$ = {currents[k]}')




""" line 1 : Varying beta """

I = 1.6
gamma = 0.1
betas = [0.1, 0.15, 0.2, 0.25]

all_diffs = []
all_phis = []

for beta in betas :

    vector_x = []
    vector_g = []

    xrange = np.linspace(0, 1, 1001) # phi
    yrange = np.linspace(0, 20, 1001) # T
    X, Y = np.meshgrid(xrange,yrange)

    phi = []
    T = []

    F = - np.exp(-(1+2*gamma)*X*Y)*u2(X,Y, gamma, beta) - u1(X,Y, gamma, beta) +1 - np.exp(-(1+2*gamma)*X*Y)*gamma*beta
    G = 2*I*(1-np.exp(-(1-X)*Y)) + u1(X,Y, gamma, beta)*np.exp(-(1-X)*Y) -1 -u2(X,Y, gamma, beta) +gamma*beta*np.exp(-(1-X)*Y)

    c1 = plt.contour(X, Y, F , [0], colors='blue')
    c2 = plt.contour(X, Y, G , [0], colors='red')

    # Since it's symmetric at phi=0.5, we only need values for one side !
    if len(c1.collections[0].get_paths()) == 2 :
        p1 = [c1.collections[0].get_paths()[0], c1.collections[0].get_paths()[1]]
        v1 = np.concatenate( (p1[0].vertices, p1[1].vertices), axis = 0)
    else :
        p1 = c1.collections[0].get_paths()[0]
        v1 = p1.vertices

    x1 = v1[:,0]
    y1 = v1[:,1]

    # We dont need these values here since x2 = 1-x1 so F <=> G
    p2 = c2.collections[0].get_paths()[0]
    v2 = p2.vertices
    x2 = v2[:,0]
    y2 = v2[:,1]

    diff, phis = [], []
    for k in range(len(x1)):
      print(x1[k], y1[k], u1(x1[k], y1[k], gamma, beta) + gamma*beta, u2(x1[k], y1[k], gamma, beta) + gamma*beta)
      print(x1[k], y1[k], theo_u1(x1[k], y1[k], gamma, beta) + gamma*beta, theo_u2(x1[k], y1[k], gamma, beta) + gamma*beta)
      if u1(x1[k], y1[k], gamma, beta) + gamma*beta >= 1 or u2(x1[k], y1[k], gamma, beta) + gamma*beta >= 1 :
          phis.append(x1[k])
          diff.append(0)
          print('phi cannot exist : immediate synchrony')
      elif theo_u1(x1[k], y1[k], gamma, beta) + gamma*beta >= 1 or theo_u2(x1[k], y1[k], gamma, beta) + gamma*beta >= 1 or x1[k] == 0 or x1[k] == 1:
          phis.append(x1[k])
          if x1[k] < 0.5 :
              diff.append(-x1[k])
          elif x1[k] > 0.5 :
              diff.append(1-x1[k])
      else :
          phis.append(x1[k])
          [x1n, y1n] = fsolve(tosolve, [x1[k], y1[k]], args=(x1[k], y1[k], gamma, beta))
          print('new phi : ', x1n, ' new T : ', y1n)
          if x1n <= 0 :
              diff.append(-x1[k])
          elif x1n >= 1 :
              diff.append(1-x1[k])
          else :
              diff.append(x1n-x1[k])
      print()

    all_phis.append(phis)
    all_diffs.append(diff)

ax[1, 0].plot([0, 1], [0, 0], linestyle='--', color='k')

for k in range(len(all_phis)):
    ax[1, 0].plot(all_phis[k], all_diffs[k], color=c[k], label=f'$\\beta$ = {betas[k]}')

""" line 1 : Varying Gamma """

I = 1.6
gammas = [0.1, 0.15, 0.2, 0.25]
beta = 0.1

all_diffs = []
all_phis = []

for gamma in gammas :

    vector_x = []
    vector_g = []

    xrange = np.linspace(0, 1, 1001) # phi
    yrange = np.linspace(0, 20, 1001) # T
    X, Y = np.meshgrid(xrange,yrange)

    phi = []
    T = []

    F = - np.exp(-(1+2*gamma)*X*Y)*u2(X,Y, gamma, beta) - u1(X,Y, gamma, beta) +1 - np.exp(-(1+2*gamma)*X*Y)*gamma*beta
    G = 2*I*(1-np.exp(-(1-X)*Y)) + u1(X,Y, gamma, beta)*np.exp(-(1-X)*Y) -1 -u2(X,Y, gamma, beta) +gamma*beta*np.exp(-(1-X)*Y)

    c1 = plt.contour(X, Y, F , [0], colors='blue')
    c2 = plt.contour(X, Y, G , [0], colors='red')

    # Since it's symmetric at phi=0.5, we only need values for one side !
    if len(c1.collections[0].get_paths()) == 2 :
        p1 = [c1.collections[0].get_paths()[0], c1.collections[0].get_paths()[1]]
        v1 = np.concatenate( (p1[0].vertices, p1[1].vertices), axis = 0)
    else :
        p1 = c1.collections[0].get_paths()[0]
        v1 = p1.vertices

    x1 = v1[:,0]
    y1 = v1[:,1]

    # We dont need these values here since x2 = 1-x1 so F <=> G
    p2 = c2.collections[0].get_paths()[0]
    v2 = p2.vertices
    x2 = v2[:,0]
    y2 = v2[:,1]

    diff, phis = [], []
    for k in range(len(x1)):
      print(x1[k], y1[k], u1(x1[k], y1[k], gamma, beta) + gamma*beta, u2(x1[k], y1[k], gamma, beta) + gamma*beta)
      print(x1[k], y1[k], theo_u1(x1[k], y1[k], gamma, beta) + gamma*beta, theo_u2(x1[k], y1[k], gamma, beta) + gamma*beta)
      if u1(x1[k], y1[k], gamma, beta) + gamma*beta >= 1 or u2(x1[k], y1[k], gamma, beta) + gamma*beta >= 1 :
          phis.append(x1[k])
          diff.append(0)
          print('phi cannot exist : immediate synchrony')
      elif theo_u1(x1[k], y1[k], gamma, beta) + gamma*beta >= 1 or theo_u2(x1[k], y1[k], gamma, beta) + gamma*beta >= 1 or x1[k] == 0 or x1[k] == 1:
          phis.append(x1[k])
          if x1[k] < 0.5 :
              diff.append(-x1[k])
          elif x1[k] > 0.5 :
              diff.append(1-x1[k])
      else :
          phis.append(x1[k])
          [x1n, y1n] = fsolve(tosolve, [x1[k], y1[k]], args=(x1[k], y1[k], gamma, beta))
          print('new phi : ', x1n, ' new T : ', y1n)
          if x1n <= 0 :
              diff.append(-x1[k])
          elif x1n >= 1 :
              diff.append(1-x1[k])
          else :
              diff.append(x1n-x1[k])
      print()

    all_phis.append(phis)
    all_diffs.append(diff)

ax[1, 1].plot([0, 1], [0, 0], linestyle='--', color='k')

# with this, we can plot phi_n+1 = F(phi_n) and maybe even number of cycles for synchrony
for k in range(len(all_phis)):
    ax[1, 1].plot(all_phis[k], all_diffs[k], color=c[k], label=f'$\gamma$ = {gammas[k]}')

""" line 1 : Varying I """

currents = [1.15, 1.2, 1.25, 1.3]
gamma = 0.1
beta = 0.4

all_diffs = []
all_phis = []

for I in currents :

    vector_x = []
    vector_g = []

    xrange = np.linspace(0, 1, 1001) # phi
    yrange = np.linspace(0, 100, 1001) # T
    X, Y = np.meshgrid(xrange,yrange)

    phi = []
    T = []

    F = - np.exp(-(1+2*gamma)*X*Y)*u2(X,Y, gamma, beta) - u1(X,Y, gamma, beta) +1 - np.exp(-(1+2*gamma)*X*Y)*gamma*beta
    G = 2*I*(1-np.exp(-(1-X)*Y)) + u1(X,Y, gamma, beta)*np.exp(-(1-X)*Y) -1 -u2(X,Y, gamma, beta) +gamma*beta*np.exp(-(1-X)*Y)

    c1 = plt.contour(X, Y, F , [0], colors='blue')
    c2 = plt.contour(X, Y, G , [0], colors='red')

    # Since it's symmetric at phi=0.5, we only need values for one side !
    if len(c1.collections[0].get_paths()) == 2 :
        p1 = [c1.collections[0].get_paths()[0], c1.collections[0].get_paths()[1]]
        v1 = np.concatenate( (p1[0].vertices, p1[1].vertices), axis = 0)
    else :
        p1 = c1.collections[0].get_paths()[0]
        v1 = p1.vertices

    x1 = v1[:,0]
    y1 = v1[:,1]

    # We dont need these values here since x2 = 1-x1 so F <=> G
    p2 = c2.collections[0].get_paths()[0]
    v2 = p2.vertices
    x2 = v2[:,0]
    y2 = v2[:,1]

    diff, phis = [], []
    for k in range(len(x1)):
      print(x1[k], y1[k], u1(x1[k], y1[k], gamma, beta) + gamma*beta, u2(x1[k], y1[k], gamma, beta) + gamma*beta)
      print(x1[k], y1[k], theo_u1(x1[k], y1[k], gamma, beta) + gamma*beta, theo_u2(x1[k], y1[k], gamma, beta) + gamma*beta)
      if u1(x1[k], y1[k], gamma, beta) + gamma*beta >= 1 or u2(x1[k], y1[k], gamma, beta) + gamma*beta >= 1 :
          phis.append(x1[k])
          diff.append(0)
          print('phi cannot exist : immediate synchrony')
      elif theo_u1(x1[k], y1[k], gamma, beta) + gamma*beta >= 1 or theo_u2(x1[k], y1[k], gamma, beta) + gamma*beta >= 1 or x1[k] == 0 or x1[k] == 1:
          phis.append(x1[k])
          if x1[k] < 0.5 :
              diff.append(-x1[k])
          elif x1[k] > 0.5 :
              diff.append(1-x1[k])
      else :
          phis.append(x1[k])
          [x1n, y1n] = fsolve(tosolve, [x1[k], y1[k]], args=(x1[k], y1[k], gamma, beta))
          print('new phi : ', x1n, ' new T : ', y1n)
          if x1n <= 0 :
              diff.append(-x1[k])
          elif x1n >= 1 :
              diff.append(1-x1[k])
          else :
              diff.append(x1n-x1[k])
      print()

    all_phis.append(phis)
    all_diffs.append(diff)

ax[1, 2].clear() # Clearing the implicit function plots
ax[1, 2].plot([0, 1], [0, 0], linestyle='--', color='k')

# with this, we can plot phi_n+1 = F(phi_n) and maybe even number of cycles for synchrony
for k in range(len(all_phis)):
    ax[1, 2].plot(all_phis[k], all_diffs[k], color=c[k], label=f'$I$ = {currents[k]}')

###################################################################""

ax[0, 0].legend(loc='upper left', prop={'size': 10})
ax[0, 1].legend(loc='upper left', prop={'size': 10})
ax[0, 2].legend(loc='upper left', prop={'size': 10})
ax[1, 0].legend(loc='upper left', prop={'size': 10})
ax[1, 1].legend(loc='upper left', prop={'size': 10})
ax[1, 2].legend(loc='upper left', prop={'size': 10})

ax[0, 0].set_ylabel('$H(\phi)$', size=14)
ax[1, 0].set_xlabel('$\phi$', size=14)
ax[1, 1].set_xlabel('$\phi$', size=14)
ax[1, 2].set_xlabel('$\phi$', size=14)

ax[0, 0].set_title(f'$I$ = 1.4, $\gamma$ = 0.1', size=16) #I = 1.4, gamma = 0.1
ax[0, 1].set_title(f'$I$ = 1.3, $\\beta$ = 0.1', size=16) #I = 1.3, beta = 0.1
ax[0, 2].set_title(f'$\gamma$ = 0.1, $\\beta$ = 0.1', size=16) #gamma = 0.1, beta = 0.1
ax[1, 0].set_title(f'$I$ = 1.6, $\gamma$ = 0.1', size=16) #I = 1.4, gamma = 0.1
ax[1, 1].set_title(f'$I$ = 1.6, $\\beta$ = 0.1', size=16) #I = 1.3, beta = 0.1
ax[1, 2].set_title(f'$\gamma$ = 0.1, $\\beta$ = 0.4', size=16) #gamma = 0.1, beta = 0.1

fig.suptitle('H-function for different values of $\\beta$, $\gamma$ and $I$', size=20)
fig.tight_layout()
#fig.subplots_adjust(hspace=0.25, top=0.88)

plt.savefig('allinone4.png', dpi=600)
plt.show()
