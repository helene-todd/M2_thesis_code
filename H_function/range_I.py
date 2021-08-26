from scipy.optimize import fsolve
from matplotlib import cm, rcParams
import matplotlib.pyplot as plt
import numpy as np
import math
from shapely import geometry

""" ToDo : check if this is equivalent to the G-function for weak coupling """

c = ['#aa3863', '#d97020', '#ef9f07', '#449775', '#3b7d86']
rcParams.update({'figure.autolayout': True})

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

# Only makes sense if I > I_low i.e. when forks stop existing (see bifurcation diagram code or figures) !!!
currents = [1.2]
gamma = 0.4
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

    # Closing the implicit function plots, but can be shown if wanted
    plt.show()
    plt.clf()
    plt.close()

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

# with this, we can plot phi_n+1 = F(phi_n) and maybe even number of cycles for synchrony
for k in range(len(all_phis)):
    plt.plot(all_phis[k], all_diffs[k], color=c[k], label=f'$I = {currents[k]}$')

plt.legend(loc='upper left')
plt.xlabel('$\phi$')
plt.ylabel('$\Delta \phi$')
plt.title(f'Change in $\phi$ per cycle, $\gamma = {gamma}$ and $\\beta = {beta}$')
plt.plot([0, 1], [0, 0], color='k', linestyle='--')
#plt.savefig('DeltaPhi_I_range.svg')
plt.show()
