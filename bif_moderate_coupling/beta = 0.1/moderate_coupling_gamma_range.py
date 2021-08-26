import matplotlib.pyplot as plt
import numpy as np
import math
from shapely import geometry
import os

# TODO : write to file in xppaut style rather than like this.. or open weak coupling solution

c = ['#511845', '#900c3f', '#c70039', '#ff5733']
s = ['-', '--']

flatten = lambda t: [item for sublist in contour1.collections[0].get_paths() for item in sublist]

def findIntersection(contour1,contour2):
    if len(contour1.collections[0].get_paths()) == 2 :
        p1 = [contour1.collections[0].get_paths()[0], contour1.collections[0].get_paths()[1]]
        v1 = np.concatenate( (p1[0].vertices, p1[1].vertices), axis = 0)
    else :
        p1 = contour1.collections[0].get_paths()[0]
        v1 = p1.vertices

    if len(contour1.collections[0].get_paths()) == 2 :
        p2 = [contour2.collections[0].get_paths()[0], contour2.collections[0].get_paths()[1]]
        v2 = np.concatenate( (p2[0].vertices, p2[1].vertices), axis = 0)
    else :
        p2 = contour2.collections[0].get_paths()[0]
        v2 = p2.vertices

    if len(v1) > 1 and len(v2) > 1 :
        poly1 = geometry.LineString(v1)
        poly2 = geometry.LineString(v2)
        intersection = poly1.intersection(poly2)
        return intersection
    else :
        return None

def u1(phi, T, gamma, beta) :
    return (np.exp((1-phi)*T)*(2-2*I) + 2*I)/(1 + np.exp(-2*gamma*(1-phi)*T)) - gamma*beta

def u2(phi, T, gamma, beta) :
    return (np.exp(phi*T)*(2-2*I) + 2*I)/(1 + np.exp(-2*gamma*phi*T)) - gamma*beta

# value of I when phi = 0.5 for the right T
def I_function(T, gamma, beta) :
    return -1/2*(beta*gamma*(math.exp((T*gamma + 1/2*T)) + math.exp((1/2*T))) - 2*math.exp((T*gamma + T)) + math.exp((T*gamma + 1/2*T)) - math.exp((1/2*T)))/(math.exp((T*gamma + T)) - math.exp((T*gamma + 1/2*T)) + math.exp((1/2*T)) - 1)

# Moderate or strong gammas
gammas = [0.1]
beta = 0.1

for gamma in gammas :
    if not os.path.exists(f'gamma_{gamma}.dat'):
        os.mknod(f'gamma_{gamma}.dat')

xrange = np.linspace(0, 1, 2001) # phi
yrange = np.linspace(0, 20, 2001) # T
X, Y = np.meshgrid(xrange,yrange)

for gamma in gammas :
    f = open(f'gamma_{gamma}.dat', 'w')

    upper_fork = []
    lower_fork = []
    middle_fork = []
    values_I_forks = []
    values_I_middle = []

    for I in np.linspace(1, 2, 1000) :
        print(I)
        F = - np.exp(-(1+2*gamma)*X*Y)*u2(X,Y, gamma, beta) - u1(X,Y, gamma, beta) +1 - np.exp(-(1+2*gamma)*X*Y)*gamma*beta
        #G = np.exp(-(1+2*gamma)*(1-X)*Y)*(u1(X,Y, gamma, beta)+gamma*beta) -1 + u2(X,Y, gamma, beta) # ALTERNATIVELY, PRODUCES SAME RESULTS
        G = 2*I*(1-np.exp(-(1-X)*Y)) + u1(X,Y, gamma, beta)*np.exp(-(1-X)*Y) -1 -u2(X,Y, gamma, beta) +gamma*beta*np.exp(-(1-X)*Y)

        c1 = plt.contour(X, Y, F , [0], colors='blue')
        c2 = plt.contour(X, Y, G , [0], colors='red')

        plt.show()
        plt.clf()
        plt.close()

        intersection_points = findIntersection(c1,c2)
        #print(intersection_points)

        if isinstance(intersection_points, geometry.multipoint.MultiPoint) :
            if u1(intersection_points[0].x, intersection_points[0].y, gamma, beta) + gamma*beta < 1 and u2(intersection_points[0].x, intersection_points[0].y, gamma, beta) + gamma*beta < 1 :
                values_I_forks.append(round(I, 7))
                lower_fork.append(round(intersection_points[0].x, 7))
                upper_fork.append(round(intersection_points[2].x, 7))
            if u1(intersection_points[1].x, intersection_points[1].y, gamma, beta) + gamma*beta < 1 and u2(intersection_points[1].x, intersection_points[1].y, gamma, beta) + gamma*beta < 1 :
                values_I_middle.append(round(I, 7))
                middle_fork.append(round(intersection_points[1].x, 7))

        # Once the forks have been computed, there isn't any need to go on.
        if isinstance(intersection_points, geometry.point.Point) :
            break


    # We kind of "cheat" a little here to get the fork to intersect with phi = 0.5 in the plot
    values_I_forks.append(max(values_I_forks))
    lower_fork.append(0.5)
    upper_fork.append(0.5)

    for I in np.linspace(2, max(values_I_middle), 200):
        f.write(f'{round(I, 7)} 0.5 0.5 2 1 0\n')

    for k in range(len(values_I_middle)-1, -1, -1):
        f.write(f'{values_I_middle[k]} {middle_fork[k]} {middle_fork[k]} 1 2 0\n')

    for k in range(len(values_I_forks)-1, -1, -1):
        f.write(f'{values_I_forks[k]} {upper_fork[k]} {upper_fork[k]} 2 2 0\n')

    for k in range(len(values_I_forks)-1, -1, -1):
        f.write(f'{values_I_forks[k]} {lower_fork[k]} {lower_fork[k]} 2 2 0\n')

f.close()
