import matplotlib.pyplot as plt
import numpy as np
import math
from shapely import geometry
import os

# TODO : write to file in xppaut style rather than like this.. or open weak coupling solution

c = ['#511845', '#900c3f', '#c70039', '#ff5733']
s = ['-', '--']

flatten = lambda t: [item for sublist in contour1.collections[0].get_paths() for item in sublist]

def findIntersection(contour1, contour2):
    temp = []
    for el in contour1.collections[0].get_paths() :
        temp.append(el.vertices)
    v1 = np.concatenate(temp, axis=0)

    temp = []
    for el in contour2.collections[0].get_paths() :
        temp.append(el.vertices)
    v2 = np.concatenate(temp, axis=0)

    if len(v1) > 1 and len(v2) > 1 :
        poly1 = geometry.LineString(v1)
        poly2 = geometry.LineString(v2)
        intersection = poly1.intersection(poly2)
        return intersection
    else :
        return None

def A(phi, T, gamma, beta, kappa, alpha):
    term = - (kappa*alpha*alpha*np.exp(-alpha*T)/((1-np.exp(-alpha*T))*(1-alpha)**2)) * ((1-alpha)*T-1-np.exp(-phi*T*(1-alpha))*((1-alpha)*(1-phi)*T-1)) \
    - (kappa*alpha*alpha*np.exp(-2*alpha*T)*T/((1-alpha) * (1-np.exp(-alpha*T))**2)) * (1-np.exp(-phi*T*(1-alpha)))
    return term

def B(phi, T, gamma, beta, kappa, alpha):
    term = - (kappa*alpha*alpha*np.exp(-phi*T)/((1-np.exp(-alpha*T))*(1-alpha)**2)) * (np.exp(phi*T*(1-alpha))*((1-alpha)*phi*T-1)+1) \
    - (kappa*alpha*alpha*np.exp(-T*(alpha+phi))*T/((1-alpha) * (1-np.exp(-alpha*T))**2)) * (np.exp(phi*T*(1-alpha))-1)
    return term

def C(phi, T, gamma, beta, kappa, alpha):
    term = - (kappa*alpha*alpha*np.exp(-alpha*T)/((1-np.exp(-alpha*T))*(1+2*gamma-alpha)**2)) * ((1+2*gamma-alpha)*T -1 -np.exp(-(1+2*gamma-alpha)*phi*T)*((1+2*gamma-alpha)*(1-phi)*T-1)) \
    - (kappa*alpha*alpha*np.exp(-2*alpha*T)*T/((1+2*gamma-alpha) * (1-np.exp(-alpha*T))**2)) * (1-np.exp(-phi*T*(1+2*gamma-alpha)))
    return term

def D(phi, T, gamma, beta, kappa, alpha):
    term = - (kappa*alpha*alpha*np.exp(-(1+2*gamma)*phi*T)/((1-np.exp(-alpha*T))*(1+2*gamma-alpha)**2)) * (np.exp(phi*T*(1+2*gamma-alpha))*((1+2*gamma-alpha)*phi*T-1) +1) \
    - (kappa*alpha*alpha*np.exp(-((1+2*gamma)*phi+alpha)*T)*T/((1+2*gamma-alpha) * (1-np.exp(-alpha*T))**2)) * (np.exp(phi*T*(1+2*gamma-alpha))-1)
    return term

def u1(phi, T, gamma, beta, kappa, alpha) :
    return (np.exp((1-phi)*T)*(2*(1-I)-B(1-phi,T,gamma,beta,kappa,alpha)-A(1-phi,T,gamma,beta,kappa,alpha)-D(1-phi,T,gamma,beta,kappa,alpha)+C(1-phi,T,gamma,beta,kappa,alpha)) + 2*I)/(1 + np.exp(-2*gamma*(1-phi)*T)) - gamma*beta

def u2(phi, T, gamma, beta, kappa, alpha) :
    return (np.exp(phi*T)*(2*(1-I)-A(phi,T,gamma,beta,kappa,alpha)-B(phi,T,gamma,beta,kappa,alpha)+C(phi,T,gamma,beta,kappa,alpha)-D(phi,T,gamma,beta,kappa,alpha)) + 2*I)/(1 + np.exp(-2*gamma*phi*T)) - gamma*beta

# Moderate or strong gammas
rhos = [[0, 0.1], [0.02, 0.08], [0.05, 0.05], [0.08, 0.02], [0.1, 0]] #
beta = 0.2
alpha = 4

for gamma, kappa in rhos :
    print('gamma = ', gamma)
    print('kappa = ', kappa)
    if not os.path.exists(f'gamma={gamma}_kappa={kappa}.dat'):
        os.mknod(f'gamma={gamma}_kappa={kappa}.dat')

    xrange = np.linspace(0, 1, 1001) # phi
    yrange = np.linspace(0.5, 20, 1001) # T
    X, Y = np.meshgrid(xrange,yrange)


    f = open(f'gamma={gamma}_kappa={kappa}.dat', 'w')

    upper_fork = []
    lower_fork = []
    middle_fork = []
    values_I_forks = []
    values_I_middle = []

    for I in np.linspace(1, 2, 601) : #201
        print('I = ', I)
        F = - np.exp(-(1+2*gamma)*X*Y)*(u2(X,Y,gamma,beta,kappa,alpha)+gamma*beta) +C(X,Y,gamma,beta,kappa,alpha) -D(X,Y,gamma,beta,kappa,alpha) - u1(X,Y,gamma,beta,kappa,alpha) +1
        G = np.exp(-(1+2*gamma)*(1-X)*Y)*(u1(X,Y,gamma,beta,kappa,alpha)+gamma*beta) +D(1-X,Y,gamma,beta,kappa,alpha) -C(1-X,Y,gamma,beta,kappa,alpha) - 1 + u2(X,Y,gamma,beta,kappa,alpha)

        c1 = plt.contour(X, Y, F , [0], colors='blue')
        c2 = plt.contour(X, Y, G , [0], colors='red')

        #plt.show()
        plt.clf()
        plt.close()

        intersection_points = findIntersection(c1,c2)
        print('INTERSECTION POINTS : ', intersection_points)

        if isinstance(intersection_points, geometry.multipoint.MultiPoint) and len(intersection_points) != 0 :
            if (len(intersection_points) == 3 and intersection_points[0].x > 0.01) or len(intersection_points) == 5:

                if u1(intersection_points[int(len(intersection_points)/2)-1].x, intersection_points[int(len(intersection_points)/2)-1].y, gamma, beta, kappa, alpha) + gamma*beta < 1 and u2(intersection_points[int(len(intersection_points)/2)-1].x, intersection_points[int(len(intersection_points)/2)-1].y, gamma, beta, kappa, alpha) + gamma*beta < 1 :
                    print(intersection_points[int(len(intersection_points)/2)-1])
                    print(intersection_points[int(len(intersection_points)/2)+1])

                    values_I_forks.append(round(I, 7))
                    lower_fork.append(round(intersection_points[int(len(intersection_points)/2)-1].x, 7))
                    upper_fork.append(round(intersection_points[int(len(intersection_points)/2)+1].x, 7))

                    #print('u1 ', u1(intersection_points[int(len(intersection_points)/2)-1].x, intersection_points[int(len(intersection_points)/2)-1].y, gamma, beta, kappa, alpha))
                    #print('u2 ', u2(intersection_points[int(len(intersection_points)/2)-1].x, intersection_points[int(len(intersection_points)/2)-1].y, gamma, beta, kappa, alpha))

            if len(intersection_points) == 4 :
                values = []
                for el in intersection_points :
                    if el.x > 0.01 and el.x < 0.99 :
                        print(el)
                        values.append(el)

                if len(values) == 3 :
                    if u1(intersection_points[0].x, intersection_points[0].y, gamma, beta, kappa, alpha) + gamma*beta < 1 and u2(intersection_points[0].x, intersection_points[0].y, gamma, beta, kappa, alpha) + gamma*beta < 1 :
                        values_I_forks.append(round(I, 7))
                        lower_fork.append(round(values[0].x, 7))
                        upper_fork.append(round(values[2].x, 7))

                        #print('u1 ', u1(intersection_points[0].x, intersection_points[0].y, gamma, beta, kappa, alpha))
                        #print('u2 ', u2(intersection_points[0].x, intersection_points[0].y, gamma, beta, kappa, alpha))

            if (len(intersection_points) == 3 and intersection_points[0].x > 0.01) or len(intersection_points) == 5:

                if u1(intersection_points[int(len(intersection_points)/2)].x, intersection_points[int(len(intersection_points)/2)].y, gamma, beta, kappa, alpha) + gamma*beta < 1 and u2(intersection_points[int(len(intersection_points)/2)].x, intersection_points[int(len(intersection_points)/2)].y, gamma, beta, kappa, alpha) + gamma*beta < 1 :
                    print(intersection_points[int(len(intersection_points)/2)])
                    values_I_middle.append(round(I, 7))
                    middle_fork.append(round(intersection_points[int(len(intersection_points)/2)].x, 7))

            if len(intersection_points) == 4 :
                values = []
                for el in intersection_points :
                    if el.x > 0.01 and el.x < 0.99 :
                        print(el)
                        values.append(el)

                if len(values) == 3 :
                    if u1(intersection_points[1].x, intersection_points[1].y, gamma, beta, kappa, alpha) + gamma*beta < 1 and u2(intersection_points[1].x, intersection_points[1].y, gamma, beta, kappa, alpha) + gamma*beta < 1 :
                        values_I_middle.append(round(I, 7))
                        middle_fork.append(round(values[1].x, 7))


        # Once the forks have been computed, there isn't any need to go on.
        #if isinstance(intersection_points, geometry.point.Point) :
            #if u1(intersection_points.x, intersection_points.y, gamma, beta, kappa, alpha) + gamma*beta < 1 and u2(intersection_points.x, intersection_points.y, gamma, beta, kappa, alpha) + gamma*beta < 1 :
                #values_I_forks.append(round(I, 7))
                #middle_fork.append(round(intersection_points.x, 7))

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
