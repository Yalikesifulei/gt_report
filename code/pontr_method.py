import numpy as np
from scipy.optimize import root_scalar

alpha, beta = 1, 2
rho, sigma = 2, 1

x_0, dx_0 = np.array([3, 2]), np.array([1, 1])
y_0, dy_0 = np.array([1, 0]), np.array([0, 1])

z1_0 = x_0 - y_0
z2_0 = dx_0
z3_0 = dy_0

def proj(t, z1, z2, z3): 
    return z1 + (1 - np.exp(-alpha*t))/alpha * z2 - (1 - np.exp(-beta*t))/beta * z3

def T0_finder(t):
    res = rho/alpha**2 * (alpha*t + np.exp(-alpha*t) - 1)
    res = res - sigma/beta**2 * (beta*t + np.exp(-beta*t) - 1)
    res = res**2
    res -= np.linalg.norm(proj(t, z1_0, z2_0, z3_0))**2
    return res

T0 = root_scalar(T0_finder, bracket=(0, 100)).root

def w(t):
    return rho/alpha * (1-np.exp(-alpha*t)) - sigma/beta*(1-np.exp(-beta*t))

print('T0 =', T0)