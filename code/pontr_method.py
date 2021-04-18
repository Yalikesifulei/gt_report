import numpy as np
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
import tikzplotlib

alpha, beta = 1, 2
rho, sigma = 2, 1

x_0, dx_0 = np.array([3, 2]), np.array([1, 1])
y_0, dy_0 = np.array([1, 0]), np.array([0, 1])

z1_0 = x_0 - y_0
z2_0 = dx_0
z3_0 = dy_0

def proj(t, z1, z2, z3): 
    return z1 + (1 - np.exp(-alpha*t))/alpha * z2 - (1 - np.exp(-beta*t))/beta * z3

def r(t):
    res = rho/alpha**2 * (alpha*t + np.exp(-alpha*t) - 1)
    res = res - sigma/beta**2 * (beta*t + np.exp(-beta*t) - 1)
    return res

T0 = root_scalar(lambda t: r(t) - np.linalg.norm(proj(t, z1_0, z2_0, z3_0)), bracket=(0, 100)).root

print('T0 =', T0)
print('r(T0) =', r(T0))

t_domain = np.linspace(0, T0*1.1, 100)
plt.plot(t_domain, r(t_domain), label='$r(t)$')
plt.plot(t_domain, [np.linalg.norm(proj(t, z1_0, z2_0, z3_0)) for t in t_domain], label='$\Vert\pi e^{At} z_0\Vert$')
plt.scatter(T0, r(T0), color='red')
plt.legend(loc='lower right')
plt.show()
#tikzplotlib.save('./code/pontr_method_plot.tex')