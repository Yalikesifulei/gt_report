import numpy as np
from scipy.optimize import root_scalar
from scipy.integrate import quad
import matplotlib.pyplot as plt
import tikzplotlib

alpha, beta = 1, 2
rho, sigma = 2, 1

x_0, dx_0 = np.array([3, 2]), np.array([1, 1])
y_0, dy_0 = np.array([1, 0]), np.array([0, 1])

z1_0 = x_0 - y_0
z2_0 = dx_0
z3_0 = dy_0

def xi(t, z1, z2, z3): 
    return z1 + (1 - np.exp(-alpha*t))/alpha * z2 - (1 - np.exp(-beta*t))/beta * z3

def w(tau):
    return (1 - np.exp(-alpha*tau))/alpha * rho - (1 - np.exp(-beta*tau))/beta * sigma

def int_w(t):
    return quad(w, 0, t)[0]

T0 = root_scalar(lambda t: np.linalg.norm(xi(t, z1_0, z2_0, z3_0)) - int_w(t), bracket=(0, 100)).root
print('T0 =', T0)
print('||xi(T0, z1_0, z2_0, z3_0)|| =', np.linalg.norm(xi(T0, z1_0, z2_0, z3_0)))

t_domain = np.linspace(0, T0*1.1, 100)
plt.plot(t_domain, [int_w(t) for t in t_domain], label='$\int_0^t \omega(\\tau)d\\tau$')
plt.plot(t_domain, [np.linalg.norm(xi(t, z1_0, z2_0, z3_0)) for t in t_domain], label='$\left\Vert\\xi(T_0, z, 0)\\right\Vert$')
plt.scatter(T0, int_w(T0), color='red')
plt.legend(loc='lower right')
plt.show()
#tikzplotlib.save('./code/res_func_plot.tex')