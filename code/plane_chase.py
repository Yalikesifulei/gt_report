import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib

def RungeKutta(f, t0, y0, b, h):
    """
    Solve y' = f(t, y), y(t0) = y0 for 
    vector y and scalar t for t in [t0, b]
    with step size h
    """
    n = int(np.floor((b-t0)/h))
    t, y = t0, y0.copy()
    t_hist, y_hist = [t], [y]
    for _ in range(n):
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2*k1)
        k3 = f(t + h/2, y + h/2*k2)
        k4 = f(t + h, y + h*k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t = t + h
        t_hist.append(t), y_hist.append(y)
    return {'t': np.array(t_hist),
            'y': np.array(y_hist).reshape((len(y_hist), -1))}

alpha, beta = 3, 1
f = lambda t, y: np.array([
    -alpha * (y[:2] - y[2:])/np.linalg.norm(y[:2] - y[2:]),
    #-beta * (y[:2] - y[2:])/np.linalg.norm(y[:2] - y[2:])
    -beta * np.array([np.cos(t), np.sin(t)])
]).flatten()
x0, y0 = np.array([0, 0]), np.array([1, 1])
T = np.linalg.norm(x0 - y0)/(alpha - beta)

res = RungeKutta(f, 0, np.array([x0, y0]).flatten(), T*1.1, 0.005)
plt.plot(res['y'][:, 0], res['y'][:, 1], label='$x(t)$', color='#FF0000', alpha=0.5)
plt.plot(res['y'][:, 2], res['y'][:, 3], label='$y(t)$', color='#00FF00', linestyle='dashed', linewidth=3)
plt.legend()
print(f'T = {T}, last_point = {res["y"][-1]}')
dist = ((res['y'][:, :2] - res['y'][:, 2:])**2).sum(axis=1)
print(np.argmin(dist), np.min(dist))
print(res['t'][np.argmin(dist)])
print(res['y'][np.argmin(dist)])
#plt.show()
#tikzplotlib.save('./code/unoptimal_chase_2d.tex')