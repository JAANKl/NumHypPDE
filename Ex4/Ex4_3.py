import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sc


def initial_values(x):
    return np.sin(np.pi * x) + 0.5


def f(x):
    return (x ** 2) / 2


def F(j, u_GOD):
    return np.max((f(np.max(u_GOD[j], 0)), f(np.min(u_GOD[(j + 1) % N], 0))))


def g(x_0, t, x):
    return x_0 + (np.sin(np.pi * x_0) + 1 / 2) * t - x


def g_prime(x_0, t, x):
    return 1 + np.pi * np.cos(np.pi * x_0) * t


def u_exact(x):
    x_s = 1 + 1 / 2 * tend

    no_prob = x <= 0.2
    prob_is_left = np.logical_and(x <= x_s, x > 1.0)
    prob_is_right = x > x_s
    init_val = 0.5 * prob_is_left + 1.5 * prob_is_right + (0) * no_prob
    x_0 = sc.newton(g, x0=init_val, fprime=g_prime, args=(t, x), tol=1e-16, maxiter=100)
    return initial_values(x_0)


def time_step(u_GOD):
    u_copy = u_GOD.copy()
    # calculate first entry using periodic bc:
    u_GOD[0] += -dt / dx * (F(0, u_copy) - F(N - 1, u_copy))
    u_GOD[N - 1] = u_GOD[0]
    for j in range(1, N - 1):
        u_GOD[j] += -dt / dx * (F(j, u_copy) - F(j - 1, u_copy))
    return u_GOD


tend = 1.5 / np.pi
N = 100
dx = 2 / N
dt = 1 / (4 * N)  # <= 1/(2N)
x = np.linspace(0, 2, N)
# Initial values:
u_GOD = initial_values(x)
t = 0
while t < tend:
    u_GOD = time_step(u_GOD)
    t += dt

# Plotting
plt.scatter(x, u_GOD, label="Godunov", s=0.5)
plt.plot(x, u_exact(x), label="exact solution")
plt.ylim((-2, 2))
plt.legend()
plt.show()
