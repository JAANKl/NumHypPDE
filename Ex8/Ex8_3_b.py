import numpy as np
import matplotlib.pyplot as plt

def initial_values(x):
    return np.sin(4 * np.pi * x)


def f(x):
    return x


def f_prime(x):
    return np.ones_like(x)


def u_exact(x):
    t = tend
    return initial_values(x - t)


# takes in all values of u = (u_j^n)_j at time n and returns vector of fluxes (F_{j+1/2})_j

def godunov_flux(u_left, u_right, dx, dt):
    return np.minimum(u_left, u_right)


def minmod(a):
    # if not all entries have the same sign, return 0
    if np.all(a > 0) or np.all(a < 0):
        return np.min(np.abs(a))
    else:
        return 0


def superbee(a, b):
    if a * b <= 0:
        return 0
    else:
        return max(min(2 * abs(a), b), min(a, 2 * abs(b)))


def van_leer(a, b):
    if a * b <= 0:
        return 0
    else:
        return 2 * a * b / (a + b)


def mc_limiter(a, b):
    return max(0, min(min(2 * a, (a + b) / 2), 2 * b))


def sigma(u, dx):
    du = np.diff(u)
    # boundary
    return minmod(np.array([du / dx, np.roll(du, 1) / dx]))


def u_minus(u, dx):
    # vector containing u_{j+1/2}^{-} for all j
    return u + sigma(u, dx) * dx / 2


def u_plus(u, dx):
    # vector containing u_{j+1/2}^{+} for all j
    return u - sigma(u, dx) * dx / 2


tend = 1.
mesh_sizes = np.array([100])
err_l1 = np.zeros(n := len(mesh_sizes))
err_l2 = np.zeros(n)
err_linf = np.zeros(n)
numerical_solutions = []

# number of decimals for reporting values
precision = 4
for i, N in enumerate(mesh_sizes):
    dx = 1 / N
    # choosing dt according to CFL condition
    dt = 1 / (2 * N)  # <= 1/(2N)

    x = np.linspace(0, 1, N)
    # Initial values:
    # u = initial_values_average(x, dx)
    u = initial_values(x)
    u = np.concatenate([[u[-1]], u, [u[0]]])
    for _ in range(int(tend / dt)):
        u[0] = u[-2]  # Apply periodic boundary conditions
        u[-1] = u[1]
        u_star = u
        u_minus_values = u_minus(u, dx)[:-1]
        u_plus_values = u_plus(u, dx)[1:]
        F_j = godunov_flux(u_minus_values, u_plus_values, dx, dt)
        F_j_diff = F_j[1:] - F_j[:-1]
        u[1:-1] = u[1:-1] - dt / dx * F_j_diff
        u_star[1:-1] = u[1:-1] - dt / dx * F_j_diff
        u[1:-1] = (u_star[1:-1] + u[1:-1]) / 2

    u = u[1:-1]

    numerical_solutions.append(u)
    err_l1[i] = np.sum(np.abs(u - u_exact(x))) * dx
    err_l2[i] = np.sqrt(np.sum((np.abs(u - u_exact(x))) ** 2) * dx)
    err_linf[i] = np.max(np.abs(u - u_exact(x)))

# print only one numerical solution with exact solution

index = 0
for index, mesh_size in enumerate(mesh_sizes):
    plt.plot(np.linspace(0, 1, mesh_size), numerical_solutions[index], '-',
             label=f"{mesh_sizes[index]} mesh points", linewidth=0.5)
    plt.xlabel("x")
    plt.ylabel("u(x)")
plt.plot(x := np.linspace(0, 1, mesh_sizes[-1]), u_exact(x), label="exact solution", linewidth=0.5)
plt.legend()
plt.show()
