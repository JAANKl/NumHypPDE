import numpy as np
import matplotlib.pyplot as plt

tend = 1


def initial_values(x):
    return -np.ones(N) + 2 * (x > 0)


def f(x):
    return x


def u_exact(x):
    t = tend
    return initial_values(x - t)


mesh_sizes = np.array([100])
err_l1 = np.zeros(n := len(mesh_sizes))
err_l2 = np.zeros(n)
err_linf = np.zeros(n)
numerical_solutions = []

# number of decimals for reporting values
precision = 4


def beam_warming_flux_local(u_left, u_right):
    a = 1
    return (3*f(u_left) - f(u_right)) / 2 - a * dt / (2 * dx) * (f(u_right) - f(u_left))

# takes in all values of u = (u_j^n)_j at time n and returns vector of fluxes (F_{j+1/2})_j

def beam_warming_flux(u):
    # boundary
    u_left = np.concatenate(([u[0]], u))
    u_right = np.concatenate((u, [u[-1]]))

    return beam_warming_flux_local(u_left, u_right)


for i, N in enumerate(mesh_sizes):
    dx = 10 / N
    # choosing dt according to CFL condition
    dt = 10 / (20 * N)  # <= 1/(2N)

    x = np.linspace(-5, 5, N)
    # Initial values:
    u = initial_values(x)
    for _ in range(int(tend / dt)):
        F_j = beam_warming_flux(u)
        F_j_diff = F_j[1:] - F_j[:-1]
        u = u - dt / dx * F_j_diff
    numerical_solutions.append(u)
    err_l1[i] = np.sum(np.abs(u - u_exact(x))) * dx
    err_l2[i] = np.sqrt(np.sum((np.abs(u - u_exact(x))) ** 2) * dx)
    err_linf[i] = np.max(np.abs(u - u_exact(x)))


# print only one numerical solution with exact solution

index = 0
plt.plot(np.linspace(-5, 5, mesh_sizes[index]), numerical_solutions[index], '-',
         label=f"{mesh_sizes[index]} mesh points")
plt.plot(x := np.linspace(-5, 5, mesh_sizes[-1]), u_exact(x), label="exact solution")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.legend()
plt.show()
