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


def beam_warming_flux_diff(u):
    flux_diff = np.zeros(len(u) - 1)  # Adjust the length of fh
    a = 1
    flux_diff[2:] = a * (3 * u[2:-1] - 4 * u[1:-2] + u[:-3])/2 + a ** 2 / 2 * dt/dx * (u[2:-1] - 2 * u[1:-2] + u[:-3])
    flux_diff[0] = flux_diff[2]
    flux_diff[1] = flux_diff[2]
    return flux_diff[1:]


for i, N in enumerate(mesh_sizes):
    dx = 10 / N
    # choosing dt according to CFL condition
    dt = 10 / (20 * N)  # <= 1/(2N)

    x = np.linspace(-5, 5, N)
    # Initial values:
    u = initial_values(x)
    for _ in range(int(tend / dt)):
        F_j_diff = beam_warming_flux_diff(u)
        u[1:-1] = u[1:-1] - dt / dx * F_j_diff
        u[0] = u[1]
        u[-1] = u[-2]
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
