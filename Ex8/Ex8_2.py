import numpy as np
import matplotlib.pyplot as plt

tend = 1.


def initial_values(x):
    return np.sin(2*np.pi*x)


def f(x):
    return x


def f_prime(x):
    return np.ones_like(x)


def u_exact(x):
    t = tend
    return initial_values(x - t)


mesh_sizes = np.array([10, 40, 80, 160, 320, 640, 1280])
err_l1 = np.zeros(n := len(mesh_sizes))
err_l2 = np.zeros(n)
err_linf = np.zeros(n)
numerical_solutions = []

# number of decimals for reporting values
precision = 4


# takes in all values of u = (u_j^n)_j at time n and returns vector of fluxes (F_{j+1/2})_j

def rusanov_flux(u_left, u_right, dx, dt):
    return (f(u_left) + f(u_right)) / 2 - np.max([np.abs(f_prime(u_left)), np.abs(f_prime(u_right))]) / 2 * (u_right - u_left)

# TODO: I think I am using this incorrectly
def minmod(a):
    # if not all entries have the same sign, return 0
    if np.all(a > 0) or np.all(a < 0):
        return np.min(np.abs(a))
    else:
        return 0


def sigma(u, dx):
    du = np.diff(u)
    # boundary
    return minmod(np.array([du / dx, np.roll(du,1)/dx]))


def u_minus(u, dx):
    # vector containing u_{j+1/2}^{-} for all j
    return u + sigma(u, dx) * dx / 2


def u_plus(u, dx):
    # vector containing u_{j+1/2}^{+} for all j
    return u - sigma(u, dx) * dx / 2


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
        F_j = rusanov_flux(u_minus_values, u_plus_values, dx, dt)
        F_j_diff = F_j[1:] - F_j[:-1]
        u[1:-1] = u[1:-1] - dt / dx * F_j_diff
        u_star[1:-1] = u[1:-1] - dt / dx * F_j_diff
        u[1:-1] = (u_star[1:-1] + u[1:-1])/2

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

mesh_widths = 1 / mesh_sizes
plt.loglog(mesh_widths, err_l1, label="$L^{1}$-Error")
plt.loglog(mesh_widths, err_l2, label="$L^{2}$-Error")
plt.loglog(mesh_widths, err_linf, label="$L^{\infty}$-Error")
plt.loglog(mesh_widths, 10 * mesh_widths, label="$h^{1}$ (for comparison)")
plt.loglog(mesh_widths, 10 * mesh_widths ** 0.5, label="$h^{0.5}$ (for comparison)")
plt.xlabel("mesh width h")
plt.ylabel("error")
plt.legend()
plt.show()

print("L1 average convergence rate:", np.polyfit(np.log(mesh_widths), np.log(err_l1), 1)[0])
print("L2 average convergence rate:", np.polyfit(np.log(mesh_widths), np.log(err_l2), 1)[0])
print("Linf average convergence rate:", np.polyfit(np.log(mesh_widths), np.log(err_linf), 1)[0])

print(f"N={mesh_sizes[0]}")
print(f"L1 Error at N={mesh_sizes[0]}: {err_l1[0]}")
print(f"L2 Error  at N={mesh_sizes[0]}: {err_l2[0]}")

print(f"Linf Error at N={mesh_sizes[0]}: {err_linf[0]}")
rates_l1 = []
rates_l2 = []
rates_linf = []
for i, N in enumerate(mesh_sizes[1:]):
    print(f"N={N}")
    print(f"L1 Error at N={N}:", err_l1[i + 1])
    print(f"L2 Error  at N={N}:", err_l2[i + 1])
    print(f"Linf Error at N={N}:", err_linf[i + 1])
    rate_l1 = np.polyfit(np.log(mesh_widths[i:i + 2]), np.log(err_l1[i:i + 2]), 1)[0]
    rate_l2 = np.polyfit(np.log(mesh_widths[i:i + 2]), np.log(err_l2[i:i + 2]), 1)[0]
    rate_linf = np.polyfit(np.log(mesh_widths[i:i + 2]), np.log(err_linf[i:i + 2]), 1)[0]
    rates_l1.append(np.round(rate_l1, precision))
    rates_l2.append(np.round(rate_l2, precision))
    rates_linf.append(np.round(rate_linf, precision))

    print(f"L1 local convergence rate at N={N} :", rate_l1)
    print(f"L2 local convergence rate  at N={N}:", rate_l2)
    print(f"Linf local  convergence rate at N={N}:", rate_linf)
