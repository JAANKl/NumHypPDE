import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate


def init(dx, x):
    u0_ = np.zeros(len(x))
    for j in range(len(x)):
        u0_[j] = 1 / dx * integrate.quad(initial_values, x[j] - 0.5 * dx, x[j] + 0.5 * dx)[0]
    return u0_


def initial_values(x):
    return np.sin(2 * np.pi * x)


def f(x):
    return x


def f_prime(x):
    return np.ones_like(x)


def u_exact(x):
    t = tend
    return initial_values(x - t)


# takes in all values of u = (u_j^n)_j at time n and returns vector of fluxes (F_{j+1/2})_j

def rusanov_flux(u_left, u_right, dx, dt):
    return (f(u_left) + f(u_right)) / 2 - np.max([np.abs(f_prime(u_left)), np.abs(f_prime(u_right))]) / 2 * (
            u_right - u_left)


def vectorized_minmod(a, b):
    # Create a condition where a and b have opposite signs (or either is zero)
    mask = a * b <= 0
    # Where this condition is true, we return 0
    result = np.where(mask, 0, np.where(np.abs(a) < np.abs(b), a, b))
    return result


# def sigma(u, dx):
#     du = np.diff(u)
#     # boundary
#     sigma_inside = vectorized_minmod(du[1:], du[:-1])/dx
#     return np.concatenate([[sigma_inside[-1]], sigma_inside, [sigma_inside[0]]])

def sigma(u, dx):
    du = np.concatenate([np.diff(u), [0]])
    # boundary
    return vectorized_minmod(du / dx, np.roll(du, 1) / dx)


def u_minus(u, dx):
    # vector containing u_{j+1/2}^{-} for all j
    return u + sigma(u, dx) * dx / 2


def u_plus(u, dx):
    # vector containing u_{j+1/2}^{+} for all j
    return u - sigma(u, dx) * dx / 2


tend = 1.
mesh_sizes = np.array([40, 80, 160, 320, 640, 1280])
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
    # u = initial_values(x)
    u = init(dx, x)
    u = np.concatenate([[u[-1]], u, [u[0]]])
    for _ in range(int(tend / dt)):
        u[0] = u[-2]  # Apply periodic boundary conditions
        u[-1] = u[1]
        u_star = u
        u_minus_values = u_minus(u, dx)[:-1]
        u_plus_values = u_plus(u, dx)[1:]
        F_j = rusanov_flux(u_minus_values, u_plus_values, dx, dt)
        F_j_diff = F_j[1:] - F_j[:-1]
        u_star[1:-1] = u[1:-1] - dt / dx * F_j_diff
        u_minus_values = u_minus(u_star, dx)[:-1]
        u_plus_values = u_plus(u_star, dx)[1:]
        F_j = rusanov_flux(u_minus_values, u_plus_values, dx, dt)
        F_j_diff = F_j[1:] - F_j[:-1]
        u_star[1:-1] = u_star[1:-1] - dt / dx * F_j_diff
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


def latex_out(mesh_sizes, err_l1, err_l2, err_linf, rates_l1, rates_l2, rates_linf, precision=3):
    # Example output:
    """
    40 & 0.389 & - & 0.439 & - & 0.633 & - \\
    \hline
    80 & 0.244 & 0.676 & 0.274 & 0.680 & 0.394  & 0.686\\
    \hline
    160 & 0.137 &  0.826 & 0.154 & 0.829 & 0.221 & 0.835\\
    \hline
    320 & 0.073 & 0.909 &  0.082 & 0.912 & 0.117  & 0.919\\
    \hline
    640 & 0.038 & 0.952 & 0.042 & 0.955 &  0.060 & 0.960 \\
    """
    # first line
    N = mesh_sizes[0]
    i = 0
    # rounding errors
    err_l1 = np.round(err_l1, precision)
    err_l2 = np.round(err_l2, precision)
    err_linf = np.round(err_linf, precision)

    print(f"{N} & {err_l1[0]} & - & {err_l2[0]} & - & {err_linf[0]} & - \\\\")
    print(r"\hline")

    for i, N in enumerate(mesh_sizes[1:]):
        print(
            f"{N} & {err_l1[i + 1]} & {rates_l1[i]} & {err_l2[i + 1]} & {rates_l2[i]} & {err_linf[i + 1]} & {rates_linf[i]} \\\\")
        print(r"\hline")


latex_out(mesh_sizes, err_l1, err_l2, err_linf, rates_l1, rates_l2, rates_linf)
