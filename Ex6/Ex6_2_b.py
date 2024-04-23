import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sc

tend = 1.5 / np.pi


def initial_values(x):
    return np.sin(np.pi * x) + 0.5


def f(x):
    return (x ** 2) / 2


def g(x_0, t, x):
    return x_0 + (np.sin(np.pi * x_0) + 1 / 2) * t - x


def g_prime(x_0, t, x):
    return 1 + np.pi * np.cos(np.pi * x_0) * t


def u_exact(x):
    t = tend
    x_s = 1 + 1 / 2 * t

    no_prob = x <= 0.2
    prob_is_left = np.logical_and(x <= x_s, x > 1.0)
    prob_is_right = x > x_s
    init_val = 0.5 * prob_is_left + 1.5 * prob_is_right + (0) * no_prob
    x_0 = sc.newton(g, x0=init_val, fprime=g_prime, args=(t, x), tol=1e-5, maxiter=100)
    return initial_values(x_0)


mesh_sizes = np.array([40, 80, 160, 320, 640])
err_l1 = np.zeros(n := len(mesh_sizes))
err_l2 = np.zeros(n)
err_linf = np.zeros(n)
numerical_solutions = []

# number of decimals for reporting values
precision = 4


def f(u):
    return u ** 2 / 2


def f_prime(u):
    return u


def lax_wendroff_flux_local(u_left, u_right, tol=1e-7):
    if np.linalg.norm(u_left - u_right) < tol:
        a = f_prime(u_left)
    else:
        a = (f(u_right) - f(u_left)) / (u_right - u_left)
    return (f(u_left) + f(u_right)) / 2 - a * dt / (2 * dx) * (f(u_right) - f(u_left))


# takes in all values of u = (u_j^n)_j at time n and returns vector of fluxes (F_{j+1/2})_j

def lax_wendroff_flux(u):
    # periodic boundary
    u_left = np.concatenate(([u[-1]], u))
    u_right = np.concatenate((u, [u[0]]))

    return lax_wendroff_flux_local(u_left, u_right)


for i, N in enumerate(mesh_sizes):
    dx = 2 / N
    # choosing dt according to CFL condition
    dt = 2 / (8 * N)  # <= 1/(2N)

    x = np.linspace(0, 2, N)
    # Initial values:
    u = initial_values(x)
    for _ in range(int(tend / dt)):
        F_j_minus = lax_wendroff_flux(u)
        # print(F_j_minus)
        F_j_diff = F_j_minus[1:] - F_j_minus[:-1]
        u = u - dt / dx * F_j_diff
    numerical_solutions.append(u)
    err_l1[i] = np.sum(np.abs(u[:int(N/8*3)] - u_exact(x)[:int(N/8*3)])) * dx
    err_l2[i] = np.sqrt(np.sum((np.abs(u[:int(N/8*3)] - u_exact(x)[:int(N/8*3)])) ** 2) * dx)
    err_linf[i] = np.max(np.abs(u[:int(N/8*3)] - u_exact(x)[:int(N/8*3)]))

# Plotting:
for i, N in enumerate(mesh_sizes):
    plt.scatter(np.linspace(0, 0.75, int(N/8*3)), numerical_solutions[i][:int(N/8*3)], label=f"{N} mesh points", s=1)

plt.xlabel("x")
plt.ylabel("u(x)")
plt.plot(x := np.linspace(0, 0.75, int(mesh_sizes[-1]/8*3)), u_exact(x), label="exact solution")
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

# print only one numerical solution with exact solution

index = 3
N = mesh_sizes[index]
plt.plot(np.linspace(0, 0.75, int(N/8*3)), numerical_solutions[index][:int(N/8*3)], '-',
         label=f"{N} mesh points")
plt.plot(x := np.linspace(0, 0.75, int(mesh_sizes[-1]/8*3)), u_exact(x)[:int(mesh_sizes[-1]/8*3)], label="exact solution")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.legend()
plt.show()
