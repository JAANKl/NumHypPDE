import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate


def init(dx, x):
    u0_ = np.zeros(len(x))
    for j in range(len(x)):
        u0_[j] = 1 / dx * integrate.quad(initial_values, x[j] - 0.5 * dx, x[j] + 0.5 * dx)[0]  # Midpoint rule
    return u0_


u_L = -1
u_R = 1


def initial_values(x):
    return 2 * (x <= 0.5) + 1 * (x > 0.5)
    # return np.sin(np.pi * x)
    # Bugers' equation
    # return np.where(x < 0, u_L, u_R)
    # return np.sin(np.pi*x) + 0.5


a = 1


def f(u):
    # Linear advection:
    return a*u
    # Burgers' equation:
    # return u ** 2 / 2


def f_prime(u):
    # Linear advection:
    return a*np.ones_like(u)
    # Burgers' equation:
    # return u


def u_exact(x, t):
    # Linear advection:
    return initial_values(x - a*t)
    # Burgers' equation shock: (u_L > u_R)
    # s = (f(u_L) - f(u_R)) / (u_L - u_R)
    # return np.where((x < s*t), u_L, u_R)
    # Burgers' equation rarefaction: (u_L < u_R)
    # u = np.zeros(len(x))
    # for i in range(len(x)):
    #     if x[i] <= f_prime(u_L) * t:
    #         u[i] = u_L
    #     elif x[i] <= f_prime(u_R) * t:
    #         u[i] = x[i] / t
    #     else:
    #         u[i] = u_R
    # return u


def apply_bc(u, which_bc):
    if which_bc == "neumann":
        u[0] = u[1]  # Apply Neumann boundary conditions
        u[-1] = u[-2]
    elif which_bc == "periodic":
        u[0] = u[-2]  # Apply Periodic boundary conditions
        u[-1] = u[1]
    else:
        raise NotImplementedError("Only neumann and periodic boundary conditions possible")
    return u


def get_flux(u_left, u_right, which_scheme):
    if which_scheme == "lax_friedrichs":
        return lax_friedrichs_flux(u_left, u_right)
    elif which_scheme == "rusanov":
        return rusanov_flux(u_left, u_right)
    elif which_scheme == "enquist_osher":
        return enquist_osher_flux(u_left, u_right)
    elif which_scheme == "godunov":
        return godunov_flux(u_left, u_right)
    elif which_scheme == "roe":
        return roe_flux(u_left, u_right)
    elif which_scheme == "lax_wendroff":
        return lax_wendroff_flux(u_left, u_right)
    else:
        raise NotImplementedError(f"{which_scheme} scheme isn't implemented.")



def lax_friedrichs_flux(u_left, u_right):
    return 0.5 * (f(u_left) + f(u_right)) - 0.5 * (u_right - u_left) / cfl


def rusanov_flux(u_left, u_right):
    return (f(u_left) + f(u_right)) / 2 - np.max([np.abs(f_prime(u_left)), np.abs(f_prime(u_right))]) / 2 * (
            u_right - u_left)


def enquist_osher_flux(u_left, u_right):
    integrand = lambda theta: np.abs(f_prime(theta))
    integrals = np.zeros_like(u_left)
    for i in range(len(integrals)):
        integrals[i] = integrate.quad(integrand, u_left[i], u_right[i])[0]
    return (f(u_left) + f(u_right)) / 2 - integrals


def godunov_flux(u_left, u_right):
    fh = np.zeros(len(u_left))
    for i in range(len(u_left)):
        if u_left[i] <= u_right[i]:
            uu = np.linspace(u_left[i], u_right[i], 100)
            ff = f(uu) * np.ones(len(uu))
            fh[i] = min(ff)
        else:
            uu = np.linspace(u_left[i], u_right[i], 100)
            ff = f(uu) * np.ones(len(uu))
            fh[i] = max(ff)
    return fh


def roe_flux(u_left, u_right):
    A_hat = np.zeros(len(u_left))
    for j in range(len(u_left)):
        if np.abs(u_left[j] - u_right[j]) < 1e-7:
            A_hat[j] = f_prime(u_left[j])
        else:
            A_hat[j] = (f(u_right[j]) - f(u_left[j])) / (u_right[j] - u_left[j])
    fh = np.where(A_hat >= 0, f(u_left), f(u_right))
    return fh

def lax_wendroff_flux(u_left, u_right):
    A_hat = np.zeros(len(u_left))
    for j in range(len(u_left)):
        if np.abs(u_left[j] - u_right[j]) < 1e-7:
            A_hat[j] = f_prime(u_left[j])
        else:
            A_hat[j] = (f(u_right[j]) - f(u_left[j])) / (u_right[j] - u_left[j])
    return (f(u_left) + f(u_right)) / 2 - A_hat*cfl/2 * (f(u_right) - f(u_left))


tend = 1
x_left = -2
x_right = 2
cfl = 0.5  # = dt/dx
which_bc = "neumann"
# which_bc = "periodic"
which_scheme = "roe"
# lax_friedrichs, rusanov, enquist_osher, godunov, roe, lax_wendroff
# TODO: implement and upwind scheme (or is this the same as roe?)
mesh_sizes = np.array([32, 64, 128, 256, 512])
err_l1 = np.zeros(n := len(mesh_sizes))
err_l2 = np.zeros(n)
err_linf = np.zeros(n)
numerical_solutions = []

# number of decimals for reporting values
precision = 4

for i, N in enumerate(mesh_sizes):
    dx = (x_right - x_left) / N
    dt = cfl * dx

    x = np.linspace(x_left, x_right, N)
    u = init(dx, x)
    u = np.concatenate([[u[0]], u, [u[-1]]])  # Add ghost cells
    for _ in range(int(tend / dt)):
        u = apply_bc(u, which_bc)
        F_j = get_flux(u[:-1], u[1:], which_scheme)
        F_j_diff = F_j[1:] - F_j[:-1]
        u[1:-1] = u[1:-1] - dt / dx * F_j_diff

    u = u[1:-1]  # Strip ghost cells

    numerical_solutions.append(u)
    err_l1[i] = np.sum(np.abs(u - u_exact(x, tend))) * dx
    err_l2[i] = np.sqrt(np.sum((np.abs(u - u_exact(x, tend))) ** 2) * dx)
    err_linf[i] = np.max(np.abs(u - u_exact(x, tend)))

# Plotting:

plt.plot(np.linspace(x_left, x_right, mesh_sizes[-1]), numerical_solutions[-1], '-',
         label=f"{which_scheme}", linewidth=1)
plt.xlabel("x")
plt.ylabel("u(x)")
plt.plot(x := np.linspace(x_left, x_right, mesh_sizes[-1]), u_exact(x, tend), label="exact solution", linewidth=1,
         color="black")
plt.title(f"{mesh_sizes[-1]} Points")
plt.legend()
plt.show()

for i, N in enumerate(mesh_sizes):
    plt.scatter(np.linspace(x_left, x_right, N), numerical_solutions[i], label=f"{N} mesh points", s=1)

plt.xlabel("x")
plt.ylabel("u(x)")
plt.plot(x := np.linspace(x_left, x_right, mesh_sizes[-1]), u_exact(x, tend), label="exact solution", color="black")
plt.legend()
plt.show()
mesh_widths = 1 / mesh_sizes
plt.loglog(mesh_widths, err_l1, label="$L^{1}$-Error")
plt.loglog(mesh_widths, err_l2, label="$L^{2}$-Error")
plt.loglog(mesh_widths, err_linf, label="$L^{\infty}$-Error")
plt.loglog(mesh_widths, 10 * mesh_widths, label="$h^{1}$ (for comparison)", linestyle='dashed')
plt.loglog(mesh_widths, 1 * mesh_widths ** 0.5, label="$h^{0.5}$ (for comparison)", linestyle='dashed')
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
