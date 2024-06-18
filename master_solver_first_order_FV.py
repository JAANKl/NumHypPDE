import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

exercise_name = "Ex_2.3a"
save_plots = False
compute_rates = True

tend = 0.5/np.pi
x_left = 0
x_right = 2
cfl = 0.4  # = dt/dx
which_bc = "periodic"
#which_bc = "neumann"
which_schemes =  ["enquist_osher"]#["roe", "lax_friedrichs", "rusanov", "enquist_osher", "godunov", "lax_wendroff"]
# lax_friedrichs, rusanov, enquist_osher, godunov, roe, lax_wendroff

mesh_sizes = np.array([40, 80, 160, 320, 640]) #np.array([100]) 
mesh_index_to_plot = -1

#only for Riemann problem
u_L = -1
u_R = 1

#only for linear advection
a = 2

def init(dx, x):
    u0_ = np.zeros(len(x))
    for j in range(len(x)):
        u0_[j] = 1 / dx * integrate.quad(initial_values, x[j] - 0.5 * dx, x[j] + 0.5 * dx)[0]  # Midpoint rule
    return u0_


def initial_values(x):
    #return 2 * (x <= 0.5) + 1 * (x > 0.5)
    # return np.sin(np.pi * x)
    # Bugers' equation
    # return np.array(np.where(x < 0, u_L, u_R), dtype=float)
    return np.sin(np.pi*x) + 0.5


def f(u):
    # Linear advection:
    # return a*u
    # Burgers' equation:
    return u ** 2 / 2


def f_prime(u):
    # Linear advection:
    # return a*np.ones_like(u)
    # Burgers' equation:
    return u

def u_exact(x, t):
    # Linear advection:
    # return initial_values(x - a*t)
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

    ##NETWON
    import scipy.optimize as sc
    t = tend
    x_s = 1 + 1 / 2 * t # Shock position

    def g(x_0, t, x):
        return x_0 + (np.sin(np.pi * x_0) + 1 / 2) * t - x


    def g_prime(x_0, t, x):
        return 1 + np.pi * np.cos(np.pi * x_0) * t

    no_prob = x <= 0.2
    prob_is_left = np.logical_and(x <= x_s, x > 1.0)
    prob_is_right = x > x_s
    init_val = 0.5 * prob_is_left + 1.5 * prob_is_right + (0) * no_prob
    x_0 = sc.newton(g, x0=init_val, fprime=g_prime, args=(t, x), tol=1e-5, maxiter=100)
    return initial_values(x_0)


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
    return (f(u_left) + f(u_right)) / 2 - integrals/2


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

err_l1 = {}
err_l2 = {}
err_linf = {}
numerical_solutions = {}

# number of decimals for reporting values
precision = 4

for which_scheme in which_schemes:
    numerical_solutions[which_scheme] = []
    err_l1[which_scheme] = np.zeros(n := len(mesh_sizes))
    err_l2[which_scheme] = np.zeros(n)
    err_linf[which_scheme] = np.zeros(n)
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

        numerical_solutions[which_scheme].append(u)
        err_l1[which_scheme][i] = np.sum(np.abs(u - u_exact(x, tend))) * dx
        err_l2[which_scheme][i] = np.sqrt(np.sum((np.abs(u - u_exact(x, tend))) ** 2) * dx)
        err_linf[which_scheme][i] = np.max(np.abs(u - u_exact(x, tend)))


# Plotting:
fig, ax = plt.subplots()
ax: plt.Axes
for which_scheme in which_schemes:
    ax.plot(np.linspace(x_left, x_right, mesh_sizes[mesh_index_to_plot]), numerical_solutions[which_scheme][mesh_index_to_plot], '-',
            label=f"{which_scheme}", linewidth=1)
ax.set_xlabel("x")
ax.set_ylabel("u(x)")
ax.plot(x := np.linspace(x_left, x_right, mesh_sizes[mesh_index_to_plot]), u_exact(x, tend), label="exact solution", linewidth=1,
         color="black")
ax.set_title(f"{mesh_sizes[mesh_index_to_plot]} Points")
ax.legend()
if save_plots:
    fig.savefig(f"{exercise_name}_plot_mesh_N={mesh_sizes[mesh_index_to_plot]}.png")
plt.show()

for which_scheme in which_schemes:
    fig, ax = plt.subplots()
    for i, N in enumerate(mesh_sizes):
        ax.scatter(np.linspace(x_left, x_right, N), numerical_solutions[which_scheme][i], label=f"{N} mesh points", s=1)
    ax.set_xlabel("x")
    ax.set_ylabel("u(x)")
    ax.plot(x := np.linspace(x_left, x_right, mesh_sizes[-1]), u_exact(x, tend), label="exact solution", color="black")
    ax.set_title(f"{which_scheme}")
    ax.legend()
    if save_plots:
        fig.savefig(f"{exercise_name}_{which_scheme}_mesh_comparison.png")
    plt.show()


if not compute_rates:
    exit()

rates_l1 = {}
rates_l2 = {}
rates_linf = {}

for which_scheme in which_schemes:
    print(f"\n-->Errors and Rates for Scheme: {which_scheme}<--\n")

    fig, ax = plt.subplots()
    ax: plt.Axes
    mesh_widths = 1 / mesh_sizes
    ax.loglog(mesh_widths, err_l1[which_scheme], label="$L^{1}$-Error")
    ax.loglog(mesh_widths, err_l2[which_scheme], label="$L^{2}$-Error")
    ax.loglog(mesh_widths, err_linf[which_scheme], label="$L^{\infty}$-Error")
    ax.loglog(mesh_widths, 10 * mesh_widths, label="$h^{1}$ (for comparison)", linestyle='dashed')
    ax.loglog(mesh_widths, 1 * mesh_widths ** 0.5, label="$h^{0.5}$ (for comparison)", linestyle='dashed')
    ax.set_xlabel("mesh width h")
    ax.set_ylabel("error")
    ax.set_title("Convergence rates for Scheme: " + which_scheme)
    ax.legend()
    if save_plots:
        fig.savefig(f"{exercise_name}_{which_scheme}_convergence_rates.png")
    plt.show()

    print("L1 average convergence rate:", np.polyfit(np.log(mesh_widths), np.log(err_l1[which_scheme]), 1)[0])
    print("L2 average convergence rate:", np.polyfit(np.log(mesh_widths), np.log(err_l2[which_scheme]), 1)[0])
    print("Linf average convergence rate:", np.polyfit(np.log(mesh_widths), np.log(err_linf[which_scheme]), 1)[0])

    print(f"N={mesh_sizes[0]}")
    print(f"L1 Error at N={mesh_sizes[0]}: {err_l1[which_scheme][0]}")
    print(f"L2 Error  at N={mesh_sizes[0]}: {err_l2[which_scheme][0]}")

    print(f"Linf Error at N={mesh_sizes[0]}: {err_linf[which_scheme][0]}")

    rates_l1[which_scheme] = []
    rates_l2[which_scheme] = []
    rates_linf[which_scheme] = []
    for i, N in enumerate(mesh_sizes[1:]):
        print(f"N={N}")
        print(f"L1 Error at N={N}:", err_l1[which_scheme][i + 1])
        print(f"L2 Error  at N={N}:", err_l2[which_scheme][i + 1])
        print(f"Linf Error at N={N}:", err_linf[which_scheme][i + 1])
        rate_l1 = np.polyfit(np.log(mesh_widths[i:i + 2]), np.log(err_l1[which_scheme][i:i + 2]), 1)[0]
        rate_l2 = np.polyfit(np.log(mesh_widths[i:i + 2]), np.log(err_l2[which_scheme][i:i + 2]), 1)[0]
        rate_linf = np.polyfit(np.log(mesh_widths[i:i + 2]), np.log(err_linf[which_scheme][i:i + 2]), 1)[0]
        rates_l1[which_scheme].append(np.round(rate_l1, precision))
        rates_l2[which_scheme].append(np.round(rate_l2, precision))
        rates_linf[which_scheme].append(np.round(rate_linf, precision))

        print(f"L1 local convergence rate at N={N} :", rate_l1)
        print(f"L2 local convergence rate  at N={N}:", rate_l2)
        print(f"Linf local  convergence rate at N={N}:", rate_linf)
