import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

exercise_name = "Ex_2.3a"
save_plots = False
compute_rates = True

tend = 1
x_left = 0
x_right = 1
cfl = 0.4  # = dt/dx
which_bc = "periodic"
#which_bc = "neumann"
which_schemes =  ["lax_wendroff"]#["upwind", "lax_wendroff", "beam_warming", "minmod", "superbee", "mc", "vanleer"]
# upwind, lax_wendroff, beam_warming, minmod, superbee, mc, vanleer

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
    return np.sin(2*np.pi * x)
    # Bugers' equation
    # return np.array(np.where(x < 0, u_L, u_R), dtype=float)
    # return np.sin(np.pi*x) + 0.5


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
        u[0:2] = u[2]  # Apply Neumann boundary conditions
        u[-2:] = u[-3]
    elif which_bc == "periodic":
        u[0:2] = u[-5:-3]  # Apply Periodic boundary conditions
        u[-2:] = u[3:5]
    else:
        raise NotImplementedError("Only neumann and periodic boundary conditions possible")
    return u

#upwind, lax_wendroff, beam_warming, minmod, superbee, mc, vanleer
def upwind_flux_limiter(u_left, u_middle, u_right):
    return np.zeros_like(u_left)

def lax_wendroff_flux_limiter(u_left, u_middle, u_right):
    return np.ones_like(u_left)

def beam_warming_flux_limiter(u_left, u_middle, u_right):
    return (u_middle-u_left)/(u_right-u_middle)

def minmod_flux_limiter(u_left, u_middle, u_right):
    return np.zeros_like(u_left)

def superbee_flux_limiter(u_left, u_middle, u_right):
    return np.zeros_like(u_left)

def mc_flux_limiter(u_left, u_middle, u_right):
    return np.zeros_like(u_left)

def vanleer_flux_limiter(u_left, u_middle, u_right):
    return np.zeros_like(u_left)


def flux_limiter_phi(u_left, u_middle, u_right, which_scheme):
    if which_scheme == "upwind":
        return upwind_flux_limiter(u_left, u_middle, u_right)
    elif which_scheme == "lax_wendroff":
        return lax_wendroff_flux_limiter(u_left, u_middle, u_right)
    elif which_scheme == "beam_warming":
        return beam_warming_flux_limiter(u_left, u_middle, u_right)
    elif which_scheme == "minmod":
        return minmod_flux_limiter(u_left, u_middle, u_right)
    elif which_scheme == "superbee":
        return superbee_flux_limiter(u_left, u_middle, u_right)
    elif which_scheme == "mc":
        return mc_flux_limiter(u_left, u_middle, u_right)
    elif which_scheme == "vanleer":
        return vanleer_flux_limiter(u_left, u_middle, u_right)
    else:
        raise NotImplementedError(f"{which_scheme} scheme isn't implemented.")

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
        u = np.concatenate([[u[0]],[u[0]], u, [u[-1]], [u[-1]]])  # Add ghost cells
        for _ in range(int(tend / dt)):
            u = apply_bc(u, which_bc)
            u_left = u[:-2]
            u_middle = u[1:-1]
            u_right = u[2:]
            phi = flux_limiter_phi(u_left, u_middle, u_right, which_scheme)
            F_j = a*(u_middle + phi/2*(u_right - u_left))
            F_j_diff = F_j[1:] - F_j[:-1]
            u[2:-2] = u[2:-2] - dt / dx * F_j_diff[:-1]

        u = u[2:-2]  # Strip ghost cells

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
