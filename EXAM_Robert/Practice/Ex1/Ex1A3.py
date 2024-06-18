import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate


exercise_name = "Ex_2.3a"
save_plots = False

tend = 1
x_left = 0
x_right = 1
cfl = 0.4  # = dt/dx
which_bc = "periodic"
which_scheme = "upwind"
# central, upwind, lax_wendroff_advection, lax_wendroff_general

#only for Riemann problem
u_L = -1
u_R = 1

#linear advection
a = 2


def initial_values(x):
    #return 2 * (x <= 0.5) + 1 * (x > 0.5)
    # return np.sin(np.pi * x)
    return np.sin(2*np.pi*x)
    # Bugers' equation
    # return np.where(x < 0, u_L, u_R)
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
        u[0] = u[1]  # Apply Neumann boundary conditions
        u[-1] = u[-2]
    elif which_bc == "periodic":
        u[0] = u[-2]  # Apply Periodic boundary conditions
        u[-1] = u[1]
    else:
        raise NotImplementedError("Only neumann and periodic boundary conditions possible")
    return u


def three_point_scheme_step(u_left, u_middle, u_right, dt, dx, which_scheme):
    if which_scheme == "central":
        return u_middle - a * dt / (2 * dx) * (u_right - u_left)
    elif which_scheme == "upwind":
        if a > 0:
            return u_middle - a * dt / dx * (u_middle - u_left)
        else:
            return u_middle - a * dt / dx * (u_right - u_middle)
    elif which_scheme == "lax_wendroff_advection":
        return u_middle - a * dt / (2 * dx) * (u_right - u_left) + (a * dt / dx) ** 2 / 2 * (
                u_right - 2 * u_middle + u_left)
    elif which_scheme == "lax_wendroff_general":
        return general_lax_wendroff_step(u_left, u_middle, u_right, dt, dx)
    #elif which_scheme == "beam_warming":
    else:
        raise NotImplementedError(f"{which_scheme} scheme isn't implemented.")

def general_lax_wendroff_step(u_left, u_middle, u_right, dt, dx):
    #a_j_ph := a_{j+1/2}
    #a_j_mh := a_{j-1/2}

    #a_j_ph = f_prime((u_middle+u_right)/2)
    #a_j_pm = f_prime((u_left+u_middle)/2)

    #alternative
    #take care of division by zero
    div_by_0 = np.isclose(u_right, u_middle)
    a_j_ph = np.zeros_like(u_middle)
    a_j_ph[div_by_0] = f_prime(u_middle[div_by_0])
    a_j_ph[~div_by_0] = (f(u_right[~div_by_0])-f(u_middle[~div_by_0]))/(u_right[~div_by_0]-u_middle[~div_by_0])

    a_j_mh = np.zeros_like(u_middle)
    div_by_0 = np.isclose(u_left, u_middle)
    a_j_mh[div_by_0] = f_prime(u_middle[div_by_0])
    a_j_mh[~div_by_0] = (f(u_middle[~div_by_0])-f(u_left[~div_by_0]))/(u_middle[~div_by_0]-u_left[~div_by_0])

    return u_middle - dt / (2 * dx) *(f(u_right) - f(u_left)) + (dt /dx)**2 /2 * (
            a_j_ph * (f(u_right) - f(u_middle)) - a_j_mh * (f(u_middle) - f(u_left))
        ) 
    



mesh_sizes = np.array([40, 80, 160, 320, 640]) #np.array([100]) 
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
    u = initial_values(x)
    u = np.concatenate([[u[0]], u, [u[-1]]])  # Add ghost cells
    for _ in range(int(tend / dt)):
        u = apply_bc(u, which_bc)
        u_left = u[:-2]
        u_middle = u[1:-1]
        u_right = u[2:]
        u[1:-1] = three_point_scheme_step(u_left, u_middle, u_right, dt, dx, which_scheme)

    u = u[1:-1]  # Strip ghost cells

    numerical_solutions.append(u)
    err_l1[i] = np.sum(np.abs(u - u_exact(x, tend))) * dx
    err_l2[i] = np.sqrt(np.sum((np.abs(u - u_exact(x, tend))) ** 2) * dx)
    err_linf[i] = np.max(np.abs(u - u_exact(x, tend)))

# Plotting:
mesh_index_to_plot = -1
plt.plot(np.linspace(x_left, x_right, mesh_sizes[mesh_index_to_plot]), numerical_solutions[-1], '-',
         label=f"{which_scheme}", linewidth=1)
plt.xlabel("x")
plt.ylabel("u(x)")
plt.plot(x := np.linspace(x_left, x_right, mesh_sizes[mesh_index_to_plot]), u_exact(x, tend), label="exact solution", linewidth=1,
         color="black")
plt.title(f"{mesh_sizes[mesh_index_to_plot]} Points")
plt.legend()
if save_plots:
    plt.savefig(f"{exercise_name}_plot_mesh_N={mesh_sizes[mesh_index_to_plot]}.png")

plt.show()

for i, N in enumerate(mesh_sizes):
    plt.scatter(np.linspace(x_left, x_right, N), numerical_solutions[i], label=f"{N} mesh points", s=1)
if save_plots:
    plt.savefig(f"{exercise_name}_comparison.png")
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
if save_plots:
    plt.savefig(f"{exercise_name}_convergence_rates.png")
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
