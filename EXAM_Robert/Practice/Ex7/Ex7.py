import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

exercise_name = "Ex_2.3a_Beam_Warming"
save_plots = True


def init(dx, x):
    u0_ = np.zeros(len(x))
    for j in range(len(x)):
        u0_[j] = 1 / dx * integrate.quad(initial_values, x[j] - 0.5 * dx, x[j] + 0.5 * dx)[0]  # Midpoint rule
    return u0_


u_L = -1
u_R = 1


def initial_values(x):
    return np.where(x < 0, u_L, u_R)



a = 1


def f(u):
    # Linear advection:
    return a*u



def f_prime(u):
    # Linear advection:
    return a*np.ones_like(u)
    # Burgers' equation:
    # return u


def u_exact(x, t):
    # Linear advection:
    return initial_values(x - a*t)

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


def vectorized_minmod(a, b):
    # Create a condition where a and b have opposite signs (or either is zero)
    mask = a * b <= 0
    # Where this condition is true, we return 0
    result = np.where(mask, 0, np.where(np.abs(a) < np.abs(b), a, b))
    return result

def vectorized_maxmod(a, b):
    # Create a condition where a and b have opposite signs (or either is zero)
    mask = a * b <= 0
    # Where this condition is true, we return 0
    result = np.where(mask, 0, np.where(np.abs(a) > np.abs(b), a, b))
    return result

def sigma_minmod(u, dx):
    du = np.diff(u)
    # boundary
    sigma_inside = vectorized_minmod(du[1:], du[:-1])/dx
    return np.concatenate([[sigma_inside[0]], sigma_inside, [sigma_inside[-1]]])

def sigma_superbee(u, dx):
    du = np.diff(u)
    # boundary
    sigma_l_inside = vectorized_minmod(2*du[:-1], du[1:])/dx
    sigma_r_inside = vectorized_minmod(du[:-1], 2*du[1:])/dx
    sigma_inside = vectorized_maxmod(sigma_l_inside, sigma_r_inside)
    return np.concatenate([[sigma_inside[0]], sigma_inside, [sigma_inside[-1]]])

def general_minmod(a):
    # if not all entries have the same sign, return 0
    minmod_condition = np.logical_or(np.all(a>0, axis=1), np.all(a<0, axis=1))
    return np.min(np.abs(a), axis = 1)*(minmod_condition)

def sigma_mc(u, dx):
    du = np.diff(u)
    right_diff = du[1:]
    left_diff = du[:-1]
    central_diff = (right_diff+left_diff)/2
    # boundary
    data = np.concatenate((right_diff[:, None], central_diff[:, None], left_diff[:, None]),axis=1)/dx
    sigma_inside = general_minmod(data)
    return np.concatenate([[sigma_inside[0]], sigma_inside, [sigma_inside[-1]]])

def mc(a,b):
    return np.maximum(0, np.sign(a) * np.minimum(np.abs(2*a), np.minimum(np.abs((a+b)/2), np.abs(2*b))))

def sigma_van_leer(u, dx):
    du = np.diff(u)
    # boundary
    divbyzero = (du[:-1]==0)
    du[:-1][divbyzero] = 1
    r = (1-divbyzero)*du[1:]/du[:-1]
    sigma_inside = (r+np.abs(r))/(1+np.abs(r))
    return np.concatenate([[sigma_inside[0]], sigma_inside, [sigma_inside[-1]]])



def u_minus(u, dx, sigma_func):
    # vector containing u_{j+1/2}^{-} for all j
    return u + sigma_func(u, dx) * dx / 2


def u_plus(u, dx, sigma_func):
    # vector containing u_{j+1/2}^{+} for all j
    return u - sigma_func(u, dx) * dx / 2

#no limiter
def sigma_zero(u, dx):
    return np.zeros_like(u)

def runge_kutta_ssp_step(u, dx, slope, dt, which_scheme):
        F_j = get_flux(u_minus(u, dx, slope)[:-1], u_plus(u, dx, slope)[1:], which_scheme)
        F_j_diff = F_j[1:] - F_j[:-1]
        u_ = u.copy()
        u_[2:-2] = u_[2:-2] - dt / dx * F_j_diff[1:-1]
        F_j = get_flux(u_minus(u_, dx, slope)[:-1], u_plus(u_, dx, slope)[1:], which_scheme)
        F_j_diff = F_j[1:] - F_j[:-1]
        u_[2:-2] = u_[2:-2] - dt / dx * F_j_diff[1:-1]
        u[2:-2] = (u[2:-2] + u_[2:-2])/2
        #no return since inplace operation

def forward_euler_step(u, dx, slope, dt, which_scheme):
    F_j = get_flux(u_minus(u, dx, slope)[:-1], u_plus(u, dx, slope)[1:], which_scheme)
    F_j_diff = F_j[1:] - F_j[:-1]
    u[2:-2] = u[2:-2] - dt / dx * F_j_diff[1:-1]

tend = 1.0
x_left = -5
x_right = 5
cfl = 0.5  # = dt/dx
which_bc = "neumann"
which_scheme = "lax_wendroff"
# lax_friedrichs, rusanov, enquist_osher, godunov, roe, lax_wendroff
# TODO: implement upwind scheme (or is this the same as roe?)
slope_limiters = {"minmod": sigma_minmod, "mc": sigma_mc, "van_leer": sigma_van_leer, "superbee": sigma_superbee, "no_limiter": sigma_zero}
slope = slope_limiters["no_limiter"]
mesh_sizes = np.array([100, 200, 300, 500, 1000, 5000])
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
    u = np.concatenate([[u[0]], [u[0]], u, [u[-1]], [u[-1]]])  # Add ghost cells
    for _ in range(int(tend / dt)):
        u = apply_bc(u, which_bc)
        #runge_kutta_ssp_step(u, dx, slope, dt, which_scheme)
        forward_euler_step(u, dx, slope, dt, which_scheme)

    u = u[2:-2]  # Strip ghost cells

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
plt.title(f"{mesh_sizes[-1]} Points")
plt.legend()
if save_plots:
    plt.savefig(f"{exercise_name}_plot_mesh_N={mesh_sizes[mesh_index_to_plot]}.png")
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
plt.loglog(mesh_widths, 1000 * mesh_widths ** 2, label="$h^{2}$ (for comparison)", linestyle='dashed')
plt.xlabel("mesh width h")
plt.ylabel("error")
plt.legend()
if save_plots:
    plt.savefig(f"{exercise_name}_plot_mesh_comparison.png")
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
