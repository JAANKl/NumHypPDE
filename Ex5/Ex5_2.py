import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sc


def initial_values(x):
    return np.sin(np.pi * x) + 0.5


def f(x):
    return (x ** 2) / 2


def f_prime(x):
    return x


def g(x_0, t, x):
    return x_0 + (np.sin(np.pi * x_0) + 1 / 2) * t - x


def g_prime(x_0, t, x):
    return 1 + np.pi * np.cos(np.pi * x_0) * t


tend = 0.5 / np.pi


def u_exact(x):
    t = tend
    x_s = 1 + 1 / 2 * t

    no_prob = x <= 0.2
    prob_is_left = np.logical_and(x <= x_s, x > 1.0)
    prob_is_right = x > x_s
    init_val = 0.5 * prob_is_left + 1.5 * prob_is_right + (0) * no_prob
    x_0 = sc.newton(g, x0=init_val, fprime=g_prime, args=(t, x), tol=1e-5, maxiter=100)
    return initial_values(x_0)


# number of decimals for reporting values
precision = 4


# takes in all values of u = (u_j^n)_j at time n and returns vector of fluxes (F_{j+1/2})_j

def godunov_flux(u, dx, dt):
    # periodic boundary
    u_left = np.concatenate(([u[-1]], u))
    u_right = np.concatenate((u, [u[0]]))

    return np.maximum(f(np.maximum(u_left, 0)), f(np.minimum(u_right, 0)))


def roe_flux(u, dx, dt):
    # periodic boundary
    u_left = np.concatenate(([u[-1]], u))
    u_right = np.concatenate((u, [u[0]]))

    is_equal = (u_left == u_right)
    is_different = np.logical_not(is_equal)
    # avoid division by zero, if equal replace by f_prime(U-J^n)

    A_j = np.zeros_like(u_left)
    A_j[is_different] = (f(u_right) - f(u_left))[is_different] / (u_right - u_left)[is_different]
    A_j[is_equal] = f_prime(u_left)[is_equal]

    # check if A_j is correct for Burger's equation
    assert np.allclose(A_j, (u_right + u_left) / 2)

    return (A_j >= 0) * f(u_left) + (A_j < 0) * f(u_right)


def lax_friedrichs_flux(u, dx, dt):
    # periodic boundary
    u_left = np.concatenate(([u[-1]], u))
    u_right = np.concatenate((u, [u[0]]))

    return 0.5 * (f(u_left) + f(u_right)) - 0.5 * dx / (dt) * (u_right - u_left)


def rusanov(u, dx, dt):
    # periodic boundary
    u_left = np.concatenate(([u[-1]], u))
    u_right = np.concatenate((u, [u[0]]))

    return 0.5 * (f(u_left) + f(u_right)) - np.maximum(np.abs(f_prime(u_left)), np.abs(f_prime(u_right))) * (
                u_right - u_left)


def enquist_osher_flux(u, dx, dt):
    # works as long as f has a unique minimum at 0
    # periodic boundary
    u_left = np.concatenate(([u[-1]], u))
    u_right = np.concatenate((u, [u[0]]))

    return f(np.maximum(u_left, 0)) + f(np.minimum(u_right, 0)) - f(0)


def get_dt_by_cfl(u, dx):
    return dx / (2 * np.max(np.abs(f_prime(u))))


def latex_out(mesh_sizes, err_l1, err_l2, err_linf, rates_l1, rates_l2, rates_linf, precision=precision):
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


mesh_sizes = np.array([40, 80, 160, 320, 640])
err_l1 = np.zeros(n := len(mesh_sizes))
err_l2 = np.zeros(n)
err_linf = np.zeros(n)
numerical_solutions = []

# implement finite volume method
fluxes = [roe_flux, lax_friedrichs_flux, rusanov, enquist_osher_flux]
nr_figures = len(fluxes)
n_best = 4  # plot n_best against exact solution for each flux

fig, axs = plt.subplots(nr_figures, 3, figsize=(7 * 3, 7 * nr_figures))
for current_flux_nr, flux in enumerate(fluxes):
    print(f"Flux: {flux.__name__}")
    numerical_solutions.append([])
    ax_plot = axs[current_flux_nr, 0]
    ax_n_best_numerical = axs[current_flux_nr, 1]
    ax_rates = axs[current_flux_nr, 2]

    ax_plot.set_title(f"Flux: {flux.__name__}")
    ax_n_best_numerical.set_title(f"Flux: {flux.__name__}")
    ax_rates.set_title(f"Flux: {flux.__name__} rates")

    for i, N in enumerate(mesh_sizes):
        dx = 2 / N
        x = np.linspace(0, 2, N)
        # Initial values:
        u = initial_values(x)
        # choosing dt according to CFL condition
        dt = get_dt_by_cfl(u, dx)
        for _ in range(int(tend / dt)):
            F_j_minus = flux(u, dx, dt)
            # print(F_j_minus)
            F_j_diff = F_j_minus[1:] - F_j_minus[:-1]
            u = u - dt / dx * F_j_diff
        numerical_solutions[current_flux_nr].append(u)
        err_l1[i] = np.sum(np.abs(u - u_exact(x))) * dx
        err_l2[i] = np.sqrt(np.sum((np.abs(u - u_exact(x))) ** 2) * dx)
        err_linf[i] = np.max(np.abs(u - u_exact(x)))

    # Plotting:
    for i, N in enumerate(mesh_sizes):
        ax_plot.scatter(np.linspace(0, 2, N), numerical_solutions[current_flux_nr][i], label=f"{N} mesh points", s=1.5)

    n_best_numerical = numerical_solutions[current_flux_nr][-n_best]
    ax_n_best_numerical.plot(np.linspace(0, 2, mesh_sizes[-n_best]), n_best_numerical,
                             label=f"{mesh_sizes[-n_best]} mesh points")
    # plot exact
    ax_n_best_numerical.plot(x := np.linspace(0, 2, mesh_sizes[-n_best]), u_exact(x), label="exact solution")
    ax_n_best_numerical.legend()

    ax_plot.set_xlabel("x")
    ax_plot.set_ylabel("u(x)")
    ax_plot.plot(x := np.linspace(0, 2, mesh_sizes[-1]), u_exact(x), label="exact solution")
    ax_plot.legend()
    mesh_widths = 1 / mesh_sizes
    ax_rates.loglog(mesh_widths, err_l1, label=r"$L^{1}$-Error")
    ax_rates.loglog(mesh_widths, err_l2, label=r"$L^{2}$-Error")
    ax_rates.loglog(mesh_widths, err_linf, label=r"$L^{\infty}$-Error")
    ax_rates.set_xlabel("mesh width h")
    ax_rates.set_ylabel("error")
    ax_rates.legend()
    ax_rates.loglog(mesh_widths, 10 * mesh_widths, label=r"$h^{1}$ (for comparison)")
    ax_rates.loglog(mesh_widths, 10 * mesh_widths ** 0.5, label=r"$h^{0.5}$ (for comparison)")
    ax_rates.set_xlabel("mesh width h")
    ax_rates.set_ylabel("error")
    ax_rates.legend()

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

    latex_out(mesh_sizes, err_l1, err_l2, err_linf, rates_l1, rates_l2, rates_linf)

fig.tight_layout()
plt.show()
