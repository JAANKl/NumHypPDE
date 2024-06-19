import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate


def init(dx, x):
    u0_ = np.zeros(len(x))
    for j in range(len(x)):
        u0_[j] = 1 / dx * integrate.quad(initial_values, x[j] - 0.5 * dx, x[j] + 0.5 * dx)[0]
    return u0_


def initial_values(x):
    if isinstance(x, (int, float)):
        x = np.array([x])
    return ((np.array([[1., 1.]]).T) * np.array(x > 0)[None, :] + (np.array([[0, 1]]).T) * np.array(x <= 0)[None, :]).T


def u_exact(x):
    t = tend
    if isinstance(x, (int, float)):
        x = np.array([x])
    return ((np.array([[0, 1]]).T) * np.array(x < -2 * t)[None, :] + (np.array([[1 / 2, 3 / 4]]).T) * (np.array(
        -2 * t <= x) * np.array(x < 2 * t))[None, :] + (np.array([[1, 1]]).T) * np.array(x >= 2 * t)[None, :]).T


# takes in all values of u = (u_j^n)_j at time n and returns vector of fluxes (F_{j+1/2})_j

# def rusanov_flux(u_left, u_right, dx, dt):
#     return (f(u_left) + f(u_right)) / 2 - np.max([np.abs(f_prime(u_left)), np.abs(f_prime(u_right))]) / 2 * (
#             u_right - u_left)

def godunov_flux(u_left, u_right):
    # u_left = np.concatenate(([u[-1]], u))
    # u_right =np.concatenate((u, [u[0]]))
    # is_smaller = (u_left <= u_right)
    # return is_smaller*f(u_left)+(1-is_smaller)*f(u_right)
    # return (1/2*np.array([[2, 4], [1, 2]])@u_left.T + 1/2*np.array([[-2, 4], [1, -2]])@u_right.T).T
    return (1 / 2 * np.array([[0., 4.], [1, 0]]) @ (u_left + u_right).T - (u_right - u_left).T).T


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
    result = np.where(mask, 0, np.where(np.abs(a) > np.abs(b), b, a))
    return result


def sigma_minmod(u, dx):
    du = np.diff(u, axis=0)
    # boundary
    sigma_inside = vectorized_minmod(du[1:], du[:-1]) / dx
    return np.concatenate([[sigma_inside[0]], sigma_inside, [sigma_inside[-1]]])


def sigma_superbee(u, dx):
    du = np.diff(u, axis=0)
    # boundary
    sigma_l_inside = vectorized_minmod(2 * du[:-1], du[1:]) / dx
    sigma_r_inside = vectorized_minmod(du[:-1], 2 * du[1:]) / dx
    sigma_inside = vectorized_maxmod(sigma_l_inside, sigma_r_inside)

    return np.concatenate([[sigma_inside[0]], sigma_inside, [sigma_inside[-1]]])


def superbee_slope(u, dx):
    sigmal = vectorized_minmod(2 * (u[1:-1] - u[0:-2]) / dx, (u[2:] - u[1:-1]) / dx)
    sigmar = vectorized_minmod((u[1:-1] - u[0:-2]) / dx, 2 * (u[2:] - u[1:-1]) / dx)

    sigma = np.zeros(len(sigmal))
    for i in range(len(sigmal)):
        if sigmal[i] > 0.0 and sigmar[i] > 0.0:
            sigma[i] = max(sigmal[i], sigmar[i])
        elif sigmal[i] < 0.0 and sigmar[i] < 0.0:
            sigma[i] = -max(abs(sigmal[i]), abs(sigmar[i]))
        else:
            sigma[i] = 0.0

    um = np.zeros(len(sigma) - 1)
    up = np.zeros(len(sigma) - 1)
    um = u[1:-2] + sigma[0:-1] * dx * 0.5
    up = u[2:-1] + sigma[1:] * (-dx) * 0.5
    return um, up


def general_minmod(a):
    # if not all entries have the same sign, return 0
    minmod_condition = np.logical_or(np.all(a > 0, axis=1), np.all(a < 0, axis=1))
    return np.min(np.abs(a), axis=1) * (minmod_condition)


def sigma_mc(u, dx):
    du = np.diff(u, axis=0)
    right_diff = du[1:]
    left_diff = du[:-1]
    central_diff = (right_diff + left_diff) / 2
    # boundary
    data = np.concatenate((right_diff[:, None], central_diff[:, None], left_diff[:, None]), axis=1) / dx
    sigma_inside = general_minmod(data)
    return np.concatenate([[sigma_inside[0]], sigma_inside, [sigma_inside[-1]]])


def no_sigma(u, dx):
    return 0


def sigma_van_leer(u, dx):
    du = np.diff(u, axis=0)
    # boundary
    divbyzero = (du[:-1] == 0)
    du[:-1][divbyzero] = 1
    r = (1 - divbyzero) * du[1:] / du[:-1]
    sigma_inside = (r + np.abs(r)) / (1 + np.abs(r))
    return np.concatenate([[sigma_inside[0]], sigma_inside, [sigma_inside[-1]]])


def u_minus(u, dx, sigma_func):
    # vector containing u_{j+1/2}^{-} for all j
    return u + sigma_func(u, dx) * dx / 2


def u_plus(u, dx, sigma_func):
    # vector containing u_{j+1/2}^{+} for all j
    return u - sigma_func(u, dx) * dx / 2


tend = 0.3
mesh_sizes = np.array([100])
err_l1 = np.zeros(n := len(mesh_sizes))
err_l2 = np.zeros(n)
err_linf = np.zeros(n)
numerical_solutions = []

# number of decimals for reporting values
precision = 4
sigma_func_list = [no_sigma, sigma_minmod, sigma_superbee, sigma_mc]
sigma_names = ["Godunov (no limiter)", "minmod", "superbee", "MC"]

for sigma_func in sigma_func_list:
    for i, N in enumerate(mesh_sizes):
        dx = 4 / N
        # choosing dt according to CFL condition
        dt = 4 / (4 * N)  # <= 1/(2N)

        x = np.linspace(-2, 2, N)
        # Initial values:
        # u = initial_values_average(x, dx)
        u = initial_values(x)
        # u = init(dx, x)
        u = np.concatenate([[u[0]], [u[0]], u, [u[-1]], [u[-1]]])
        for _ in range(int(tend / dt)):
            u[0] = u[1]  # Apply Neumann zero boundary conditions
            u[-1] = u[-2]
            u_minus_values = u_minus(u, dx, sigma_func)[:-1]
            u_plus_values = u_plus(u, dx, sigma_func)[1:]
            # u_left = u[:-1]
            # u_right = u[1:]
            F_j = godunov_flux(u_minus_values, u_plus_values)
            F_j_diff = F_j[1:] - F_j[:-1]
            u[1:-1] = u[1:-1] - dt / dx * F_j_diff

        u = u[2:-2]

        numerical_solutions.append(u)
        err_l1[i] = np.sum(np.abs(u - u_exact(x))) * dx
        err_l2[i] = np.sqrt(np.sum((np.abs(u - u_exact(x))) ** 2) * dx)
        err_linf[i] = np.max(np.abs(u - u_exact(x)))


#plot numerical together with exact solution and save figure as png, no sigma_func
linewidth = 0.8
fig, ax = plt.subplots()
for index, mesh_size in enumerate(mesh_sizes):
    ax.plot(np.linspace(-2, 2, mesh_size), numerical_solutions[index][:, 0], '-',
             label=r"$u_1$ Godunov (no limiter)", linewidth=linewidth)
    ax.plot(np.linspace(-2, 2, mesh_size), numerical_solutions[index][:, 1], '-',
             label=r"$u_2$ Godunov (no limiter)", linewidth=linewidth)
    ax.set_xlabel("x")
    ax.set_ylabel("u(x)")

ax.plot(x := np.linspace(-2, 2, mesh_sizes[-1]), u_exact(x)[:, 0], label=r"$u_1$ exact solution", linewidth=linewidth)
ax.plot(x := np.linspace(-2, 2, mesh_sizes[-1]), u_exact(x)[:, 1], label=r"$u_2$ exact solution", linewidth=linewidth)

ax.legend()
plt.savefig("Ex10_1_a.png")
plt.show()


index = 0

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
for index, sigma_func in enumerate(sigma_func_list):
    ax1.plot(np.linspace(-2, 2, mesh_sizes[-1]), numerical_solutions[index][:, 0], '-',
             label=f"{sigma_names[index]}", linewidth=linewidth)
    ax1.set_xlabel("x")
    ax1.set_ylabel(r"$u_1(x)$")
    ax2.plot(np.linspace(-2, 2, mesh_sizes[-1]), numerical_solutions[index][:, 1], '-',
             label=f"{sigma_names[index]}", linewidth=linewidth)
    ax2.set_xlabel("x")
    ax2.set_ylabel(r"$u_2(x)$")
ax1.plot(x := np.linspace(-2, 2, mesh_sizes[-1]), u_exact(x)[:, 0], label="exact solution", linewidth=linewidth)
ax2.plot(x := np.linspace(-2, 2, mesh_sizes[-1]), u_exact(x)[:, 1], label="exact solution", linewidth=linewidth)
ax1.legend()

plt.savefig("Ex10_1_b.png")
plt.show()

#save numerical solution as txt for each limiter
for index, sigma_func in enumerate(sigma_func_list):
    #header: x, u1, u2
    numerical_solutions[index] = np.concatenate([x[:, None], numerical_solutions[index]], axis=1)
    np.savetxt(f"Ex10_1_{sigma_names[index]}.txt", numerical_solutions[index], header="x u1 u2", comments="")
