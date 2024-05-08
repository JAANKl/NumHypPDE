import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate


def init(dx, x):
    u0_ = np.zeros(len(x))
    for j in range(len(x)):
        u0_[j] = 1 / dx * integrate.quad(initial_values, x[j] - 0.5 * dx, x[j] + 0.5 * dx)[0]
    return u0_


def initial_values(x):
    return np.sin(4*np.pi*x)


def f(x):
    return x


def f_prime(x):
    return np.ones_like(x)


def u_exact(x):
    t = tend
    return initial_values(x - t)


# takes in all values of u = (u_j^n)_j at time n and returns vector of fluxes (F_{j+1/2})_j

def general_minmod(a):
    # if not all entries have the same sign, return 0
    minmod_condition = np.logical_or(np.all(a>0, axis=1), np.all(a<0, axis=1))
    return np.min(np.abs(a), axis = 1)*(minmod_condition)

def rusanov_flux(u_left, u_right, dx, dt):
    return (f(u_left) + f(u_right)) / 2 - np.max([np.abs(f_prime(u_left)), np.abs(f_prime(u_right))]) / 2 * (
            u_right - u_left)


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


def sigma_van_leer(u, dx):
    du = np.diff(u)
    # boundary
    divbyzero = (du[:-1]==0)
    du[:-1][divbyzero] = 1
    r = (1-divbyzero)*du[1:]/du[:-1]
    sigma_inside = (r+np.abs(r))/(1+np.abs(r))
    return np.concatenate([[sigma_inside[0]], sigma_inside, [sigma_inside[-1]]])

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


def sigma(u, dx):
    du = np.diff(u)
    # boundary
    sigma_inside = vectorized_minmod(du[1:], du[:-1])/dx
    return np.concatenate([[sigma_inside[-1]], sigma_inside, [sigma_inside[0]]])

# def sigma(u, dx):
#     du = np.concatenate([np.diff(u), [0]])
#     # boundary
#     return vectorized_minmod(du / dx, np.roll(du, 1) / dx)


def u_minus(u, dx, sigma_func):
    # vector containing u_{j+1/2}^{-} for all j
    return u + sigma_func(u, dx) * dx / 2


def u_plus(u, dx, sigma_func):
    # vector containing u_{j+1/2}^{+} for all j
    return u - sigma_func(u, dx) * dx / 2


tend = 1.
mesh_sizes = np.array([100])
err_l1 = np.zeros(n := len(mesh_sizes))
err_l2 = np.zeros(n)
err_linf = np.zeros(n)
numerical_solutions = []

# number of decimals for reporting values
precision = 4
sigma_func_list = [sigma_minmod, sigma_van_leer, sigma_superbee, sigma_mc]
sigma_names = ["minmod", "van leer", "superbee", "MC"]

for sigma_func in sigma_func_list:
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
            u_minus_values = u_minus(u, dx, sigma_func)[:-1]
            u_plus_values = u_plus(u, dx, sigma_func)[1:]
            F_j = rusanov_flux(u_minus_values, u_plus_values, dx, dt)
            F_j_diff = F_j[1:] - F_j[:-1]
            u[1:-1] = u[1:-1] - dt / dx * F_j_diff

        u = u[1:-1]

        numerical_solutions.append(u)
        err_l1[i] = np.sum(np.abs(u - u_exact(x))) * dx
        err_l2[i] = np.sqrt(np.sum((np.abs(u - u_exact(x))) ** 2) * dx)
        err_linf[i] = np.max(np.abs(u - u_exact(x)))

# print only one numerical solution with exact solution

index = 0
for index, sigma_func in enumerate(sigma_func_list):
    plt.plot(np.linspace(0, 1, mesh_sizes[-1]), numerical_solutions[index], '-',
             label=f"{sigma_names[index]}", linewidth=0.5)
    plt.xlabel("x")
    plt.ylabel("u(x)")
plt.plot(x := np.linspace(0, 1, mesh_sizes[-1]), u_exact(x), label="exact solution", linewidth=0.5)
plt.legend()
plt.show()