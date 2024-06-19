import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate


def f(u):
    return 0.5*u
    # return u**2/2


def f_prime(u):
    return 0.5*np.ones_like(u)
    # return u

def init(dx, x):
    # u0_ = np.zeros(len(x))
    # for j in range(len(x)):
    #     u0_[j] = 1 / dx * integrate.quad(initial_values, x[j] - 0.5 * dx, x[j] + 0.5 * dx)[0]
    # return u0_
    return u_exact(0, x)


def initial_values(x):
    return u_exact(0, x)
    # return 2 * (x <= 0.5) + 1 * (x > 0.5)


# def u_exact(x):
#     t = tend
#     return initial_values(x - t)

def u_exact(t, x):
    u_L = 1
    u_R = 0
    # Rarefaction:
    # u = np.zeros(len(x))
    # for i in range(len(x)):
    #     if x[i] <= f_prime(u_L)*t:
    #         u[i] = u_L
    #     elif x[i] <= f_prime(u_R)*t:
    #         u[i] = x[i]/t
    #     else:
    #         u[i] = u_R
    # return u
    # Shock:
    # s = (f(u_L) - f(u_R))/(u_L - u_R)
    # return np.where((x < s*t), u_L, u_R)
    # Linear Transport:
    # return np.where((x < 0.5*t), u_L, u_R)
    return np.sin((x-0.5*t)*np.pi)


# takes in all values of u = (u_j^n)_j at time n and returns vector of fluxes (F_{j+1/2})_j

def rusanov_flux(u_left, u_right):
    return (f(u_left) + f(u_right)) / 2 - np.max([np.abs(f_prime(u_left)), np.abs(f_prime(u_right))]) / 2 * (
            u_right - u_left)


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


tend = 1
x_left = -1
x_right = 1
mesh_sizes = np.array([100])
err_l1 = np.zeros(n := len(mesh_sizes))
err_l2 = np.zeros(n)
err_linf = np.zeros(n)
numerical_solutions = []

# number of decimals for reporting values
precision = 4

for i, N in enumerate(mesh_sizes):
    dx = (x_right - x_left) / N
    # choosing dt according to CFL condition
    dt = dx * 2  # <= 1/(2N)

    x = np.linspace(x_left, x_right, N)
    # Initial values:
    # u = initial_values_average(x, dx)
    # u = initial_values(x)
    u = init(dx, x)
    u = np.concatenate([[u[0]], u, [u[-1]]])
    for _ in range(int(tend / dt)):
        u[0] = u[1]  # Apply Neumann boundary conditions
        u[-1] = u[-2]
        # u[0] = u[-2]  # Apply Periodic boundary conditions
        # u[-1] = u[1]
        F_j = roe_flux(u[:-1], u[1:])
        F_j_diff = F_j[1:] - F_j[:-1]
        u[1:-1] = u[1:-1] - dt / dx * F_j_diff

    u = u[1:-1]

    numerical_solutions.append(u)
    err_l1[i] = np.sum(np.abs(u - u_exact(tend, x))) * dx
    err_l2[i] = np.sqrt(np.sum((np.abs(u - u_exact(tend, x))) ** 2) * dx)
    err_linf[i] = np.max(np.abs(u - u_exact(tend, x)))

# print only one numerical solution with exact solution

index = 0
plt.plot(np.linspace(x_left, x_right, mesh_sizes[-1]), numerical_solutions[index], '-',
         label=f"{mesh_sizes}", linewidth=0.5)
plt.xlabel("x")
plt.ylabel("u(x)")
plt.plot(x := np.linspace(x_left, x_right, mesh_sizes[-1]), u_exact(tend, x), label="exact solution", linewidth=1)
plt.legend()
plt.show()
