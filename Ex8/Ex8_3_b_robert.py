import numpy as np
import matplotlib.pyplot as plt

tend = 1


def initial_values(x):
    return np.sin(4*np.pi*x)



def average(x, func, dx):
    # 1/dx*(dx/2*func(x-dx)+dx/2*func(x))
    # midpoint rule
    # (func(x-dx/2)+func(x+dx/2))/2
    # left point rule
    return 1 / dx * (dx / 2 * func(x - dx) + dx / 2 * func(x))


def f(x):
    return x


def u_exact(x):
    t = tend
    return initial_values(x - t)


mesh_sizes = np.array([100])
err_l1 = np.zeros(n := len(mesh_sizes))
err_l2 = np.zeros(n)
err_linf = np.zeros(n)
numerical_solutions = []

# number of decimals for reporting values
precision = 4

def apply_dirichlet_bc(u):
    u[0] = u[1]
    u[-1] = u[-2]
    return u

def reconstruction(u_avg, dx, limiter_sigma):
    #neumann boundaries
    u_left = np.concatenate(([u_avg[0]], u_avg))
    u_right = np.concatenate((u_avg, [u_avg[-1]]))

    #periodic boundaries
    # u_left = np.concatenate(([u_avg[-1]], u_avg))
    # u_right = np.concatenate((u_avg, [u_avg[0]])) 
    right_diff = u_right[1:] - u_right[:-1]
    left_diff = u_left[1:] - u_left[:-1]
    sigma_j = limiter_sigma(left_diff,right_diff, dx)
    u_j_plus_half_minus = u_avg + sigma_j * dx / 2
    u_j_plus_half_plus = u_right[1:] - sigma_j * dx / 2
    return u_j_plus_half_minus, u_j_plus_half_plus

def godunov_flux(u_left, u_right):
    #u_left = np.concatenate(([u[-1]], u))
    # u_right =np.concatenate((u, [u[0]]))
    is_smaller = (u_left <= u_right)
    return is_smaller*f(u_left)+(1-is_smaller)*f(u_right)
    #return f(u_left)

def minmod(a):
    # if not all entries have the same sign, return 0
    minmod_condition = np.logical_or(np.all(a>0, axis=1), np.all(a<0, axis=1))
    return np.min(np.abs(a), axis = 1)*(minmod_condition)


def sigma_minmod(right_diff, left_diff, dx):
    data = np.concatenate((right_diff[:, None], left_diff[:, None]),axis=1)/dx
    return minmod(data)

def sigma_van_leer(right_diff, left_diff, dx):
    divbyzero = (left_diff==0)
    left_diff[divbyzero] = 1
    r = (1-divbyzero)*right_diff/left_diff
    return (r+np.abs(r))/(1+np.abs(r))


def sigma_j(u_j_minus, u_j, u_j_plus, dx):
    return minmod(np.array([(u_j_plus - u_j) / dx, (u_j - u_j_minus) / dx]))


def sigma(u, dx):
    # boundary
    u_left = np.concatenate(([u[0]], u[:-1]))
    u_middle = u
    u_right = np.concatenate((u[1:], [u[-1]]))

    return np.array(
        [sigma_j(u_j_minus, u_j, u_j_plus, dx) for u_j_minus, u_j, u_j_plus in zip(u_left, u_middle, u_right)])


def u_plushalf_minus(u, dx):
    # vector containing u_{j+1/2}^{-} for all j
    # boundary
    u_left = u[:-1]
    return u_left + sigma(u_left, dx) * dx / 2


def u_plushalf_plus(u, dx):
    # vector containing u_{j+1/2}^{+} for all j
    # truncating by one element
    # u_right = np.concatenate((u[1:], [u[-1]]))
    u_right = u[1:]
    sigma_middle = sigma(u, dx)
    sigma_right = sigma_middle[1:]
    # sigma_right = np.concatenate((sigma_middle[1:], [sigma_middle[-1]]))

    return u_right - sigma_right * dx / 2


for i, N in enumerate(mesh_sizes):
    dx = 1 / N
    # choosing dt according to CFL condition
    dt = 1 / (20 * N)  # <= 1/(2N)

    x = np.linspace(0, 2, N)
    # Initial values:
    #u = initial_values_average(x, dx)
    u=initial_values_average(x, dx)
    for _ in range(int(tend / dt)):
        u_j_plus_half_minus, u_j_plus_half_plus = reconstruction(u, dx, sigma_van_leer)
        F_j_plus_half = godunov_flux(u_j_plus_half_minus, u_j_plus_half_minus)
        F_j_diff = F_j_plus_half[1:-1] - F_j_plus_half[:-2]
        u[1:-1] = u[1:-1] - dt / dx * F_j_diff
        u = apply_dirichlet_bc(u)

    numerical_solutions.append(u)
    err_l1[i] = np.sum(np.abs(u - u_exact(x))) * dx
    err_l2[i] = np.sqrt(np.sum((np.abs(u - u_exact(x))) ** 2) * dx)
    err_linf[i] = np.max(np.abs(u - u_exact(x)))

# print only one numerical solution with exact solution

index = 0
plt.plot(np.linspace(0, 2, mesh_sizes[index]), numerical_solutions[index], '-',
         label=f"{mesh_sizes[index]} mesh points")
plt.plot(x := np.linspace(0, 2, mesh_sizes[-1]), u_exact(x), label="exact solution")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.legend()
plt.show()
