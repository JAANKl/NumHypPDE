import numpy as np
import matplotlib.pyplot as plt


# Define the limiter functions
def minmod(a, b):
    if a * b <= 0:
        return 0
    elif abs(a) < abs(b):
        return a
    else:
        return b


def superbee(a, b):
    if a * b <= 0:
        return 0
    else:
        return max(min(2 * abs(a), b), min(a, 2 * abs(b)))


def van_leer(a, b):
    if a * b <= 0:
        return 0
    else:
        return 2 * a * b / (a + b)


def mc_limiter(a, b):
    return max(0, min(min(2 * a, (a + b) / 2), 2 * b))


# Define the Godunov flux calculation using limiters
def godunov_flux(u, dx, limiter_func):
    du = np.diff(u)
    dflux = np.zeros_like(u)
    for i in range(1, len(u) - 1):
        dflux[i] = limiter_func(du[i - 1], du[i])

    f = u  # Flux function f(u) = u for linear advection
    flux = f[:-1] + 0.5 * dflux[:-1] * (dx - np.abs(f[:-1]))
    return flux


# Define the solver for advection with periodic boundary conditions
def solve_advection_periodic(N, tend, cfl, limiter_func):
    x = np.linspace(0, 1, N + 2)  # Includes ghost cells
    dx = x[1] - x[0]
    dt = cfl * dx
    t = 0
    u = np.sin(4 * np.pi * x)  # Smooth initial condition
    while t < tend:
        u[0] = u[-2]  # Apply periodic boundary conditions
        u[-1] = u[1]
        flux = godunov_flux(u, dx, limiter_func)
        u[1:-1] -= dt / dx * (flux[1:] - flux[:-1])
        t += dt
    return x[1:-1], u[1:-1]  # Exclude ghost cells


# Set parameters for the simulation
N = 100
tend = 1.0
cfl = 0.9

# Solve the advection equation using different limiters
x, u_minmod = solve_advection_periodic(N, tend, cfl, minmod)
x, u_superbee = solve_advection_periodic(N, tend, cfl, superbee)
x, u_van_leer = solve_advection_periodic(N, tend, cfl, van_leer)
x, u_mc = solve_advection_periodic(N, tend, cfl, mc_limiter)

# Plotting results
plt.figure(figsize=(12, 8))
plt.plot(x, u_minmod, label='Minmod')
plt.plot(x, u_superbee, label='Superbee')
plt.plot(x, u_van_leer, label='Van Leer')
plt.plot(x, u_mc, label='MC Limiter')
plt.legend()
plt.title('Numerical solutions with different limiters at t=1 for Smooth Initial Data')
plt.xlabel('x')
plt.ylabel('u')
plt.grid(True)
plt.show()
