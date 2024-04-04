import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sc


def initial_values(x):
    return np.sin(np.pi * x) + 0.5


def f(x):
    return (x ** 2) / 2

def g(x_0, t, x):
    return x_0 + (np.sin(np.pi * x_0) + 1 / 2) * t - x


def g_prime(x_0, t, x):
    return 1 + np.pi * np.cos(np.pi * x_0) * t

tend = 1.5/np.pi

def u_exact(x):
    
    t = tend
    x_s = 1+1/2*t
    
    no_prob = x<=0.2
    prob_is_left = np.logical_and(x <= x_s, x>1.0)
    prob_is_right = x > x_s
    init_val = 0.5*prob_is_left+1.5*prob_is_right+(0)*no_prob
    x_0 = sc.newton(g, x0=init_val,fprime=g_prime, args=(t, x),tol=1e-5, maxiter=100)
    return initial_values(x_0)


mesh_sizes = np.array([20, 100, 200, 400, 800])
err_l1 = np.zeros(n := len(mesh_sizes))
err_l2 = np.zeros(n)
err_linf = np.zeros(n)
numerical_solutions = []

#number of decimals for reporting values
precision = 4

def f(u):
    return u**2/2

def godunov_flux_local(u_left, u_right):
    return f(u_left)
    #if u_left <= u_right:
    #    return f(u_left)
    #else:
    #    return f(u_right)

#takes in all values of u = (u_j^n)_j at time n and returns vector of fluxes (F_{j+1/2})_j

def godunov_flux(u):
    #periodic boundary
    u_left = np.concatenate(([u[-1]], u))
    u_right =np.concatenate((u, [u[0]]))
    
    return np.maximum(f(np.maximum(u_left, 0)), f(np.minimum(u_right, 0)))

    """ implementation case by case, also works:
    is_smaller = (u_left <= u_right)
    is_bigger = (1-is_smaller)
    
    case_1 = is_smaller*(0 <= u_left)
    cases_before = case_1
    case_2 = is_smaller*(u_right <= 0)*(1-cases_before)
    cases_before = np.logical_or(case_2, cases_before)
    case_3 = (u_left <= 0)*(u_right >= 0)*(1-cases_before)
    cases_before = np.logical_or(case_3, cases_before)
    case_4 = (is_bigger)*(0 <= u_right)*(1-cases_before)
    cases_before = np.logical_or(case_4, cases_before)
    case_5 = (is_bigger)*(u_left <= 0)*(1-cases_before)
    cases_before = np.logical_or(case_5, cases_before)
    case_6 = 1-cases_before
    
    max_abs = np.maximum(np.abs(u_left), np.abs(u_right))
    min_abs = np.minimum(np.abs(u_left), np.abs(u_right))
    
    return (case_1+case_2)*f(min_abs)+(case_4+case_5)*f(max_abs)+case_6*f(max_abs)
    
    """
for i, N in enumerate(mesh_sizes):
    dx = 2 / N
    #choosing dt according to CFL condition
    dt = 2 / (4 * N)  # <= 1/(2N)

    x = np.linspace(0, 2, N)
    # Initial values:
    u = initial_values(x)
    for _ in range(int(tend / dt)):
        F_j_minus = godunov_flux(u)
        #print(F_j_minus)
        F_j_diff = F_j_minus[1:]-F_j_minus[:-1]
        u = u - dt/dx*F_j_diff
    numerical_solutions.append(u)
    err_l1[i] = np.sum(np.abs(u - u_exact(x))) * dx
    err_l2[i] = np.sqrt(np.sum((np.abs(u - u_exact(x))) ** 2) * dx)
    err_linf[i] = np.max(np.abs(u - u_exact(x)))

# Plotting:
for i, N in enumerate(mesh_sizes):
    plt.scatter(np.linspace(0, 2, N), numerical_solutions[i], label=f"{N} mesh points", s=1)

plt.xlabel("x")
plt.ylabel("u(x)")
plt.plot(x := np.linspace(0, 2, mesh_sizes[-1]), u_exact(x), label="exact solution")
plt.legend()
plt.show()
mesh_widths = 1 / mesh_sizes
plt.loglog(mesh_widths, err_l1, label="$L^{1}$-Error")
plt.loglog(mesh_widths, err_l2, label="$L^{2}$-Error")
plt.loglog(mesh_widths, err_linf, label="$L^{\infty}$-Error")
plt.loglog(mesh_widths, 10 * mesh_widths, label="$h^{1}$ (for comparison)")
plt.loglog(mesh_widths, 10 * mesh_widths ** 0.5, label="$h^{0.5}$ (for comparison)")
plt.xlabel("mesh width h")
plt.ylabel("error")
plt.legend()
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

#print only one numerical solution with exact solution

index = 2
plt.plot(np.linspace(0, 2, mesh_sizes[index]), numerical_solutions[index], '-', label=f"{mesh_sizes[index]} mesh points")
plt.plot(x := np.linspace(0, 2, mesh_sizes[-1]), u_exact(x), label="exact solution")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.legend()