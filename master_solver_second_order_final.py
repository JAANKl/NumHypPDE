import numpy as np
import math, sys
import matplotlib.pyplot as plt
import scipy.integrate as integrate

####################################################################
exercise_name = "Ex_2.3a"
save_plots = False
compute_rates = True

tend = 1
xleft = 0
xright = 1
## setup
cfl = 0.4

# which_bc = "neumann"
which_bc = "periodic"

which_schemes =  ["godunov"]#["lax_friedrichs", "rusanov", "godunov", "enquist_osher", "roe", "beam_warming"]
# lax_friedrichs, rusanov, godunov, enquist_osher, roe, beam_warming
which_limiters = ["zero", "vanleer"]#["zero", "minmod", "superbee", "mc", "vanleer"]

# time_integration_method = "euler"
time_integration_method = "rk2"

mesh_sizes = np.array([40, 80, 160, 320, 640]) #np.array([100]) 
mesh_index_to_plot = 1

#only for Riemann problem
u_L = -1
u_R = 1

#only for linear advection
a = 2

iordert = 2
icase = 3
ischeme = which_schemes[0]
islope = which_limiters[1]

####################################################################

def init(dx, x):
    u0_ = np.zeros(len(x)-1)
    for j in range(len(x)-1):
        u0_[j] = 1 / dx * integrate.quad(initial_values, x[j], x[j+1])[0]  # Midpoint rule
    return u0_




def initial_values(x):
    # return 2 * (x <= 0.5) + 1 * (x > 0.5)
    return np.sin(2 * np.pi * x)
    # Bugers' equation
    # return np.array(np.where(x < 0, u_L, u_R), dtype=float)
    # return np.sin(np.pi*x) + 0.5

def flux(u):
    # Linear advection:
    return a*u
    # Burgers' equation:
    # return u ** 2 / 2


def fluxp(u):
    # Linear advection:
    return a*np.ones_like(u)
    # Burgers' equation:
    # return u


def u_exact(t, x):
    # Linear advection:
    return initial_values(x - a*t)
    # Burgers' equation shock: (u_L > u_R)
    # s = (flux(u_L) - flux(u_R)) / (u_L - u_R)
    # return np.where((x < s*t), u_L, u_R)
    # Burgers' equation rarefaction: (u_L < u_R)
    # u = np.zeros(len(x))
    # for i in range(len(x)):
    #     if x[i] <= fluxp(u_L) * t:
    #         u[i] = u_L
    #     elif x[i] <= fluxp(u_R) * t:
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


def getflux(ischeme, islope, dt, dx, flux, fluxp, u):
    if ischeme == "lax_friedrichs":
        fh = lax_friedrichs(islope, dx, u)
    elif ischeme == "rusanov":
        fh = rusanov(islope, dx, u)
    elif ischeme == "godunov":
        fh = godunov(islope, dx, u)
    elif ischeme == "enquist_osher":
        fh = enquist_osher(islope, dx, u)
    elif ischeme == "roe":
        fh = roe(islope, dx, u)
    else:
        sys.exit("ERROR: this is not defined here.")
    rhs = -(fh[1:] - fh[0:-1]) / dx
    return rhs

def getslope(dx, u, islope):
    if islope == "zero":
        return zero_slope(dx, u)
    elif islope == "minmod":
        return minmod_slope(dx, u)
    elif islope == "superbee":
        return superbee_slope(dx, u)
    elif islope == "mc":
        return mc_slope(dx, u)
    elif islope == "vanleer":
        return vanleer_slope(dx, u)
    else:
        sys.exit("ERROR: this is not defined here.")


def lax_friedrichs(islope, dx, u):
    um, up = getslope(dx, u, islope)

    ump = np.hstack((np.abs(fluxp(um)), np.abs(fluxp(up))))
    alp = max(ump)
    fh = 0.5 * (flux(um) + flux(up)) - 0.5 * alp * (up - um)
    return fh


def roe(islope, dx, u):
    um, up = getslope(dx, u, islope)
    A_hat = np.zeros(len(um))
    for j in range(len(um)):
        if np.abs(um[j] - up[j]) < 1e-7:
            A_hat[j] = fluxp(um[j])
        else:
            A_hat[j] = (flux(up[j]) - flux(um[j])) / (up[j] - um[j])
    fh = np.where(A_hat >= 0, flux(um), flux(up))
    return fh


def rusanov(islope, dx, u):
    um, up = getslope(dx, u, islope)

    ## advection equation
    # alp = np.maximum(np.abs(fluxp(um)), np.abs(fluxp(up)))
    # fh = 0.5 * (flux(um) + flux(up)) - 0.5 * alp * (up - um)

    ## burgers' equation
    #    alp = np.maximum(np.absolute(um), np.absolute(up))
    #    fh = 0.5*(flux(um)+flux(up))-0.5*alp*(up-um)

    fh = (flux(um) + flux(up)) / 2 - np.max([np.abs(fluxp(um)), np.abs(fluxp(up))]) / 2 * (up - um)
    return fh

def enquist_osher(islope, dx, u):
    um, up = getslope(dx, u, islope)
    integrand = lambda theta: np.abs(fluxp(theta))
    integrals = np.zeros_like(um)
    for i in range(len(integrals)):
        integrals[i] = integrate.quad(integrand, um[i], up[i])[0]
    return (flux(um) + flux(up)) / 2 - integrals


def godunov(islope, dx, u):
    f = flux(u) * np.ones(len(u))
    fp = fluxp(u) * np.ones(len(u))
    um, up = getslope(dx, u, islope)

    fh = np.zeros(len(um))
    for i in range((len(um))):
        ## advection equation
        if um[i] <= up[i]:
            fh[i] = min(flux(um[i]), flux(up[i]))
        else:
            fh[i] = max(flux(um[i]), flux(up[i]))

        # burgers's equation
        #    a = max(um[i], 0)
        #    b = min(up[i], 0)
        #    fh[i] = max(flux(a), flux(b))

    # fh = np.zeros(len(um))
    # for i in range(len(um)):
    #     if um[i] <= up[i]:
    #         uu = np.linspace(um[i], up[i], 100)
    #         ff = flux(uu) * np.ones(len(uu))
    #         fh[i] = min(ff)
    #     else:
    #         uu = np.linspace(um[i], up[i], 100)
    #         ff = flux(uu) * np.ones(len(uu))
    #         fh[i] = max(ff)
    return fh


def minmod(a, b):
    return (np.sign(a) + np.sign(b)) / 2.0 * np.minimum(np.abs(a), np.abs(b))


def minmod2(a, b, c):
    return (np.sign(a) + np.sign(b) + np.sign(c)) / 3.0 * np.minimum(np.abs(a), np.abs(b), np.abs(c))


def zero_slope(dx, u):
    um = u[1:-2]
    up = u[2:-1]
    return um, up

def minmod_slope(dx, u):
    sigma = minmod(u[2:] - u[1:-1], u[1:-1] - u[0:-2]) / dx

    um = np.zeros(len(sigma) - 1)
    up = np.zeros(len(sigma) - 1)
    um = u[1:-2] + sigma[0:-1] * dx * 0.5
    up = u[2:-1] + sigma[1:] * (-dx) * 0.5
    return um, up


def upwind_slope(dx, u):
    sigma = (u[1:-1] - u[0:-2]) / dx

    um = np.zeros(len(sigma) - 1)
    up = np.zeros(len(sigma) - 1)
    um = u[1:-2] + sigma[0:-1] * dx * 0.5
    up = u[2:-1] + sigma[1:] * (-dx) * 0.5
    return um, up


def downwind_slope(dx, u):
    sigma = (u[2:] - u[1:-1]) / dx

    um = np.zeros(len(sigma) - 1)
    up = np.zeros(len(sigma) - 1)
    um = u[1:-2] + sigma[0:-1] * dx * 0.5
    up = u[2:-1] + sigma[1:] * (-dx) * 0.5
    return um, up


def superbee_slope(dx, u):
    sigmal = minmod(2 * (u[1:-1] - u[0:-2]) / dx, (u[2:] - u[1:-1]) / dx)
    sigmar = minmod((u[1:-1] - u[0:-2]) / dx, 2 * (u[2:] - u[1:-1]) / dx)

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


def mc_slope(dx, u):
    sigma = minmod2(2 * (u[2:] - u[1:-1]) / dx, (u[2:] - u[0:-2]) / 2 / dx, 2 * (u[1:-1] - u[0:-2]) / dx)

    um = np.zeros(len(sigma) - 1)
    up = np.zeros(len(sigma) - 1)
    um = u[1:-2] + sigma[0:-1] * dx * 0.5
    up = u[2:-1] + sigma[1:] * (-dx) * 0.5
    return um, up


def vanleer_slope(dx, u):
    #    a = u[2:]-u[1:-1]
    #    b = u[1:-1]-u[0:-2]
    #    sigma = np.zeros(len(a))
    #    for i in range(len(a)):
    #        r = a[i]/(b[i]+1e-10)
    #        sigma[i] = (r+abs(r))/(1+abs(r))

    r = (u[2:] - u[1:-1]) / (u[1:-1] - u[0:-2] + 1e-10)
    sigma = (r + np.abs(r)) / (1 + np.abs(r))

    um = np.zeros(len(sigma) - 1)
    up = np.zeros(len(sigma) - 1)
    um = u[1:-2] + sigma[0:-1] * dx * 0.5
    up = u[2:-1] + sigma[1:] * (-dx) * 0.5
    return um, up


def euler_forward(dt, u, rhs):
    u[0, 2:-2] = u[0, 2:-2] + dt * rhs
    return u


def rk2(k, dt, u, rhs):
    if k == 0:
        u[1, 2:-2] = u[0, 2:-2] + dt * rhs
    elif k == 1:
        u[0, 2:-2] = 0.5 * u[0, 2:-2] + 0.5 * (u[1, 2:-2] + dt * rhs)
    return u


def L1err(dx, uc, uex):
    u_diff = abs(uc[0:-1] - uex[0:-1])
    err = sum(u_diff) * dx
    return err


def L2err(dx, uc, uex):
    u_diff = abs(uc[0:-1] - uex[0:-1])
    err = sum(np.square(u_diff)) * dx
    err = math.sqrt(err)
    return err


def L8err(uc, uex):
    u_diff = abs(uc - uex)
    err = np.max(u_diff)
    return err


def ErrorOrder(err, NN):
    ord = np.log(err[0:-1] / err[1:]) / np.log(NN[1:] / NN[0:-1])
    return ord



L1_ErrorStore = np.zeros(len(mesh_sizes))
L2_ErrorStore = np.zeros(len(mesh_sizes))
L8_ErrorStore = np.zeros(len(mesh_sizes))

L1_OrderStore = np.zeros(len(mesh_sizes) - 1)
L2_OrderStore = np.zeros(len(mesh_sizes) - 1)
L8_OrderStore = np.zeros(len(mesh_sizes) - 1)
numerical_solutions = []
for index in range(len(mesh_sizes)):
    N = mesh_sizes[index]
    N_ = N + 1

    ## uniform grid
    dx = (xright - xleft) / N
    xh = np.linspace(xleft - 0.5 * dx, xright + 0.5 * dx, N_ + 1)
    xi = np.linspace(xleft, xright, N_)
    dt = cfl * dx

    ## initialize
    uc = np.zeros((2, N_ + 4))  # 2 ghost points for the boundary on the left and 2 ghost points on the right.
    uc[0, 2:-2] = init(dx, xh)

    ## exact solution
    uex = np.zeros(N_)
    uex = u_exact(tend, xi)

    ## time evolution
    time = 0
    kt = 0
    kmax = 10 ** 5
    # while (time < tend)&(kt<kmax):
    while time < tend:

        if time + dt >= tend:
            dt = tend - time
        time = time + dt
        kt = kt + 1

        if iordert == 1:
            ## call bc
            apply_bc(uc[0, :], which_bc)

            ## spatial discretization
            if icase == 1:
                rhs = getflux(ischeme, islope, dt, dx, flux, fluxp, uc[0, 1:-1])
            elif icase == 2:
                rhs = getflux(ischeme, islope, dt, dx, flux, fluxp, uc[0, 0:-2])
            elif icase == 3:
                rhs = getflux(ischeme, islope, dt, dx, flux, fluxp, uc[0, :])
            else:
                sys.exit("ERROR: this is not defined here.")
            euler_forward(dt, uc, rhs)
        elif iordert == 2:
            for k in range(iordert):
                ## call bc
                apply_bc(uc[k, :], which_bc)

                ## spatial discretization
                if icase == 1:
                    rhs = getflux(ischeme, islope, dt, dx, flux, fluxp, uc[k, 1:-1])
                elif icase == 2:
                    rhs = getflux(ischeme, islope, dt, dx, flux, fluxp, uc[k, 0:-2])
                elif icase == 3:
                    rhs = getflux(ischeme, islope, dt, dx, flux, fluxp, uc[k, :])
                else:
                    sys.exit("ERROR: this is not defined here.")
                rk2(k, dt, uc, rhs)
    apply_bc(uc[0, :], which_bc)
    uc_ = uc[0, 2:-2]
    numerical_solutions.append(uc_)

    L1_ErrorStore[index] = L1err(dx, uc_, uex)
    L2_ErrorStore[index] = L2err(dx, uc_, uex)
    L8_ErrorStore[index] = L8err(uc_, uex)

L1_OrderStore = ErrorOrder(L1_ErrorStore, mesh_sizes)
L2_OrderStore = ErrorOrder(L1_ErrorStore, mesh_sizes)
L8_OrderStore = ErrorOrder(L1_ErrorStore, mesh_sizes)


plt.plot(np.linspace(xleft, xright, mesh_sizes[-1] + 1), numerical_solutions[-1], '-',
            label=f"{ischeme}", linewidth=1)
plt.xlabel("x")
plt.ylabel("u(x)")
plt.plot(x := np.linspace(xleft, xright, mesh_sizes[-1] + 1), u_exact(tend, xi), label="exact solution", linewidth=1,
            color="black")
plt.title(f"{mesh_sizes[-1] + 1} Points")
plt.legend()
plt.show()

for i, N in enumerate(mesh_sizes):
    plt.scatter(np.linspace(xleft, xright, N+1), numerical_solutions[i], label=f"{N} mesh points", s=1)
plt.show()

mesh_widths = 1 / mesh_sizes
plt.loglog(mesh_widths, L1_ErrorStore, label="$L^{1}$-Error")
plt.loglog(mesh_widths, L2_ErrorStore, label="$L^{2}$-Error")
plt.loglog(mesh_widths, L8_ErrorStore, label="$L^{\infty}$-Error")
plt.loglog(mesh_widths, 10 * mesh_widths, label="$h^{1}$ (for comparison)", linestyle='dashed')
plt.loglog(mesh_widths, 100 * mesh_widths ** 2, label="$h^{2}$ (for comparison)", linestyle='dashed')
plt.xlabel("mesh width h")
plt.ylabel("error")
plt.legend()
plt.show()







## print out the errors
# with open('error_%s_%s.txt'%(ischeme, islope), 'w') as file:
#     file.write("%6d %s %.2e %s %4s %s %.2e %s %4s %s %.2e %s %4s %4s\n"%
#     (NN[0], '&', L1_ErrorStore[0], '&', '-', '&', L2_ErrorStore[0], '&', '-', '&', L8_ErrorStore[0], '&', '-', '\\\\'))
#     for index in range(1, len(NN)):
#         file.write("%6d %s %.2e %s %.2f %s %.2e %s %.2f %s %.2e %s %.2f %4s\n"%
#     (NN[index], '&', L1_ErrorStore[index], '&', L1_OrderStore[index-1], '&', L2_ErrorStore[index], '&', L2_OrderStore[index-1], '&', L8_ErrorStore[index], '&', L8_OrderStore[index-1],'\\\\'))
# file.close()

## plot numerical solution
#    with open ('num_sol_{}.txt'.format(ischeme), 'w', encoding='utf-8') as file:
#        for index in range(len(xi)):
#            file.write("%f %f %f \n"%(xi[index], uex[index], uc_[index]))
#
#    plt.figure()
#    plt.plot(xi, uc_, 'r--', markerfacecolor="None", markersize=1, label= '{}'.format(islope))
#    plt.plot(xi, uex, 'k-', markerfacecolor="None", markersize=1, label='exact')
#    plt.legend()
#    plt.xlabel('x');plt.ylabel('u')
#    plt.show()
