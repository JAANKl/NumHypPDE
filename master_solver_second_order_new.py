import numpy as np
import math, sys
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.optimize as sc


def init(dx, x):
    u0_ = np.zeros(len(x)-1)
    for j in range(len(x)-1):
        u0_[j] = 1 / dx * integrate.quad(initial_values, x[j], x[j+1])[0]  # Midpoint rule
    return u0_


u_L = -1
u_R = 1


def initial_values(x):
    # return 2 * (x <= 0.5) + 1 * (x > 0.5)
    # return np.sin(2 * np.pi * x)
    # Burgers' equation Riemann Problem
    # return np.where(x < 0, u_L, u_R)
    # Burgers' equation sin wave
    return np.sin(np.pi*x) + 0.5


a = 1


def flux(u):
    # Linear advection:
    # return a*u
    # Burgers' equation:
    return u ** 2 / 2


def fluxp(u):
    # Linear advection:
    # return a*np.ones_like(u)
    # Burgers' equation:
    return u


def exactu(t, x):
    # Linear advection:
    # return initial_values(x - a*t)
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
    # Burgers' equation sin wave:
    x_s = 1 + 1 / 2 * t
    def g(x_0, t, x):
        return x_0 + (np.sin(np.pi * x_0) + 1 / 2) * t - x
    def g_prime(x_0, t, x):
        return 1 + np.pi * np.cos(np.pi * x_0) * t
    prob_is_left = x <= x_s
    prob_is_right = x > x_s
    init_val = 0 * prob_is_left + 2 * prob_is_right
    x_0 = sc.newton(g, x0=init_val, fprime=g_prime, args=(t, x), tol=1e-5, maxiter=100)
    return initial_values(x_0)


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
    if ischeme == "upwind":
        fh = upwind(u)
    elif ischeme == "godunov":
        fh = godunov(u)
    elif ischeme == "lax_friedrichs":
        fh = lax_friedrichs(u)
    elif ischeme == "rusanov":
        fh = rusanov(u)
    elif ischeme == "lax_wendroff":
        fh = lax_wendroff(dt, dx, u)
    elif ischeme == "enquist_osher":
        fh = enquist_osher(dx, u)
    elif ischeme == "roe":
        fh = roe(dx, u)
    elif ischeme == "beam-warming":
        fh = beam_warming(dt, dx, u)
    elif ischeme == "lax_friedrichs_m":
        fh = lax_friedrichs_m(islope, dx, u)
    elif ischeme == "rusanov_m":
        fh = rusanov_m(islope, dx, u)
    elif ischeme == "godunov_m":
        fh = godunov_m(islope, dx, u)
    elif ischeme == "enquist_osher_m":
        fh = enquist_osher_m(islope, dx, u)
    elif ischeme == "roe_m":
        fh = roe_m(islope, dx, u)
    else:
        sys.exit("ERROR: this is not defined here.")
    rhs = -(fh[1:] - fh[0:-1]) / dx
    return rhs


def godunov(u):
    # f = flux(u) * np.ones(len(u))
    # fp = fluxp(u) * np.ones(len(u))
    fh = np.zeros(len(u) - 1)
    for i in range(len(u) - 1):
        #     ## advection equation
        #     if u[i] <= u[i+1]:
        #         fh[i] = min(f[i], f[i+1])
        #     else:
        #         fh[i] = max(f[i], f[i+1])

        ## burgers's equation
        #        a = max(u[i], 0)
        #        b = min(u[i+1], 0)
        #        fh[i] = max(flux(a), flux(b))

        ## another way for burgers' equation
        #        if u[i] <= u[i+1]:
        #            if u[i] < 0 < u[i+1]:
        #                fh[i] = 0
        #            else:
        #                fh[i] = min(f[i], f[i+1])
        #        else:
        #            fh[i] = max(f[i], f[i+1])

        ## general case for f(u)
        if u[i] <= u[i + 1]:
            uu = np.linspace(u[i], u[i + 1], 100)
            ff = flux(uu) * np.ones(len(uu))
            fh[i] = min(ff)
        else:
            uu = np.linspace(u[i + 1], u[i], 100)
            ff = flux(uu) * np.ones(len(uu))
            fh[i] = max(ff)
    return fh


def lax_friedrichs(u):
    f = flux(u) * np.ones(len(u))
    fp = fluxp(u) * np.ones(len(u))
    alp = max(np.absolute(fp))
    fh = 0.5 * (f[0:-1] + f[1:]) - 0.5 * alp * (u[1:] - u[0:-1])
    return fh


def rusanov(u):
    um, up = u[:-1], u[1:]
    fh = (flux(um) + flux(up)) / 2 - np.max([np.abs(fluxp(um)), np.abs(fluxp(up))]) / 2 * (up - um)
    return fh


def beam_warming(dt, dx, u):
    f = flux(u) * np.ones(len(u))
    fp = fluxp(u) * np.ones(len(u))
    ah = np.zeros(len(u) - 1)
    for i in range(len(u) - 1):
        if u[i] == u[i + 1]:
            ah[i] = fp[i]
        else:
            ah[i] = (f[i + 1] - f[i]) / (u[i + 1] - u[i])
    fh = f[1:] + 0.5 * ah * (1 - dt / dx * ah) * (u[1:] - u[0:-1])
    return fh


def lax_wendroff(dt, dx, u):
    f = flux(u) * np.ones(len(u))
    fp = fluxp(u) * np.ones(len(u))
    ah = np.zeros(len(u) - 1)
    for i in range(len(u) - 1):
        if u[i] == u[i + 1]:
            ah[i] = fp[i]
        else:
            ah[i] = (f[i + 1] - f[i]) / (u[i + 1] - u[i])
    fh = np.zeros(len(u) - 1)
    fh = 0.5 * (f[0:-1] + f[1:]) - 0.5 * dt / dx * ah ** 2 * (u[1:] - u[0:-1])
    return fh

def enquist_osher(dx, u):
    um, up = u[:-1], u[1:]
    integrand = lambda theta: np.abs(fluxp(theta))
    a = um
    b = up
    integral = (b-a)/6 * (integrand(a) + 4*integrand((a+b)/2) + integrand(b)) # use Simpson rule to approximate
    return (flux(um) + flux(up)) / 2 - integral/2

def roe(dx, u):
    um, up = u[:-1], u[1:]
    A_hat = np.zeros(len(um))
    for j in range(len(um)):
        if np.abs(um[j] - up[j]) < 1e-7:
            A_hat[j] = fluxp(um[j])
        else:
            A_hat[j] = (flux(up[j]) - flux(um[j])) / (up[j] - um[j])
    fh = np.where(A_hat >= 0, flux(um), flux(up))
    return fh

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


def lax_friedrichs_m(islope, dx, u):
    um, up = getslope(dx, u, islope)

    ump = np.hstack((np.abs(fluxp(um)), np.abs(fluxp(up))))
    alp = max(ump)
    fh = 0.5 * (flux(um) + flux(up)) - 0.5 * alp * (up - um)
    return fh


def roe_m(islope, dx, u):
    um, up = getslope(dx, u, islope)
    A_hat = np.zeros(len(um))
    for j in range(len(um)):
        if np.abs(um[j] - up[j]) < 1e-7:
            A_hat[j] = fluxp(um[j])
        else:
            A_hat[j] = (flux(up[j]) - flux(um[j])) / (up[j] - um[j])
    fh = np.where(A_hat >= 0, flux(um), flux(up))
    return fh


def rusanov_m(islope, dx, u):
    um, up = getslope(dx, u, islope)

    ## advection equation
    # alp = np.maximum(np.abs(fluxp(um)), np.abs(fluxp(up)))
    # fh = 0.5 * (flux(um) + flux(up)) - 0.5 * alp * (up - um)

    ## burgers' equation
    #    alp = np.maximum(np.absolute(um), np.absolute(up))
    #    fh = 0.5*(flux(um)+flux(up))-0.5*alp*(up-um)

    fh = (flux(um) + flux(up)) / 2 - np.max([np.abs(fluxp(um)), np.abs(fluxp(up))]) / 2 * (up - um)
    return fh

def enquist_osher_m(islope, dx, u):
    um, up = getslope(dx, u, islope)
    integrand = lambda theta: np.abs(fluxp(theta))
    a = um
    b = up
    integral = (b - a) / 6 * (integrand(a) + 4 * integrand((a + b) / 2) + integrand(b))  # use Simpson rule to approximate
    return (flux(um) + flux(up)) / 2 - integral / 2


def godunov_m(islope, dx, u):
    f = flux(u) * np.ones(len(u))
    fp = fluxp(u) * np.ones(len(u))
    um, up = getslope(dx, u, islope)

    fh = np.zeros(len(um))
    # for i in range((len(um))):
    #     ## advection equation
    #     if um[i] <= up[i]:
    #         fh[i] = min(flux(um[i]), flux(up[i]))
    #     else:
    #         fh[i] = max(flux(um[i]), flux(up[i]))

        ## burgers's equation
    #        a = max(um[i], 0)
    #        b = min(up[i], 0)
    #        fh[i] = max(flux(a), flux(b))

    fh = np.zeros(len(um))
    for i in range(len(um)):
        if um[i] <= up[i]:
            uu = np.linspace(um[i], up[i], 100)
            ff = flux(uu) * np.ones(len(uu))
            fh[i] = min(ff)
        else:
            uu = np.linspace(um[i], up[i], 100)
            ff = flux(uu) * np.ones(len(uu))
            fh[i] = max(ff)
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


if __name__ == "__main__":
    ## setup
    cfl = 0.5
    tend = 1.5/np.pi

    xleft = 0
    xright = 2


    # which_bc = "neumann"
    which_bc = "periodic"


    iordert = 2
    # iordert 1: euler
    # iordert 2: rk2
    icase = 3
    # icase 1: "upwind", "godunov", "lax_friedrichs", "rusanov", "lax_wendroff", "enquist_osher", roe
    # icase 2: "beam-warming"
    # icase 3: "lax_friedrichs_m", "rusanov_m", "godunov_m", "enquist_osher_m", "roe_m"
    ischeme = "enquist_osher_m"
    islope = "minmod"
    # "zero", "minmod", "superbee", "mc", "vanleer"

    ii = np.arange(2, 9)
    NN = 10 * 2 ** ii

    L1_ErrorStore = np.zeros(len(NN))
    L2_ErrorStore = np.zeros(len(NN))
    L8_ErrorStore = np.zeros(len(NN))

    L1_OrderStore = np.zeros(len(NN) - 1)
    L2_OrderStore = np.zeros(len(NN) - 1)
    L8_OrderStore = np.zeros(len(NN) - 1)
    numerical_solutions = []
    for index in range(len(NN)):
        N = NN[index]
        N_ = N + 1

        ## uniform grid
        dx = (xright - xleft) / N
        xh = np.linspace(xleft - 0.5 * dx, xright + 0.5 * dx, N_ + 1)
        xi = np.linspace(xleft, xright, N_)
        dt = cfl * dx

        ## initialize
        uc = np.zeros((iordert, N_ + 4))  # 2 ghost points for the boundary on the left and 2 ghost points on the right.
        uc[0, 2:-2] = init(dx, xh)

        ## exact solution
        uex = np.zeros(N_)
        uex = exactu(tend, xi)

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

    L1_OrderStore = ErrorOrder(L1_ErrorStore, NN)
    L2_OrderStore = ErrorOrder(L1_ErrorStore, NN)
    L8_OrderStore = ErrorOrder(L1_ErrorStore, NN)


    plt.plot(np.linspace(xleft, xright, NN[-1] + 1), numerical_solutions[-1], '-',
             label=f"{ischeme}", linewidth=1)
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.plot(x := np.linspace(xleft, xright, NN[-1] + 1), exactu(tend, xi), label="exact solution", linewidth=1,
             color="black")
    plt.title(f"{NN[-1] + 1} Points")
    plt.legend()
    plt.show()

    for i, N in enumerate(NN):
        plt.scatter(np.linspace(xleft, xright, N+1), numerical_solutions[i], label=f"{N} mesh points", s=1)
    plt.show()

    mesh_widths = 1 / NN
    plt.loglog(mesh_widths, L1_ErrorStore, label="$L^{1}$-Error")
    plt.loglog(mesh_widths, L2_ErrorStore, label="$L^{2}$-Error")
    plt.loglog(mesh_widths, L8_ErrorStore, label="$L^{\infty}$-Error")
    plt.loglog(mesh_widths, 10 * mesh_widths, label="$h^{1}$ (for comparison)", linestyle='dashed')
    plt.loglog(mesh_widths, 100 * mesh_widths ** 2, label="$h^{2}$ (for comparison)", linestyle='dashed')
    plt.xlabel("mesh width h")
    plt.ylabel("error")
    plt.legend()
    plt.show()

    print("L1 average convergence rate:", np.polyfit(np.log(mesh_widths), np.log(L1_ErrorStore), 1)[0])
    print("L2 average convergence rate:", np.polyfit(np.log(mesh_widths), np.log(L2_ErrorStore), 1)[0])
    print("Linf average convergence rate:", np.polyfit(np.log(mesh_widths), np.log(L8_ErrorStore), 1)[0])

    print(f"N={NN[0] + 1}")
    print(f"L1 Error at N={NN[0] + 1}: {L1_ErrorStore[0]}")
    print(f"L2 Error  at N={NN[0] + 1}: {L2_ErrorStore[0]}")

    print(f"Linf Error at N={NN[0] + 1}: {L8_ErrorStore[0]}")
    rates_l1 = []
    rates_l2 = []
    rates_linf = []
    precision = 4
    for i, N in enumerate(NN[1:]):
        print(f"N={N}")
        print(f"L1 Error at N={N+1}:", L1_ErrorStore[i + 1])
        print(f"L2 Error  at N={N+1}:", L2_ErrorStore[i + 1])
        print(f"Linf Error at N={N+1}:", L8_ErrorStore[i + 1])
        rate_l1 = np.polyfit(np.log(mesh_widths[i:i + 2]), np.log(L1_ErrorStore[i:i + 2]), 1)[0]
        rate_l2 = np.polyfit(np.log(mesh_widths[i:i + 2]), np.log(L2_ErrorStore[i:i + 2]), 1)[0]
        rate_linf = np.polyfit(np.log(mesh_widths[i:i + 2]), np.log(L8_ErrorStore[i:i + 2]), 1)[0]
        rates_l1.append(np.round(rate_l1, precision))
        rates_l2.append(np.round(rate_l2, precision))
        rates_linf.append(np.round(rate_linf, precision))

        print(f"L1 local convergence rate at N={N+1} :", rate_l1)
        print(f"L2 local convergence rate  at N={N+1}:", rate_l2)
        print(f"Linf local  convergence rate at N={N+1}:", rate_linf)







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
