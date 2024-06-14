import numpy as np
import math, sys
import matplotlib.pyplot as plt
import scipy.integrate as integrate

## Finite Volume Method.
"Function: solve u_t+u_x=0."
"u_0(x) = 2, x<0.5; 1, x>0.5, x=[0, 1]"


def init(dx, x):
    # u0_ = np.zeros(len(x) - 1)
    # for i in range(len(x) - 1):
    #     if x[i + 1] < 0:
    #         u0_[i] = 1
    #     elif x[i] >= 0:
    #         u0_[i] = 0
    #     else:
    #         u0_[i] = (0 - 2 * x[i] + x[i + 1]) / dx
    # return u0_
    return exactu(0,x[:-1])


def exactu(t, x):
    # Rarefaction:
    u = np.zeros(len(x))
    for i in range(len(x)):
        if x[i] <= 0:
            u[i] = 0
        elif x[i] <= t:
            u[i] = x[i]/t
        else:
            u[i] = 1
    return u
    # Shock:
    # return np.where((x < 0.5*t), 0, 1)


def apply_boundary_condition(uc, which_bc):
    if which_bc == "neumann":
        # Neumann bc:
        ## zero order extrapolation
        uc[0:2] = uc[2]
        uc[-2:] = uc[-3]
    elif which_bc == "periodic":
        # Periodic bc:
        uc[0:2] = uc[-5:-3]
        uc[-2:] = uc[3:5]
    else:
        raise NotImplementedError("Only Neumann or Periodic boundary conditions possible.")
    return uc


def getflux(ischeme, islope, dt, dx, flux, fluxp, u):
    f = flux(u) * np.ones(len(u))
    fp = fluxp(u) * np.ones(len(u))

    if ischeme == "upwind":
        fh = upwind(u, f, fp)
    elif ischeme == "godunov":
        fh = godunov(u, f, flux)
    elif ischeme == "lax-friedrichs":
        fh = lax_friedrichs(u, f, fp)
    elif ischeme == "rusanov":
        fh = rusanov(u, f, fp, fluxp)
    elif ischeme == "lax-wendroff":
        fh = lax_wendroff(dt, dx, u, f, fp)
    elif ischeme == "beam-warming":
        fh = beam_warming(dt, dx, u, f, fp)
    elif ischeme == "lax-friedrichs_m":
        fh = lax_friedrichs_m(islope, dx, u, flux, fluxp)
    elif ischeme == "rusanov_m":
        fh = rusanov_m(islope, dx, u, flux, fluxp)
    elif ischeme == "godunov_m":
        fh = godunov_m(islope, dx, u, fp, flux)
    else:
        sys.exit("ERROR: this is not defined here.")
    rhs = -(fh[1:] - fh[0:-1]) / dx
    return rhs


def godunov(u, f, flux):
    fh = np.zeros(len(u) - 1)
    for i in range(len(u) - 1):
        ## advection equation
        # if u[i] <= u[i + 1]:
        #     fh[i] = min(f[i], f[i + 1])
        # else:
        #     fh[i] = max(f[i], f[i + 1])

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
           if u[i] <= u[i+1]:
               uu = np.linspace(u[i], u[i+1], 100)
               ff = flux(uu)*np.ones(len(uu))
               fh[i] = min(ff)
           else:
               uu = np.linspace(u[i+1], u[i], 100)
               ff = flux(uu)*np.ones(len(uu))
               fh[i] = max(ff)
    return fh


def lax_friedrichs(u, f, fp):
    fh = np.zeros(len(u) - 1)

    alp = max(np.absolute(fp))
    fh = 0.5 * (f[0:-1] + f[1:]) - 0.5 * alp * (u[1:] - u[0:-1])
    return fh


def rusanov(u, f, fp, fluxp):
    fh = np.zeros(len(u) - 1)

    ## advection equation
    # alp = np.absolute(fp[0:-1])
    # fh = 0.5 * (f[0:-1] + f[1:]) - alp / 2 * (u[1:] - u[0:-1])

    ## burgers' equation
    #    alp = np.maximum(np.absolute(u[0:-1]), np.absolute(u[1:]))
    #    fh = 0.5*(f[0:-1]+f[1:])-0.5*alp*(u[1:]-u[0:-1])

    ## general case
    alp = np.zeros(len(u)-1)
    for i in range(len(u)-1):
        a = min(u[i], u[i+1])
        b = max(u[i], u[i+1])
        uu = np.linspace(a, b, 100)
        ff = fluxp(uu)*np.ones(len(uu))
        alp[i] = max(np.absolute(ff))
    fh = 0.5*(f[0:-1]+f[1:])-alp/2*(u[1:]-u[0:-1])
    return fh


def beam_warming(dt, dx, u, f, fp):
    ah = np.zeros(len(u) - 1)
    for i in range(len(u) - 1):
        if u[i] == u[i + 1]:
            ah[i] = fp[i]
        else:
            ah[i] = (f[i + 1] - f[i]) / (u[i + 1] - u[i])
    fh = f[1:] + 0.5 * ah * (1 - dt / dx * ah) * (u[1:] - u[0:-1])
    return fh


def lax_wendroff(dt, dx, u, f, fp):
    ah = np.zeros(len(u) - 1)
    for i in range(len(u) - 1):
        if u[i] == u[i + 1]:
            ah[i] = fp[i]
        else:
            ah[i] = (f[i + 1] - f[i]) / (u[i + 1] - u[i])
    fh = np.zeros(len(u) - 1)
    fh = 0.5 * (f[0:-1] + f[1:]) - 0.5 * dt / dx * ah ** 2 * (u[1:] - u[0:-1])
    return fh


def lax_friedrichs_m(islope, dx, u, flux, fluxp):
    if islope == "minmod":
        um, up = minmod_slope(dx, u)
    elif islope == "superbee":
        um, up = superbee_slope(dx, u)
    elif islope == "mc":
        um, up = mc_slope(dx, u)
    elif islope == "vanleer":
        um, up = vanleer_slope(dx, u)
    else:
        sys.exit("ERROR: this is not defined here.")

    ump = np.hstack((np.abs(fluxp(um)), np.abs(fluxp(up))))
    alp = max(ump)
    fh = 0.5 * (flux(um) + flux(up)) - 0.5 * alp * (up - um)
    return fh


def rusanov_m(islope, dx, u, flux, fluxp):
    if islope == "minmod":
        um, up = minmod_slope(dx, u)
    elif islope == "superbee":
        um, up = superbee_slope(dx, u)
    elif islope == "mc":
        um, up = mc_slope(dx, u)
    elif islope == "vanleer":
        um, up = vanleer_slope(dx, u)
    else:
        sys.exit("ERROR: this is not defined here.")

    ## advection equation
    alp = np.maximum(np.abs(fluxp(um)), np.abs(fluxp(up)))
    fh = 0.5 * (flux(um) + flux(up)) - 0.5 * alp * (up - um)

    ## burgers' equation
    #    alp = np.maximum(np.absolute(um), np.absolute(up))
    #    fh = 0.5*(flux(um)+flux(up))-0.5*alp*(up-um)
    return fh


def godunov_m(islope, dx, u, fp, flux):
    if islope == "minmod":
        um, up = minmod_slope(dx, u)
    elif islope == "superbee":
        um, up = superbee_slope(dx, u)
    elif islope == "mc":
        um, up = mc_slope(dx, u)
    elif islope == "vanleer":
        um, up = vanleer_slope(dx, u)
    else:
        sys.exit("ERROR: this is not defined here.")
    fh = np.zeros(len(um))
    for i in range((len(um))):
        ## advection equation
        if um[i] <= up[i]:
            fh[i] = min(flux(um[i]), flux(up[i]))
        else:
            fh[i] = max(flux(um[i]), flux(up[i]))

        ## burgers's equation
    #        a = max(um[i], 0)
    #        b = min(up[i], 0)
    #        fh[i] = max(flux(a), flux(b))
    return fh


def minmod(a, b):
    return (np.sign(a) + np.sign(b)) / 2.0 * np.minimum(np.abs(a), np.abs(b))


def minmod2(a, b, c):
    return (np.sign(a) + np.sign(b) + np.sign(c)) / 3.0 * np.minimum(np.abs(a), np.abs(b), np.abs(c))


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
    cfl = 0.4
    tend = 1

    xleft = -1
    xright = 3

    flux = lambda u: u**2/2
    fluxprime = lambda u: u

    iordert = 1
    # iordert 1: euler
    # iordert 2: rk2
    icase = 1
    # icase 1: "upwind", "godunov", "lax-friedrichs", "rusanov", "lax-wendroff"
    # icase 2: "beam-warming"
    # icase 3: "lax-friedrichs_m", "rusanov_m", "godunov_m" (for second order reconstruction)
    ischeme = "lax-wendroff"
    islope = "mc"
    # "minmod", "superbee", "mc", "vanleer"
    which_boundary_condition = "neumann"
    # "neumann", "periodic"

    N = 100
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
    counter = 0
    counter_max = 10 ** 5
    # while (time < tend)&(kt<kmax):
    while time < tend:

        if time + dt >= tend:
            dt = tend - time
        time = time + dt
        counter = counter + 1

        if iordert == 1:
            ## call bc
            uc[0, :] = apply_boundary_condition(uc[0, :], which_boundary_condition)

            ## spatial discretization
            if icase == 1:
                rhs = getflux(ischeme, islope, dt, dx, flux, fluxprime, uc[0, 1:-1])
            elif icase == 2:
                rhs = getflux(ischeme, islope, dt, dx, flux, fluxprime, uc[0, 0:-2])
            elif icase == 3:
                rhs = getflux(ischeme, islope, dt, dx, flux, fluxprime, uc[0, :])
            else:
                sys.exit("ERROR: this is not defined here.")
            euler_forward(dt, uc, rhs)
        elif iordert == 2:
            for k in range(iordert):
                ## call bc
                uc[k, :] = apply_boundary_condition(uc[k, :], which_boundary_condition)

                ## spatial discretization
                if icase == 1:
                    rhs = getflux(ischeme, islope, dt, dx, flux, fluxprime, uc[k, 1:-1])
                elif icase == 2:
                    rhs = getflux(ischeme, islope, dt, dx, flux, fluxprime, uc[k, 0:-2])
                elif icase == 3:
                    rhs = getflux(ischeme, islope, dt, dx, flux, fluxprime, uc[k, :])
                else:
                    sys.exit("ERROR: this is not defined here.")
                rk2(k, dt, uc, rhs)
    uc[0, :] = apply_boundary_condition(uc[0, :], which_boundary_condition)
    uc_ = uc[0, 2:-2]

    ## plot numerical solution
    # with open('num_sol_{}_{}_sh.txt'.format(ischeme, islope), 'w', encoding='utf-8') as file:
    #     for index in range(len(xi)):
    #         file.write("%f %f %f \n" % (xi[index], uex[index], uc_[index]))

    plt.figure()
    plt.plot(xi, uc_, 'r--', markerfacecolor="None", markersize=1, label='{}'.format(ischeme))
    plt.plot(xi, uex, 'k-', markerfacecolor="None", markersize=1, label='exact')
    plt.legend()
    plt.xlabel('x');
    plt.ylabel('u')
    plt.show()