import numpy as np
import math, sys
import matplotlib.pyplot as plt
import scipy.integrate as integrate

"Function: solve u_t+u_x=0."
"u_0(x) = -1, x<0; 1, x>0. x:[-5,5]"

def time_step(cfl, dx, uc, fluxp):
    fp = fluxp(uc)*np.ones(len(uc))
    alpha = max(np.absolute(fp))
    dt = cfl*dx/(alpha+1e-14)
    return dt


def init(dx, x):
    u0_ = np.zeros(len(x)-1)
    for j in range(len(x)-1):
        u0_[j] = 1/dx * integrate.quad(lambda y: np.where(y < 0.5, 2, 1), x[j] - 0.5 * dx, x[j] + 0.5 * dx)[0]
    return u0_
    
def exactu(t, x):
    uex = lambda t, x: np.where(x-t < 0.5, 2, 1)
    u = uex(t, x)
    return u
    
def bc(uc, N):
    ## periodic
    uc[0] = uc[3]  
    uc[1] = uc[3]
    uc[2] = uc[3]

    uc[N+1] = uc[N]
    uc[N+2] = uc[N]

    return uc


def minmod(a, b):
    return (np.sign(a)+np.sign(b))/2.0*np.minimum(np.abs(a), np.abs(b))

def maxmod(a,b):
    return (np.sign(a)+np.sign(b))/2.0*np.maximum(np.abs(a), np.abs(b))

def superbee(a,b):
    return maxmod(minmod(2 * a, b), minmod(a, 2 * b))

def mc(a,b):
    return np.maximum(0, np.sign(a) * np.minimum(np.abs(2*a), np.minimum(np.abs((a+b)/2), np.abs(2*b))))
    if b == 0:
        if a == 0:
            return 0
        elif a > 0:
            return 2
        else:
            return 0
    elif a == 0:
        return 0
    else:
        if np.sign(a) == np.sign(b):
            return 2 * a / (a + b)
        else:
            return 0
        
def van_leer(a,b):
    if b == 0:
        return 0
    else:
        return ((a/b) + np.abs(a/b))/(1 + np.abs(a/b))
        

def limiter(dx, u, limiterfunction):
    sigma = np.zeros(len(u)-2)
    for i in range(0, len(u) - 2):
        sigma[i] = limiterfunction((u[i+2]-u[i+1])/dx, (u[i+1]-u[i])/dx)
    um = np.zeros(len(sigma)-1)
    up = np.zeros(len(sigma)-1)
    um = u[1:-2]+sigma[0:-1]*dx*0.5
    up = u[2:-1]+sigma[1:]*(-dx)*0.5
    return um, up
        
def getflux(ischeme, dt, dx, flux, fluxp, u, limiterfunction):
    f = flux(u)*np.ones(len(u))
    fp = fluxp(u)*np.ones(len(u))
    
    fh = godunov(u, flux, dx, limiterfunction)
    rhs = fh[1:]-fh[0:-1]
    return rhs

def rusanov(u, flux, fp, dx, limiterfunction):
    #fh = np.zeros(len(u)-1)
    um, up = limiter(dx, u, limiterfunction)

    ## advection equation
    alp = np.absolute(fp[2:-2])
    fh = 0.5 * (flux(up)[0:-1] + flux(um)[1:]) - alp/2*(um[1:]-up[0:-1])
    
    ## burgers' equation
#    alp = np.maximum(np.absolute(u[0:-1]), np.absolute(u[1:]))
#    fh = 0.5*(f[0:-1]+f[1:])-0.5*alp*(u[1:]-u[0:-1])
    
    ## general case
#    alp = np.zeros(len(u)-1)
#    for i in range(len(u)-1):
#        a = min(u[i], u[i+1])
#        b = max(u[i], u[i+1])
#        uu = np.linspace(a, b, 100)
#        ff = fluxp(uu)*np.ones(len(uu))
#        alp[i] = max(np.absolute(ff))
#    fh = 0.5*(f[0:-1]+f[1:])-alp/2*(u[1:]-u[0:-1])
    return fh
    

            
def lax_friedrichs_m(dx, u, fp, flux, limiterfunction):
    um, up = limiter(dx, u, limiterfunction)
    alp = max(np.absolute(fp))
    fh = 0.5*(flux(um)+flux(up))-0.5*alp*(up-um)
    return fh

def godunov(u, flux, dx, limiterfunction):

    um, up = limiter(dx, u, limiterfunction)
    fh = np.zeros(len(um)-1)

    for i in range(len(um)-1):
        ## advection equation
        if up[i] <= um[i+1]:
            fh[i] = min(flux(up)[i], flux(um)[i+1])
        else:
            fh[i] = max(flux(up)[i], flux(up)[i+1])
        
        ## burgers's equation
#        a = max(u[i], 0)
#        fh[i] = max(flux(a), flux(b))
#        b = min(u[i+1], 0)

        ## another way for burgers' equation
#        if u[i] <= u[i+1]:
#            if u[i] < 0 < u[i+1]:
#                fh[i] = 0
#            else:
#                fh[i] = min(f[i], f[i+1])
#        else:
#            fh[i] = max(f[i], f[i+1])
            
        ## general case for f(u)
#        if u[i] <= u[i+1]:
#            uu = np.linspace(u[i], u[i+1], 100)
#            ff = flux(uu)*np.ones(len(uu))
#            fh[i] = min(ff)
#        else:
#            uu = np.linspace(u[i+1], u[i], 100)
#            ff = flux(uu)*np.ones(len(uu))
#            fh[i] = max(ff)
    return fh

def L(dx, rhs):
    return -1/dx * rhs

def comp_ustar(uc, dt, N, dx, rhs):
    ustar_ = np.zeros(len(uc))
    ustar_[3:-2] = uc[3:-2] + dt * L(dx, rhs)
    bc(ustar_, N)
    return ustar_
        
if __name__ == "__main__":
    ## setup
    cfl = 0.4
    tend = 1

    xleft = 0
    xright = 2
    
    flux = lambda u: u
    fluxp = lambda u: 1

    ischeme = "rusanov"
    
    N = 100
    N_ = N+1
    ## uniform grid
    dx = (xright-xleft)/N
    xh = np.linspace(xleft-0.5*dx, xright+0.5*dx, N_+1)
    xi = np.linspace(xleft, xright, N_)
    dt = cfl * dx

    ## initialize
    uc_minmod = np.zeros(N_+4) # 2 ghost points for the boundary on the left and 2 ghost points on the right.
    uc_minmod[2:-2] = init(dx, xh)

    uc_superbee = np.zeros(N_+4) # 2 ghost points for the boundary on the left and 2 ghost points on the right.
    uc_superbee[2:-2] = init(dx, xh)

    uc_mc = np.zeros(N_+4) # 2 ghost points for the boundary on the left and 2 ghost points on the right.
    uc_mc[2:-2] = init(dx, xh)

    uc_van_leer = np.zeros(N_+4) # 2 ghost points for the boundary on the left and 2 ghost points on the right.
    uc_van_leer[2:-2] = init(dx, xh)
   
    ## exact solution
    uex = np.zeros(N_)
    uex = exactu(tend, xi)

    ## time evolution
    time = 0
    kt = 0
    
    kmax = 10**5
    #while (time < tend)&(kt<kmax):
    while time < tend:
        kt = kt+1
        if time+dt >= tend:
            dt = tend-time
        time = time+dt
        
        ## call bc
        bc(uc_minmod, N)
        bc(uc_superbee, N)
        bc(uc_mc, N)
        bc(uc_van_leer, N)
        
        rhs_minmod = getflux(ischeme, dt, dx, flux, fluxp, uc_minmod, minmod)
        rhs_superbee = getflux(ischeme, dt, dx, flux, fluxp, uc_superbee, superbee)
        rhs_mc = getflux(ischeme, dt, dx, flux, fluxp, uc_mc, mc)
        rhs_van_leer = getflux(ischeme, dt, dx, flux, fluxp, uc_van_leer, van_leer)

        ## Euler forward
        uc_minmod[3:-2] = uc_minmod[3:-2]-dt/dx*rhs_minmod
        uc_superbee[3:-2] = uc_superbee[3:-2]-dt/dx*rhs_superbee
        uc_mc[3:-2] = uc_mc[3:-2]-dt/dx*rhs_mc
        uc_van_leer[3:-2] = uc_van_leer[3:-2]-dt/dx*rhs_van_leer

        ## SSP Runge-Kutta
        #uc[3:-2] = 1/2 * (uc[3:-2] + ustarstar[3:-2])


    ## call bc
    bc(uc_minmod, N)
    uc_minmod = uc_minmod[2:-2]

    bc(uc_superbee, N)
    uc_superbee = uc_superbee[2:-2]

    bc(uc_mc, N)
    uc_mc = uc_mc[2:-2]

    bc(uc_van_leer, N)
    uc_van_leer = uc_van_leer[2:-2]
    

    plt.figure()
    plt.plot(xi, uc_minmod, '--', markerfacecolor="None", markersize=1, label= 'minmod')
    plt.plot(xi, uc_superbee, '--', markerfacecolor="None", markersize=1, label= 'superbee')
    plt.plot(xi, uc_mc, '--', markerfacecolor="None", markersize=1, label= 'mc')
    plt.plot(xi, uc_van_leer, '--', markerfacecolor="None", markersize=1, label= 'van leer')
    plt.plot(xi, uex, 'k-', markerfacecolor="None", markersize=1, label='exact')
    plt.legend()
    plt.xlabel('x');plt.ylabel('u')
    plt.show()
    