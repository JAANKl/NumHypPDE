import numpy as np
import math, sys
import matplotlib.pyplot as plt
import scipy.integrate as integrate

## Finite Volume Method.
"Function: solve u_t+u_x=0."
"u_0(x) = -1, x<0; 1, x>0. x:[-5,5]"
    
def init(dx, x):
    u0_ = np.zeros(len(x)-1)
    for i in range(len(x)-1):
        if x[i+1] <= 0:
            u0_[i] = -1
        elif x[i] > 0:
            u0_[i] = 1
        else:
            u0_[i] = (x[i]+x[i+1])/dx
    return u0_
    
def exactu(t, x):
    uex = lambda t, x: np.where(x < t, -1, 1)
    u = uex(t, x)
    return u
    
def bc(uc):
    ## zero order extrapolation
    uc[0:2] = uc[2]
    uc[-2:] = uc[-3]
    return uc
        
def getflux(ischeme, dt, dx, flux, fluxp, u):
    f = flux(u)*np.ones(len(u))
    fp = fluxp(u)*np.ones(len(u))
        
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
        fh = lax_friedrichs_m(dt, dx, u, fp, flux)
    else:
        sys.exit("ERROR: this is not defined here.")
    rhs = fh[1:]-fh[0:-1]
    return rhs
    
def godunov(u, f, flux):
    fh = np.zeros(len(u)-1)
    for i in range(len(u)-1):
        ## advection equation
        if u[i] <= u[i+1]:
            fh[i] = min(f[i], f[i+1])
        else:
            fh[i] = max(f[i], f[i+1])
        
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
#        if u[i] <= u[i+1]:
#            uu = np.linspace(u[i], u[i+1], 100)
#            ff = flux(uu)*np.ones(len(uu))
#            fh[i] = min(ff)
#        else:
#            uu = np.linspace(u[i+1], u[i], 100)
#            ff = flux(uu)*np.ones(len(uu))
#            fh[i] = max(ff)
    return fh
    
def lax_friedrichs(u, f, fp):
    fh = np.zeros(len(u)-1)
    
    alp = max(np.absolute(fp))
    fh = 0.5*(f[0:-1]+f[1:])-0.5*alp*(u[1:]-u[0:-1])
    return fh
    
def rusanov(u, f, fp, fluxp):
    fh = np.zeros(len(u)-1)
    
    ## advection equation
    alp = np.absolute(fp[0:-1])
    fh = 0.5*(f[0:-1]+f[1:])-alp/2*(u[1:]-u[0:-1])
    
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
    
def beam_warming(dt, dx, u, f, fp):
    ah = np.zeros(len(u)-1)
    for i in range(len(u)-1):
        if u[i] == u[i+1]:
            ah[i] = fp[i]
        else:
            ah[i] = (f[i+1]-f[i])/(u[i+1]-u[i])
    fh = f[1:]+0.5*ah*(1-dt/dx*ah)*(u[1:]-u[0:-1])
    return fh
    
def lax_wendroff(dt, dx, u, f, fp):
    ah = np.zeros(len(u)-1)
    for i in range(len(u)-1):
        if u[i] == u[i+1]:
            ah[i] = fp[i]
        else:
            ah[i] = (f[i+1]-f[i])/(u[i+1]-u[i])
    fh = np.zeros(len(u)-1)
    fh = 0.5*(f[0:-1]+f[1:])-0.5*dt/dx*ah**2*(u[1:]-u[0:-1])
    return fh
    
def lax_friedrichs_m(dt, dx, u, fp, flux):
    um, up = minmod_slope(dx, u)
    alp = max(np.absolute(fp))
    fh = 0.5*(flux(um)+flux(up))-0.5*alp*(up-um)
    return fh
    
def minmod(a, b):
    return (np.sign(a)+np.sign(b))/2.0*np.minimum(np.abs(a), np.abs(b))
    
def minmod_slope(dx, u):
    sigma = minmod(u[2:]-u[1:-1], u[1:-1]-u[0:-2])/dx
    um = np.zeros(len(sigma)-1)
    up = np.zeros(len(sigma)-1)
    um = u[1:-2]+sigma[0:-1]*dx*0.5
    up = u[2:-1]+sigma[1:]*(-dx)*0.5
    return um, up
    
def upwind_slope(dx, u):
    sigma = (u[1:-1]-u[0:-2])/dx
    um = np.zeros(len(sigma)-1)
    up = np.zeros(len(sigma)-1)
    um = u[1:-2]+sigma[0:-1]*dx*0.5
    up = u[2:-1]+sigma[1:]*(-dx)*0.5
    return um, up

def downwind_slope(dx, u):
    sigma = (u[2:]-u[1:-1])/dx
    um = np.zeros(len(sigma)-1)
    up = np.zeros(len(sigma)-1)
    um = u[1:-2]+sigma[0:-1]*dx*0.5
    up = u[2:-1]+sigma[1:]*(-dx)*0.5
    return um, up
        
if __name__ == "__main__":
    ## setup
    cfl = 0.4
    tend = 1

    xleft = -5
    xright = 5
    
    flux = lambda u: u
    fluxp = lambda u: 1

    icase = 1
    ischeme = "lax-friedrichs"
    # icase 1: "upwind", "godunov", "lax-friedrichs", "rusanov", "lax-wendroff"
    # icase 2: "beam-warming"
    # icase 3: "lax-friedrichs_m"
    
    N = 101
    N_ = N+1
    ## uniform grid
    dx = (xright-xleft)/N
    xh = np.linspace(xleft-0.5*dx, xright+0.5*dx, N_+1)
    xi = np.linspace(xleft, xright, N_)
    dt = cfl*dx
    
    ## initialize
    uc = np.zeros(N_+4) # 2 ghost points for the boundary on the left and 2 ghost points on the right.
    uc[2:-2] = init(dx, xh)
   
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
        bc(uc)
        
        if icase == 1:
            rhs = getflux(ischeme, dt, dx, flux, fluxp, uc[1:-1])
        elif icase == 2:
            rhs = getflux(ischeme, dt, dx, flux, fluxp, uc[0:-2])
        elif icase == 3:
            rhs = getflux(ischeme, dt, dx, flux, fluxp, uc)
        else:
            sys.exit("ERROR: this is not defined here.")
        
        ## Euler forward
        uc[2:-2] = uc[2:-2]-dt/dx*rhs
    ## call bc
    bc(uc)
    uc = uc[2:-2]
    
    with open ('num_sol_{}.txt'.format(ischeme), 'w', encoding='utf-8') as file:
        for index in range(len(xi)):
            file.write("%f %f %f \n"%(xi[index], uex[index], uc[index]))

    plt.figure()
    plt.plot(xi, uc, 'r--', markerfacecolor="None", markersize=1, label= '{}'.format(ischeme))
    plt.plot(xi, uex, 'k-', markerfacecolor="None", markersize=1, label='exact')
    plt.legend()
    plt.xlabel('x');plt.ylabel('u')
    plt.show()
    
    

    
   


    
        
    
    


