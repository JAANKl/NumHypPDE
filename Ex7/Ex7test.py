import numpy as np
import math
import matplotlib.pyplot as plt

"Function: solve u_t+2*u_x=0"

def init(x):
    u0 = lambda x: np.where(x < 0, -1, 1)
    u0_ = u0(x)
    return u0_
    
def exactu(t, x):
    uex = lambda t, x: np.where(x < t,-1, 1)
    u = uex(t, x)
    return u

def bc(N, uc):
    # periodic
    uc[0] = uc[1]
    uc[N+1] = uc[N]
    return uc
        
def getflux(uc, lambda_):
    flux = lambda u: u
    fluxp = lambda u: 1
    f = flux(uc)*np.ones(len(uc))
    fp = fluxp(uc)*np.ones(len(uc))
    

    #Lax Wendroff
    #fh = lax_wendroff(lambda_, uc, f, fp)
    #rhs = fh[1:]-fh[0:-1]

    #Beam Warming
    rhs = beam_warming(lambda_, uc, fp)
    
    return rhs

def lax_wendroff(lambda_, u, f, fp):
    
    ah = np.zeros(len(u)-1)
    for i in range(len(u)-1):
        if u[i] == u[i+1]:
            ah[i] = fp[i]
        else:
            ah[i] = (f[i+1]-f[i])/(u[i+1]-u[i])
    fh = np.zeros(len(u)-1)
    fh = 0.5*(f[0:-1]+f[1:])-0.5*lambda_*ah**2*(u[1:]-u[0:-1])
    return fh

    # Linear advection equation
    #a = fp[0]
    #fh = np.zeros(len(u)-1)
    #fh[1:] = a/2 * (u[2:] - u[0:-2]) + a**2/2 * lambda_ * (u[2:]-2*u[1:-1]+u[0:-2])
    #fh[0]=fh[1]
    #return fh[1:]


def beam_warming(lambda_, u, fp):
    fh = np.zeros(len(u) - 1)  # Adjust the length of fh
    a = fp[0:-3]
    fh[2:] = 0.5 * a * (3 * u[2:-1] - 4 * u[1:-2] + u[:-3]) + a**2/2 * lambda_ * (u[2:-1] - 2 * u[1:-2] + u[:-3])
    fh[0]=fh[2]
    fh[1]=fh[2]
    return fh[1:]

if __name__ == "__main__":
    # setup
    cfl = 0.4
    tend = 1

    xleft = -5
    xright = 5
    
    N = 100
    
    # uniform grid
    x = np.linspace(xleft, xright, N+1)
    dx = (xright-xleft)/N
    dt = cfl*dx/2

    # initialize
    uc = np.zeros(N+2) # there is 1 ghost point for the boundary on the left.
    uc[1:] = init(x)
    rhs = np.zeros(N)
    
    # exact solution
    uex = np.zeros(N+1)
    uex = exactu(tend, x)

    # time evolution
    time = 0
    kt = 0
    
    kmax = 10**5
    #while (time < tend)&(kt<kmax):
    while time < tend:
        kt = kt+1
        if time+dt >= tend:
            dt = tend-time
        time = time+dt
    
        uc = bc(N, uc)
        rhs = getflux(uc, dt/dx)
    
        # Euler forward
        uc[1:-1] = uc[1:-1]-dt/dx*rhs
    
    uc = bc(N, uc)
    
    plt.plot(x, uc[1:], label = 'N = 100')
    plt.legend()
    plt.xlabel('x');plt.ylabel('u')
    plt.plot(x, uex, 'k-', label='exact')
    plt.legend()
    plt.show()