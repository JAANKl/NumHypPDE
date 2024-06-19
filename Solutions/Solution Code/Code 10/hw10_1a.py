import numpy as np
import math, sys
import matplotlib.pyplot as plt
import scipy.integrate as integrate

## Finite Volume Method.
## Linear Systems

# First order system solvers!!

"Function: solve u_t+A*u_x=0."
    
def init(m, dx, x):
    u0_ = np.zeros((m, len(x)-1))
    for i in range(len(x)-1):
        if x[i+1] <= 0:
            u0_[0, i] = 0
            u0_[1, i] = 1
        elif x[i] > 0:
            u0_[0, i] = 1
            u0_[1, i] = 1
        else:
            u0_[0, i] = x[i+1]/dx
            u0_[1, i] = 1
    return u0_
    
def exactu(m, t, x):
    uex = np.zeros((m, len(x)))
    for i in range(len(x)):
        if x[i] <= -2*t:
            uex[0, i] = 0
            uex[1, i] = 1
        elif x[i] > 2*t:
            uex[0, i] = 1
            uex[1, i] = 1
        else:
            uex[0, i] = 0.5
            uex[1, i] = 3/4
    return uex
    
def bc(uc):
    ## zero order extrapolation
    for im in range(np.size(uc, 0)):
        uc[im, 0:2] = uc[im, 2]
        uc[im, -2:] = uc[im, -3]
    return uc
        
def getflux(ischeme, islope, dt, dx, A, Sigma, L, R, u):
    #fh = np.zeros((np.size(u, 0), np.size(u, 1)-1))
    if ischeme == "godunov_sys":
        fh = godunov_sys(A, Sigma, L, R, u)
    elif ischeme == "lax_friedrichs_sys":
        fh = lax_friedrichs_sys(A, u, dx, dt)
    elif ischeme == "rusanov_sys":
        fh = rusanov_sys(A, u)
    # elif ischeme == "godunov_sys_m":
    #     fh = godunov_sys_m(islope, dx, u, fp, flux)
    else:
        sys.exit("ERROR: this is not defined here.")
    
    for im in range(m):
        rhs[im] = -(fh[im, 1:]-fh[im, 0:-1])/dx
    return rhs
    
def godunov_sys(A, Sigma, L, R, u):
    Sigma_ab = np.absolute(Sigma)
    
    fh = 0.5*A@(u[:, 0:-1]+u[:, 1:])-0.5*R@Sigma_ab@L@(u[:, 1:]-u[:, 0:-1])
    
    #fh = np.zeros((np.size(u, 0), np.size(u, 1)-1))
#    for i in range(np.size(u, 1)-1):
#        fh[:, i] = 0.5*A@(u[:, i]+u[:, i+1])-0.5*R@Sigma_ab@L@(u[:, i+1]-u[:, i])
    return fh

def lax_friedrichs_sys(A, u, dx, dt):
    fh = 0.5 * A @ (u[:, 0:-1] + u[:, 1:]) - 0.5 * dx/dt* (u[:, 1:] - u[:, 0:-1])
    return fh

def rusanov_sys(A, u):
    lambda_max = 2
    fh = 0.5 * A @ (u[:, 0:-1] + u[:, 1:]) - 0.5 * lambda_max* (u[:, 1:] - u[:, 0:-1])
    return fh

def godunov_sys_m(islope, dx, A, Sigma, L, R, u):
    Sigma_ab = np.absolute(Sigma)
    
    w = cproject(L, u)
    wm = np.zeros((np.size(w, 0), np.size(w, 1)-3))
    wp = np.zeros((np.size(w, 0), np.size(w, 1)-3))
    for im in range(np.size(w, 0)):
        if islope == "minmod":
            wm[im, :], wp[im, :] = minmod_slope(dx, w[im, :])
        elif islope == "superbee":
            wm[im, :], wp[im, :] = superbee_slope(dx, w[im, :])
        elif islope == "mc":
            wm[im, :], wp[im, :] = mc_slope(dx, w[im, :])
        else:
            sys.exit("ERROR: this is not defined here.")
    um = cprojectbk(R, wm)
    up = cprojectbk(R, wp)
    
    fh = 0.5*A@(um+up)-0.5*R@Sigma_ab@L@(up-um)
    return fh

def cproject(u, L):
    w = L@u
    return w
    
def cprojectbk(w, R):
    u = R@w
    return u

def minmod(a, b):
    return (np.sign(a)+np.sign(b))/2.0*np.minimum(np.abs(a), np.abs(b))

def minmod2(a, b, c):
    return (np.sign(a)+np.sign(b)+np.sign(c))/3.0*np.minimum(np.abs(a), np.abs(b), np.abs(c))
    
def minmod_slope(dx, u):
    sigma = minmod(u[2:]-u[1:-1], u[1:-1]-u[0:-2])/dx
    
    um = np.zeros(len(sigma)-1)
    up = np.zeros(len(sigma)-1)
    um = u[1:-2]+sigma[0:-1]*dx*0.5
    up = u[2:-1]+sigma[1:]*(-dx)*0.5
    return um, up
        
def superbee_slope(dx, u):
    sigmal = minmod(2*(u[1:-1]-u[0:-2])/dx, (u[2:]-u[1:-1])/dx)
    sigmar = minmod((u[1:-1]-u[0:-2])/dx, 2*(u[2:]-u[1:-1])/dx)
    
    sigma = np.zeros(len(sigmal))
    for i in range(len(sigmal)):
        if sigmal[i] > 0.0 and sigmar[i]> 0.0:
            sigma[i] = max(sigmal[i], sigmar[i])
        elif sigmal[i] < 0.0 and sigmar[i] < 0.0:
            sigma[i] = -max(abs(sigmal[i]), abs(sigmar[i]))
        else:
            sigma[i] = 0.0
    
    um = np.zeros(len(sigma)-1)
    up = np.zeros(len(sigma)-1)
    um = u[1:-2]+sigma[0:-1]*dx*0.5
    up = u[2:-1]+sigma[1:]*(-dx)*0.5
    return um, up
    
def mc_slope(dx, u):
    sigma = minmod2(2*(u[2:]-u[1:-1])/dx, (u[2:]-u[0:-2])/2/dx, 2*(u[1:-1]-u[0:-2])/dx)
    
    um = np.zeros(len(sigma)-1)
    up = np.zeros(len(sigma)-1)
    um = u[1:-2]+sigma[0:-1]*dx*0.5
    up = u[2:-1]+sigma[1:]*(-dx)*0.5
    return um, up
    
def euler_forward(dt, u, rhs):
    for im in range(np.size(u)):
        u[im, 0, 2:-2] = u[im, 0, 2:-2]+dt*rhs[im]
    return u
    
def rk2(k, dt, u, rhs):
    for im in range(np.size(u, 0)):
        if k == 0:
            u[im, 1, 2:-2] = u[im, 0, 2:-2]+dt*rhs[im]
        elif k == 1:
            u[im, 0, 2:-2] = 0.5*u[im, 0, 2:-2]+0.5*(u[im, 1, 2:-2]+dt*rhs[im])
    return u
    
if __name__ == "__main__":
    ## setup
    cfl = 0.4
    tend = 1

    xleft = -2
    xright = 2
    
    m = 2
    ## Jacobian Matrix, eigenvalues, eigenvectors of right and left
    A = np.array([[0, 4], [1, 0]])
    Sigma = np.array([[-2, 0], [0, 2]])
    R = np.array([[-2, 2], [1, 1]])
    L = np.array([[-1/4, 1/2], [1/4, 1/2]])
    
    iordert = 2
    # iordert 1: euler
    # iordert 2: rk2
    icase = 1
    # icase 1: "godunov_sys", "lax_friedrichs_sys", "rusanov_sys"
    # icase 3: "godunov_sys_m"
    ischeme = "lax_friedrichs_sys"
    islope = "no limiter"
    # "minmod", "superbee", "mc"

    N = 100
    N_ = N+1
    ## uniform grid
    dx = (xright-xleft)/N
    xh = np.linspace(xleft-0.5*dx, xright+0.5*dx, N_+1)
    xi = np.linspace(xleft, xright, N_)
    
    dt = cfl*dx/max(np.abs(np.diag(Sigma)))
    
    ## initialize
    uc = np.zeros((m, iordert, N_+4)) # 2 ghost points for the boundary on the left and 2 ghost points on the right.
    uc[:, 0, 2:-2] = init(m, dx, xh)
   
    ## exact solution
    uex = np.zeros((m, N_))
    uex = exactu(m, tend, xi)
    
    rhs = np.zeros((m, N_))
    ## time evolution
    time = 0
    kt = 0
    kmax = 10**5
    #while (time < tend)&(kt<kmax):
    while time < tend:
        
        if time+dt >= tend:
            dt = tend-time
        time = time+dt
        kt = kt+1
        
        if iordert== 1:
            ## call bc
            bc(uc[:, 0, :])
            
            ## spatial discretization
            if icase == 1:
                rhs = getflux(ischeme, islope, dt, dx, A, Sigma, L, R, uc[:, 0, 1:-1])
            elif icase == 3:
                rhs = getflux(ischeme, islope, dt, dx, A, Sigma, L, R, uc[:, 0, :])
            else:
                sys.exit("ERROR: this is not defined here.")
            euler_forward(dt, uc, rhs)
        elif iordert == 2:
            for k in range(iordert):
                ## call bc
                bc(uc[:, k, :])
                
                ## spatial discretization
                if icase == 1:
                    rhs = getflux(ischeme, islope, dt, dx, A, Sigma, L, R, uc[:, k, 1:-1])
                elif icase == 3:
                    rhs = getflux(ischeme, islope, dt, dx, A, Sigma, L, R, uc[:, k, :])
                else:
                    sys.exit("ERROR: this is not defined here.")
                rk2(k, dt, uc, rhs)
    uc_ = uc[:, 0, 2:-2]
    
    ## plot numerical solution
    for im in range(m):
        with open ('num_sol_{}_u{}.txt'.format(ischeme, im+1), 'w', encoding='utf-8') as file:
            for index in range(len(xi)):
                file.write("%f %f %f \n"%(xi[index], uex[im, index], uc_[im, index]))

    plt.figure()
    for im in range(m):
        plt.plot(xi, uc_[im, :], 'rD', markerfacecolor="None", markersize=5, label= '{}'.format(ischeme))
        plt.plot(xi, uex[im, :], 'k-', markerfacecolor="None", markersize=5, label='exact')
        plt.legend()
        plt.xlabel('x');plt.ylabel('u'+str(im+1))
        plt.show()
   
    

    
   


    
        
    
    


