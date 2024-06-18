import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sc

exercise_name = "Ex_2.3a"
save_plots = True
compute_rates = True

tend = 0.5/np.pi
x_left = 0
x_right = 2
cfl = 0.4  # = dt/dx
which_bc = "periodic"
which_schemes =  ["lax_wendroff_advection", "upwind"]
# central, upwind, lax_wendroff_advection, lax_wendroff_general

x = np.linspace(x_left, x_right, 1000)


def f(x_0, t, x):
    return x_0 + (np.sin(np.pi * x_0) +1/2)*t -x
        
def f_prime(x_0, t, x):
    return 1+ np.pi*np.cos(np.pi*x_0)*t

init_val = x


x_0 = sc.newton(f, x0=init_val,fprime=f_prime, args=(t, x),tol=1e-5, maxiter=100)
plt.plot(x,np.sin(np.pi*x_0)+0.5, label=r"$t=\frac{0.5}{\pi}$")
plt.xlabel("x")
plt.ylabel("u(x,0.5/pi)")
plt.legend()
plt.show()

t = 1.5/np.pi
x_s = 1+1/2*t

no_prob = x<=0.2
prob_is_left = np.logical_and(x <= x_s, x>1.0)
prob_is_right = x > x_s
init_val = 0.5*prob_is_left+1.5*prob_is_right+(0)*no_prob
x_0 = sc.newton(f, x0=init_val,fprime=f_prime, args=(t, x),tol=1e-5, maxiter=100)

    

plt.plot(x,np.sin(np.pi*x_0)+0.5, label=r"$t=\frac{1.5}{\pi}$")
plt.xlabel("x")
plt.legend()
plt.ylabel("u(x,1.5/pi)")
plt.show()

def u_0(x_0):
    return np.sin(np.pi*x_0)+1/2

fig, ax = plt.subplots(1, 1)
t = np.linspace(0, 1, 100)
for x_0 in np.linspace(0, 2, 50):
    ax.plot(u_0(x_0)*t+x_0, t)
ax.set_xlabel("x")
ax.set_ylabel("t")

ax.hlines(0.5/np.pi, 0, 2.5, linestyles='dotted', label=r"$t=\frac{0.5}{\pi}$")
ax.hlines(1.5/np.pi, 0, 2.5, linestyles='dashed', label=r"$t=\frac{1.5}{\pi}$")
ax.legend()
fig.show()