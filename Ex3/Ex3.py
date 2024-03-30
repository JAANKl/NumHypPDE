import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sc

y = np.linspace(0, 2, 1000)
t = 0.5 / np.pi


def rhs(x, t, y):
    return np.sin(np.pi * x) * t + 0.5 * t + x - y


x = sc.newton(rhs, x0=y - t / 2, args=(t, y))
plt.plot(y, np.sin(np.pi * x) + 0.5)
plt.xlabel("x")
plt.ylabel("u(x,0.5/pi)")
plt.show()
t = 1.5 / np.pi
x = sc.newton(rhs, x0=y - t / 2, args=(t, y))
plt.plot(y, np.sin(np.pi * x) + 0.5)
plt.xlabel("x")
plt.ylabel("u(x,1.5/pi)")
plt.show()
