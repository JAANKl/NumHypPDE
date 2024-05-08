import numpy  as np
import matplotlib.pyplot as plt

data0 = np.loadtxt('num_sol_lax-friedrichs.txt')
data1 = np.loadtxt('num_sol_lax-friedrichs_m.txt')

x = data0[:, 0]
uex = data0[:, 1]

data = np.zeros((3, len(x)))
data[0,:] = data0[:, 2]
data[1,:] = data1[:, 2]

ischeme = ["Lax-Friedrichs", "Lax-Friedrichs_m"]

colors = ['r*-', 'bs-']

plt.plot(x, uex, 'k-', markersize=8, label='exact')
for i in range(2):
    plt.plot(x, data
    [i], colors[i], markerfacecolor="None", markersize=4, label='{}'.format(ischeme[i]))
plt.xlabel('x');plt.ylabel('u')
plt.legend()
plt.show()

