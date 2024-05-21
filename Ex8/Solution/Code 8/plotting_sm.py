import numpy  as np
import matplotlib.pyplot as plt

data0 = np.loadtxt('num_sol_godunov_m_minmod_sm.txt')
data1 = np.loadtxt('num_sol_godunov_m_superbee_sm.txt')
data2 = np.loadtxt('num_sol_godunov_m_mc_sm.txt')
data3 = np.loadtxt('num_sol_godunov_m_vanleer_sm.txt')

x = data0[:, 0]
uex = data0[:, 1]

data = np.zeros((4, len(x)))
data[0,:] = data0[:, 2]
data[1,:] = data1[:, 2]
data[2,:] = data2[:, 2]
data[3,:] = data3[:, 2]

ischeme = ["Minmod", "Superbee", "MC", "VanLeer"]

colors = ['r*-', 'bs-', 'g^-', 'kD-']

plt.plot(x, uex, 'k-', markersize=8, label='exact')
for i in range(4):
    plt.plot(x, data
    [i], colors[i], markerfacecolor="None", markersize=6, label='{}'.format(ischeme[i]))
plt.xlabel('x');plt.ylabel('u')
plt.legend()
plt.show()

