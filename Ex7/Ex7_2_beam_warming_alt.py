import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

tend = 1
N = 100
dx = 10 / N
dt = 10 / (20 * N)  # <= 1/(2N)
c = 1 * dt / dx
x = np.linspace(-5, 5, N)


def initial_values(x):
    return -np.ones(N) + 2 * (x > 0)


def u_exact(x, t):
    return initial_values(x - t)


# Initial values:
u_0 = initial_values(x)
u_BEAM = u_0
num_solutions_BEAM = [u_BEAM]
exact_solutions = [u_0]
# BEAM Matrix:
M_BEAM = np.diag((1 - 3/2*c + c**2/2) * np.ones(N)) + np.diag((2*c - c**2) * np.ones(N - 1), -1) + np.diag((-c/2 + c**2/2) * np.ones(N - 2), -2)
M_BEAM[0, 0] = 1
M_BEAM[1,0] += 3*c/2 - c**2/2
print(c)
print(M_BEAM)
t = 0
for _ in range(int(tend / dt)):
    u_BEAM = M_BEAM @ u_BEAM
    num_solutions_BEAM.append(u_BEAM)
    exact_solutions.append(u_exact(x, t))
    t += dt

# Plotting
plt.scatter(x, u_BEAM, label="BEAM", s=0.5)
plt.plot(x, u_exact(x, tend), label="exact solution")
plt.legend()
plt.show()

fig, ax = plt.subplots()

scat_BEAM = ax.scatter(x, u_BEAM, label="BEAM", s=0.5)
scat_exact = ax.scatter(x, u_exact(x, tend), label="exact solution", s=0.5)
plt.ylim((-2, 2))
ax.legend()


def update(frame):
    # for each frame, update the data stored on each artist.
    # update the BEAM scatter plot:
    data = np.stack([x, num_solutions_BEAM[frame*10]]).T
    scat_BEAM.set_offsets(data)
    # update the LW scatter plot:
    # update the exact line plot:
    data = np.stack([x, exact_solutions[frame*10]]).T
    scat_exact.set_offsets(data)
    return scat_BEAM, scat_exact


ani = animation.FuncAnimation(fig=fig, func=update, frames=50, interval=100)
plt.show()
