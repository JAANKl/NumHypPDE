import numpy as np
import matplotlib.pyplot as plt


def u_exact(x, t):
    return np.sin(2 * np.pi * (x - 2 * t))


def main():
    tend = 2
    mesh_sizes = np.array([40, 80, 160, 320, 640])
    err_l1 = np.zeros(n := len(mesh_sizes))
    err_l2 = np.zeros(n)
    err_linf = np.zeros(n)
    numerical_solutions = []

    for i, N in enumerate(mesh_sizes):
        dx = 1 / N
        dt = 1 / (4 * N)  # <= 1/(2N)
        c = 2 * dt / dx
        x = np.linspace(0, 1, N)
        # Initial values:
        u = np.sin(2 * np.pi * x)
        # FTBS Matrix:
        M = np.diag((1 - c) * np.ones(N)) + np.diag(c * np.ones(N - 1), -1)
        M[0, -1] = c
        for _ in range(int(tend / dt)):
            u = M @ u
        numerical_solutions.append(u)
        err_l1[i] = np.sum(np.abs(u - u_exact(x, tend)))
        err_l2[i] = np.sqrt(np.sum((np.abs(u - u_exact(x, tend))) ** 2))
        err_linf[i] = np.max(np.abs(u - u_exact(x, tend)))

    print("L1 Error:", err_l1)
    print("L2 Error:", err_l2)
    print("Linf Error:", err_linf)

    # Plotting:
    for i, N in enumerate(mesh_sizes):
        plt.plot(np.linspace(0, 1, N), numerical_solutions[i], label=f"{N} mesh points")
    plt.plot(x := np.linspace(0, 1, mesh_sizes[-1]), u_exact(x, tend), label="exact solution")
    plt.legend()
    plt.show()
    mesh_widths = 1 / mesh_sizes
    plt.loglog(mesh_widths, err_l1, label="$L^{1}$-Error")
    plt.loglog(mesh_widths, err_l2, label="$L^{2}$-Error")
    plt.loglog(mesh_widths, err_linf, label="$L^{\infty}$-Error")
    plt.loglog(mesh_widths, 10 * mesh_widths, label="$h^{1}$")
    plt.loglog(mesh_widths, 10 * mesh_widths ** 0.5, label="$h^{0.5}$")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
