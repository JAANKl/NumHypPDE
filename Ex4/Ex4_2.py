import numpy as np
import matplotlib.pyplot as plt


def u_exact(x, t):
    return np.sin(2 * np.pi * (x - 2 * t))


def main():
    tend = 1
    mesh_sizes = np.array([40, 80, 160, 320, 640])
    err_l1 = np.zeros(n := len(mesh_sizes))
    err_l2 = np.zeros(n)
    err_linf = np.zeros(n)
    numerical_solutions = []

    for i, N in enumerate(mesh_sizes):
        dx = 1 / N
        dt = 1 / (4 * N)  # <= 1/(2N) for CFL condition
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
        err_l1[i] = np.sum(np.abs(u - u_exact(x, tend))) * dx
        err_l2[i] = np.sqrt(np.sum((np.abs(u - u_exact(x, tend))) ** 2) * dx)
        err_linf[i] = np.max(np.abs(u - u_exact(x, tend)))

    # Plotting:
    for i, N in enumerate(mesh_sizes):
        plt.scatter(np.linspace(0, 1, N), numerical_solutions[i], label=f"{N} mesh points", s=0.2)

    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.plot(x := np.linspace(0, 1, mesh_sizes[-1]), u_exact(x, tend), label="exact solution")
    plt.legend()
    plt.show()
    mesh_widths = 1 / mesh_sizes
    plt.loglog(mesh_widths, err_l1, label="$L^{1}$-Error")
    plt.loglog(mesh_widths, err_l2, label="$L^{2}$-Error")
    plt.loglog(mesh_widths, err_linf, label="$L^{\infty}$-Error")
    plt.loglog(mesh_widths, 10 * mesh_widths, label="$h^{1}$ (for comparison)")
    plt.loglog(mesh_widths, 10 * mesh_widths ** 0.5, label="$h^{0.5}$ (for comparison)")
    plt.xlabel("mesh width h")
    plt.ylabel("error")
    plt.legend()
    plt.show()

    print("L1 average convergence rate:", np.polyfit(np.log(mesh_widths), np.log(err_l1), 1)[0])
    print("L2 average convergence rate:", np.polyfit(np.log(mesh_widths), np.log(err_l2), 1)[0])
    print("Linf average convergence rate:", np.polyfit(np.log(mesh_widths), np.log(err_linf), 1)[0])

    print(f"N={mesh_sizes[0]}")
    print(f"L1 Error at N={mesh_sizes[0]}: {err_l1[0]}")
    print(f"L2 Error  at N={mesh_sizes[0]}: {err_l2[0]}")

    print(f"Linf Error at N={mesh_sizes[0]}: {err_linf[0]}")

    for i, N in enumerate(mesh_sizes[1:]):
        print(f"N={N}")
        print(f"L1 Error at N={N}:", err_l1[i + 1])
        print(f"L2 Error  at N={N}:", err_l2[i + 1])
        print(f"Linf Error at N={N}:", err_linf[i + 1])
        print(f"L1 local convergence rate at N={N} :",
              np.polyfit(np.log(mesh_widths[i:i + 2]), np.log(err_l1[i:i + 2]), 1)[0])
        print(f"L2 local convergence rate  at N={N}:",
              np.polyfit(np.log(mesh_widths[i:i + 2]), np.log(err_l2[i:i + 2]), 1)[0])
        print(f"Linf local  convergence rate at N={N}:",
              np.polyfit(np.log(mesh_widths[i:i + 2]), np.log(err_linf[i:i + 2]), 1)[0])


if __name__ == "__main__":
    main()
