import numpy as np
import matplotlib.pyplot as plt


def dirac(inp):
    return np.where(inp == 0, np.inf, 0)


def discrete_dirac(inp):
    return np.where(inp == 0, 1, 0)


def rect(x):
    return np.where(abs(x) < 0.5, 1, 0) + np.where(abs(x) == 0.5, 0.5, 0)


def item_a():
    print(dirac(np.array(list(range(-2, 3)))))


def item_b():
    n = np.arange(-10, 11, 1)
    d = discrete_dirac(n)
    plt.xticks(n)
    plt.stem(n, d)
    plt.xlabel('n')
    plt.ylabel('dirac(n)')
    plt.show()


def item_c():
    x = np.arange(-5, 5.1, 0.1)
    y = rect(x)
    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('rect(x)')
    plt.show()


def item_d():
    x = np.arange(-5 * np.pi, 5 * np.pi, 0.1)
    y = np.sinc(np.pi * x / 2) * 2 / np.pi
    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('function(x)')
    plt.show()


def item_e():
    x, y = np.meshgrid(np.arange(-3, 3, 0.05), np.arange(-3, 3, 0.05))
    for (u0, u1) in {(0, 0), (1, 0), (0, 1), (1, 1), (10, 10)}:
        z = np.sin(u0 * x + u1 * y)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x, y, z)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title(f'u0: {u0}, u1: {u1}')
        plt.show()


if __name__ == "__main__":
    # item_a()
    # item_b()
    # item_c()
    # item_d()
    item_e()
