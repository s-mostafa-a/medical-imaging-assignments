import numpy as np
import matplotlib.pyplot as plt


def item_a():
    x = np.arange(-np.pi, np.pi, 0.01)
    y = np.sin(x)
    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('sin(x)')
    plt.show()


def item_b():
    n = np.arange(-10, 11, 1)
    h = np.heaviside(n, 0.5)
    plt.xticks(n)
    plt.stem(n, h)
    plt.xlabel('n')
    plt.ylabel('heaviside(n)')
    plt.show()


def item_c():
    n = np.arange(-10, 11, 1)
    k = 2
    s = np.sin((2 * np.pi * k / 3) * n)
    plt.xticks(n)
    plt.stem(n, s)
    plt.xlabel('n')
    plt.ylabel('sin(w_k*n)')
    plt.show()


def item_d():
    n = np.arange(-10, 11, 1)
    for i, k in enumerate([1, 2, 4]):
        s = np.sin((2 * np.pi * k / 3) * n)
        plt.subplot(3, 1, i + 1)
        plt.xticks(n)
        plt.stem(n, s)
        plt.xlabel('n')
        plt.ylabel(f'sin(w_{k}*n)')

    plt.show()


if __name__ == "__main__":
    # item_a()
    # item_b()
    item_c()
    # item_d()
