import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from PIL import Image


def dirac(inp):
    return np.where(inp == 0, np.inf, 0)


def discrete_dirac(inp, n=0):
    return np.where(inp == n, 1, 0)


def shift_n(arr, shift):
    new_size = arr.shape[0] + shift
    new_arr = np.zeros(shape=(new_size))
    if shift > 0:
        new_arr[shift:] = arr
    else:
        new_arr = arr[-shift:]
    return new_arr


# Shift property:
# https://eng.libretexts.org/Bookshelves/Electrical_Engineering/Signal_Processing_and_Modeling/Signals_and_Systems_(Baraniuk_et_al.)/04%3A_Time_Domain_Analysis_of_Discrete_Time_Systems/4.04%3A_Properties_of_Discrete_Time_Convolution
def item_a():
    x_n = np.zeros(shape=(6))
    x_n[0: 6] = np.array([0, 1, 2, 3, 4, 5])
    y_n = np.convolve(x_n, x_n)
    plt.stem(range(len(y_n)), y_n)
    plt.xlabel('n')
    plt.ylabel('y[n]')
    plt.show()


def _get_f_n(n):
    k = 2
    s = np.sin((2 * np.pi * k / 3) * n)
    return s


def item_b_1():
    n = np.arange(-10, 11, 1)
    n_2 = np.arange(-20, 21, 1)
    d = discrete_dirac(n)
    s = _get_f_n(n)

    plt.subplot(2, 1, 1)
    plt.xticks(n)
    plt.stem(n, s)
    plt.xlabel('n')
    plt.ylabel('sin[w_2.n]')

    con = np.convolve(s, d)
    plt.subplot(2, 1, 2)
    plt.xticks(n_2)
    plt.stem(n_2, con)
    plt.xlabel('n')
    plt.ylabel('sin[w_2.n] * dirac[n]')
    plt.show()


# tozih: chera shift midim, chon conv behemoon serfan result mide
def item_b_2():
    n = np.arange(-10, 11, 1)
    n_2 = np.arange(-20, 21, 1)
    d = discrete_dirac(n, n=3)
    s = _get_f_n(n)

    plt.subplot(2, 1, 1)
    plt.xticks(n - (- 3))
    plt.stem(n - (- 3), s)
    plt.xlabel('n')
    plt.ylabel('sin[w_2.(n - (-3))]')

    con = np.convolve(s, d)
    plt.subplot(2, 1, 2)
    plt.xticks(n_2)
    plt.stem(n_2, con)
    plt.xlabel('n')
    plt.ylabel('sin[w_2.n] * dirac[n - (-3)]')
    plt.show()


# search for why
def item_c():
    img = Image.open('./data/T1.bmp').convert('L')
    for siz in [3, 5, 9, 11, 13, 15]:
        h = (1 / (siz * siz)) * np.ones((siz, siz))
        res = ndimage.convolve(img, h, mode='nearest')
        plt.imshow(res, cmap='gray')
        plt.axis('off')
        plt.title(f'{(siz, siz)}')
        plt.show()


def item_d():
    img = Image.open('./data/T1.bmp').convert('L')
    h = np.array([[1 / 10, 5 / 10, 1 / 10],
                  [0, 0, 0],
                  [-1 / 10, -5 / 10, -1 / 10]])
    res = ndimage.convolve(img, h, mode='nearest')
    plt.imshow(res + 128, cmap='gray')
    plt.axis('off')
    plt.title(f'vertical with filter')
    plt.show()


def item_d_other():
    img = Image.open('./data/T1.bmp').convert('L')
    sobel_filter = np.array([[-1 / 10, 0, 1 / 10],
                             [-5 / 10, 0, 5 / 10],
                             [-1 / 10, 0, 1 / 10]])
    sobel_image = ndimage.convolve(img, sobel_filter, mode='nearest')
    plt.imshow(sobel_image + 128, cmap='gray')
    plt.axis('off')
    plt.title('sobel image')
    plt.show()


def item_e():
    img = Image.open('./data/T1.bmp').convert('L')
    h = np.array([[-1 / 10, 0, 1 / 10],
                  [-5 / 10, 0, 5 / 10],
                  [-1 / 10, 0, 1 / 10]])
    res = ndimage.convolve(img, h, mode='nearest')
    plt.imshow(res + 128, cmap='gray')
    plt.axis('off')
    plt.title(f'horizontal')
    plt.show()


if __name__ == "__main__":
    # item_a()
    # item_b_1()
    # item_b_2()
    # item_c()
    # item_d()
    item_e()
