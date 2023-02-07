import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def _assertion_checks(image, filter, mode):
    assert len(image.shape) == 2, "The image must be a 2d image!"
    assert len(filter.shape) == 2, "The filter must be 2d!"
    assert mode in ("valid", "same"), "The mode parameter can only be `valid` or `same`"
    assert filter.shape[0] % 2 == 1 and filter.shape[
        1] % 2 == 1, "filter shape must be a tuple of odd numbers!"


def _get_paddings_and_new_image(image, filter_shape, mode):
    padding_i = filter_shape[0] // 2
    padding_j = filter_shape[1] // 2
    if mode == "same":
        image = np.pad(image, [(padding_i,), (padding_j,)])
    return padding_i, padding_j, image


def _filter_using_broadcast(image: np.array, filter: np.array, mode: str = 'same'):
    _assertion_checks(image=image, filter=filter, mode=mode)
    padding_i, padding_j, image = _get_paddings_and_new_image(image=image,
                                                              filter_shape=filter.shape, mode=mode)
    shape = tuple([image.shape[0] - 2 * padding_i, image.shape[1] - 2 * padding_j] +
                  list(filter.shape))
    multi_dim_image = np.empty(shape=shape, dtype=image.dtype)
    for i in range(padding_i, image.shape[0] - padding_i):
        for j in range(padding_j, image.shape[1] - padding_j):
            multi_dim_image[i - padding_i, j - padding_j] = image[
                                                            i - padding_i:i + padding_i + 1,
                                                            j - padding_j:j + padding_j + 1]

    expanded_filter = np.expand_dims(np.expand_dims(filter, axis=0), axis=0)

    final_image = np.sum((multi_dim_image * expanded_filter), axis=(2, 3))
    return final_image


def get_filtered_image(image, filter):
    result_image = np.empty(shape=image.shape, dtype=int)
    if len(image.shape) == 3:
        for ch in range(3):
            result_image[:, :, ch] = _filter_using_broadcast(image=image[:, :, ch],
                                                             filter=filter, mode='same')
    elif len(image.shape) == 2:
        result_image = _filter_using_broadcast(image=image, filter=filter, mode='same')
    else:
        raise Exception
    return result_image


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


def _get_f_n(n):
    k = 2
    s = np.sin((2 * np.pi * k / 3) * n)
    return s


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


def item_c():
    img = np.array(Image.open('./data/T1.bmp').convert('L'))
    for siz in [3, 5, 9, 11, 13, 15]:
        h = (1 / (siz * siz)) * np.ones((siz, siz))
        res = get_filtered_image(img, h)
        plt.imshow(res, cmap='gray')
        plt.axis('off')
        plt.title(f'{(siz, siz)}')
        plt.show()


def item_d():
    img = np.array(Image.open('./data/T1.bmp').convert('L'))
    h = np.array([[1, 0, -1],
                  [2, 0, -2],
                  [1, 0, -1]])
    res = np.abs(get_filtered_image(image=img, filter=h))
    plt.imshow(res, cmap='gray')
    plt.axis('off')
    plt.title(f'vertical edges enhanced with the sobel filter')
    plt.show()


def item_e():
    img = np.array(Image.open('./data/T1.bmp').convert('L'))
    h = np.array([[1, 2, 1],
                  [0, 0, 0],
                  [-1, -2, -1]])
    res = np.abs(get_filtered_image(image=img, filter=h))
    plt.imshow(res, cmap='gray')
    plt.axis('off')
    plt.title(f'horizontal edges enhanced with the sobel filter')
    plt.show()


if __name__ == "__main__":
    # item_a()
    # item_b_1()
    # item_b_2()
    item_c()
    # item_d()
    # item_e()
