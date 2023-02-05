import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from PIL import Image


def _draw_magnitude_or_phase(img, name, which='magnitude', log=False):
    f = np.fft.fftshift(np.fft.fft2(img))
    if which == 'magnitude':
        if log:
            to_draw = np.log(np.abs(f))
        else:
            to_draw = np.abs(f)
    elif which == 'phase':
        to_draw = np.angle(f)
    else:
        return
    plt.imshow(to_draw, cmap='gray')
    plt.axis('off')
    plt.title(f'{name} {which}')
    plt.show()


# when there is no log, we can't see their difference. So I put log to see a better result
def item_a():
    t1 = Image.open('./data/T1.bmp').convert('L')
    t2 = Image.open('./data/T2.bmp').convert('L')

    _draw_magnitude_or_phase(t1, 'T1', which='magnitude')
    _draw_magnitude_or_phase(t1, 'T1', which='phase')
    _draw_magnitude_or_phase(t2, 'T2', which='magnitude')
    _draw_magnitude_or_phase(t2, 'T2', which='phase')

    _draw_magnitude_or_phase(t1, 'T1', which='magnitude', log=True)
    _draw_magnitude_or_phase(t2, 'T2', which='magnitude', log=True)


# why? cause their images are similar domain.
# https://dsp.stackexchange.com/questions/72674/importance-of-phase-in-fft-of-an-image
def item_b():
    t1 = Image.open('./data/T1.bmp').convert('L')
    t2 = Image.open('./data/T2.bmp').convert('L')

    f = np.fft.fft2(t1)
    f2 = np.fft.fft2(t2)

    combined_one_on_two = np.multiply(np.abs(f), np.exp(1j * np.angle(f2)))
    one_on_two_gray = np.real(np.fft.ifft2(combined_one_on_two))
    plt.imshow(one_on_two_gray, cmap='gray')
    plt.axis('off')
    plt.title(f'T1 magnitude on T2 phase')
    plt.show()

    combined_two_on_one = np.multiply(np.abs(f2), np.exp(1j * np.angle(f)))
    two_on_one_gray = np.real(np.fft.ifft2(combined_two_on_one))
    plt.imshow(two_on_one_gray, cmap='gray')
    plt.axis('off')
    plt.title(f'T2 magnitude on T1 phase')
    plt.show()


def item_c():
    for (u0, u1) in {(1, 1), (1, 10), (10, 1), (10, 10), (100, 100)}:
        x, y = np.meshgrid(np.arange(-3, 3, 0.05), np.arange(-3, 3, 0.05))
        z = np.sin(u0 * x + u1 * y)
        _draw_magnitude_or_phase(z, f'2-e u0:{u0}, u1:{u1}', which='magnitude', log=True)


# image means mris
def item_d_1():
    t1 = Image.open('./data/T1.bmp').convert('L')

    _draw_magnitude_or_phase(t1, 'T1 without filter', which='magnitude', log=True)
    for siz in [3, 9, 15]:
        h = (1 / (siz * siz)) * np.ones((siz, siz))
        res = ndimage.convolve(t1, h, mode='nearest')
        _draw_magnitude_or_phase(res, f'T1 with filter: {siz, siz}', which='magnitude', log=True)

    t2 = Image.open('./data/T2.bmp').convert('L')
    _draw_magnitude_or_phase(t2, 'T2 without filter', which='magnitude', log=True)
    for siz in [3, 9, 15]:
        h = (1 / (siz * siz)) * np.ones((siz, siz))
        res = ndimage.convolve(t2, h, mode='nearest')
        _draw_magnitude_or_phase(res, f'T2 with filter: {siz, siz}', which='magnitude', log=True)


# image means item_c
# odd!
def item_d_2():
    (u0, u1) = (100, 100)
    x, y = np.meshgrid(np.arange(-3, 3, 0.05), np.arange(-3, 3, 0.05))
    z = np.sin(u0 * x + u1 * y)
    _draw_magnitude_or_phase(z, f'2-e u0:{u0}, u1:{u1} without filter', which='magnitude',
                             log=True)
    for siz in [3, 9, 15, 25, 51]:
        h = (1 / (siz * siz)) * np.ones((siz, siz))
        res = ndimage.convolve(z, h, mode='nearest')
        _draw_magnitude_or_phase(res, f'2-e u0:{u0}, u1:{u1} with filter: {siz, siz}',
                                 which='magnitude', log=True)


if __name__ == "__main__":
    # item_a()
    # item_b()
    # item_c()
    # item_d_1()
    item_d_2()
