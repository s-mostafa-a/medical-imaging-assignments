import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from PIL import Image


def _draw_magnitude_or_phase(img, name, which='magnitude'):
    f = np.fft.fft2(img)
    if which == 'magnitude':
        to_draw = np.abs(f)
    elif which == 'phase':
        to_draw = np.angle(f)
    else:
        return
    plt.imshow(to_draw, cmap='gray')
    plt.axis('off')
    plt.title(f'{name} {which}')
    plt.show()


def item_a():
    t1 = Image.open('./data/T1.bmp').convert('L')
    t2 = Image.open('./data/T2.bmp').convert('L')

    _draw_magnitude_or_phase(t1, 'T1', which='magnitude')
    _draw_magnitude_or_phase(t1, 'T1', which='phase')
    _draw_magnitude_or_phase(t2, 'T2', which='magnitude')
    _draw_magnitude_or_phase(t2, 'T2', which='phase')


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


if __name__ == "__main__":
    item_a()
    # item_b()
    # item_c()
    # item_d()
