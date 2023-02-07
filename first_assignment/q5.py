import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.util import random_noise
from scipy.signal import wiener


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


# mean 0.2 * 255 e!
def item_a():
    img = np.array(Image.open('./data/T1_1.bmp').convert('L'))
    plt.hist(img.ravel(), bins=list(range(256)))
    plt.title(f'original histogram')
    plt.xlabel('intensity')
    plt.ylabel('number of pixels')
    plt.show()

    img_with_noise = 255 * random_noise(img, mode='gaussian', seed=None, clip=True, mean=0.2)
    plt.hist(img_with_noise.ravel(), bins=list(range(256)))
    plt.title(f'image + gaussian noise histogram')
    plt.xlabel('intensity')
    plt.ylabel('number of pixels')
    plt.show()


# https://www.mathworks.com/help/images/fourier-transform.html
def item_b():
    img = np.array(Image.open('./data/T1_1.bmp').convert('L'))

    _draw_magnitude_or_phase(img, f'T1_1 without noise', which='magnitude', log=True)
    _draw_magnitude_or_phase(img, f'T1_1 without noise', which='phase')

    for mean, var in {(0.1, 0.01), (0.1, 0.1), (0.5, 0.01), (0.5, 1)}:
        img_with_noise = 255 * random_noise(img, mode='gaussian', seed=None, clip=True, mean=mean,
                                            var=var)
        _draw_magnitude_or_phase(img_with_noise, f'T1_1 with gaussian mean: {mean}, var: {var}',
                                 which='magnitude', log=True)
        _draw_magnitude_or_phase(img_with_noise, f'T1_1 with gaussian mean: {mean}, var: {var}',
                                 which='phase')


def item_c():
    img = np.array(Image.open('./data/T1_1.bmp').convert('L'))

    for mean, var in {(0.2, 0.01)}:  # , (0.5, 1)}:
        img_with_noise = 255 * random_noise(img, mode='gaussian', seed=None, clip=True, mean=mean,
                                            var=var)
        plt.imshow(img_with_noise, cmap='gray')
        plt.axis('off')
        plt.title(f'T1_1 with gaussian m,v: {mean, var} image')
        plt.show()
        _draw_magnitude_or_phase(img_with_noise, f'T1_1 with gaussian mean: {mean}',
                                 which='magnitude', log=True)
        _draw_magnitude_or_phase(img_with_noise, f'T1_1 with gaussian mean: {mean}',
                                 which='phase')

        denoised = wiener(img_with_noise)
        plt.imshow(denoised, cmap='gray')
        plt.axis('off')
        plt.title(f'T1_1 with gaussian mean: {mean} denoised image')
        plt.show()
        _draw_magnitude_or_phase(denoised, f'T1_1 with gaussian mean: {mean} denoised',
                                 which='magnitude', log=True)
        _draw_magnitude_or_phase(denoised, f'T1_1 with gaussian mean: {mean} denoised',
                                 which='phase')


if __name__ == "__main__":
    # item_a()
    # item_b()
    item_c()
