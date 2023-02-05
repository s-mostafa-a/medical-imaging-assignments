import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage


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
    return to_draw


# reading needed
def item_a():
    zircle = np.ones((800, 800)) * 30
    for i in range(800):
        for j in range(800):
            if (i - 400) ** 2 + (j - 400) ** 2 < 2:
                zircle[i, j] = 255
    plt.imshow(zircle, cmap='gray')
    plt.axis('off')
    plt.title(f'circle')
    plt.show()

    siz = 5
    h = (1 / (siz * siz)) * np.ones((siz, siz))
    res = ndimage.convolve(zircle, h, mode='nearest')
    plt.imshow(res, cmap='gray')
    plt.axis('off')
    plt.title(f'filtered small circle with averaging filter with size 5')
    plt.show()


# plus: cause imshow works based on contrast
def item_b():
    img = np.array(Image.open('./data/mammo_lc.jpg').convert('L'))
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title(f'original image')
    plt.show()

    plt.hist(img.ravel(), bins=list(range(256)))
    plt.title(f'original histogram')
    plt.show()

    mapped = ((img - 90) * (255 / 60))
    plt.hist(mapped.ravel(), bins=list(range(256)))
    plt.title(f'transformed histogram')
    plt.show()

    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title(f'transformed image')
    plt.show()


# search and think needed
def item_c():
    img = np.array(Image.open('./data/mammo_lc.jpg').convert('L'))
    mapped = ((img - 90) * (255 / 60))
    org_arr = _draw_magnitude_or_phase(img, f'Original image', which='magnitude', log=True)
    print(np.min(org_arr), np.mean(org_arr), np.max(org_arr))
    _draw_magnitude_or_phase(img, f'Original image', which='phase')

    tr_arr = _draw_magnitude_or_phase(mapped, f'Transformed image', which='magnitude', log=True)
    print(np.min(tr_arr), np.mean(tr_arr), np.max(tr_arr))
    _draw_magnitude_or_phase(mapped, f'Transformed image', which='phase')


if __name__ == "__main__":
    # item_a()
    # item_b()
    item_c()
