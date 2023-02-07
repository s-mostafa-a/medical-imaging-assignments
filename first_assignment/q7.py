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

    m_f = (np.max(zircle) - np.min(zircle)) / (np.max(zircle) + np.min(zircle))
    print('m_f: ', m_f)

    zircle[0, 0] = 0
    plt.imshow(zircle, cmap='gray')
    plt.axis('off')
    plt.title(f'circle')
    plt.show()
    zircle[0, 0] = 30
    siz = 5
    h = (1 / (siz * siz)) * np.ones((siz, siz))
    res = get_filtered_image(zircle, h)
    m_g = (np.max(res) - np.min(res)) / (np.max(res) + np.min(res))
    print('m_g: ', m_g)
    res[0, 0] = 0
    res[-1, -1] = 255
    plt.imshow(res, cmap='gray')
    plt.axis('off')
    plt.title(f'filtered small circle with averaging filter with size 5')
    plt.show()
    print('ratio:', m_g / m_f)


# plus: cause imshow works based on contrast
def item_b():
    img = np.array(Image.open('./data/mammo_lc.jpg').convert('L'))
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title(f'original image')
    plt.show()

    plt.hist(img.ravel(), bins=list(range(256)))
    plt.title(f'original histogram')
    plt.xlabel('intensity')
    plt.ylabel('number of pixels')
    plt.show()

    mapped = ((img - 90) * (255 / 60))
    plt.hist(mapped.ravel(), bins=list(range(256)))
    plt.title(f'transformed histogram')
    plt.xlabel('intensity')
    plt.ylabel('number of pixels')
    plt.show()

    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title(f'transformed image')
    plt.show()


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
    item_a()
    # item_b()
    # item_c()
