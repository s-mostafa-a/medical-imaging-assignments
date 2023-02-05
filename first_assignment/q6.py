import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage


def _find_the_previous_divisor(num, fact):
    while num % fact != 0:
        num -= 1
    return num


def _left_right_cut_point(org, new_val):
    diff = org - new_val
    left = diff // 2
    right = diff - left
    return left, right


def _block_mean(ar, fact):
    assert isinstance(fact, int), type(fact)
    sx, sy = ar.shape
    X, Y = np.ogrid[0:sx, 0:sy]
    regions = sy // fact * (X // fact) + Y // fact
    res = ndimage.mean(ar, labels=regions, index=np.arange(regions.max() + 1))
    res.shape = (sx // fact, sy // fact)
    return res


def _block_center(ar, fact):
    assert fact % 2 == 1
    center_fact = fact // 2
    sx, sy = ar.shape
    max_x, max_y = sx // fact, sy // fact
    x_s = np.arange(0, max_x) * fact + center_fact
    y_s = np.arange(0, max_y) * fact + center_fact
    x, y = np.meshgrid(x_s, y_s)
    return ar[x.flatten(), y.flatten()].reshape(max_x, max_y, order='f')


def _find_new_image_borders_for_down_sampling(img, fact):
    new_borders = []
    for i, shp in enumerate(img.shape):
        new_shp_i = _find_the_previous_divisor(shp, fact)
        left, right = _left_right_cut_point(shp, new_shp_i)
        new_borders.append((left, right))
    return new_borders


def item_a():
    # wrote _block_center function
    pass


# interpolation is better since all the pixels are having effect in the final image.
def item_b():
    img = np.array(Image.open('./data/skin.jpg').convert('L'))

    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title(f'image itself')
    plt.show()

    fact = 5
    brd = _find_new_image_borders_for_down_sampling(img, fact=fact)
    truncated_img = img[brd[0][0]:-brd[0][1], brd[1][0]:-brd[1][1]]

    bc = _block_center(truncated_img, fact=fact)
    bc_image = Image.fromarray(bc).resize((truncated_img.shape[1], truncated_img.shape[0]))
    plt.imshow(bc_image, cmap='gray')
    plt.axis('off')
    plt.title(f'downsampled using block center')
    plt.show()

    bm = _block_mean(truncated_img, fact=fact)
    bm_image = Image.fromarray(bm).resize((truncated_img.shape[1], truncated_img.shape[0]))
    plt.imshow(bm_image, cmap='gray')
    plt.axis('off')
    plt.title(f'downsampled using block mean')
    plt.show()


# effect on the resolution? need to read
def item_c():
    img = np.array(Image.open('./data/skin.jpg').convert('L'))
    fact = 5
    brd = _find_new_image_borders_for_down_sampling(img, fact=fact)

    for sig in [1, 3, 5, 7, 9]:
        filtered_img = ndimage.gaussian_filter(img, sigma=sig)
        plt.imshow(filtered_img, cmap='gray')
        plt.axis('off')
        plt.title(f'filtered image with sigma {sig}')
        plt.show()

        truncated_img = filtered_img[brd[0][0]:-brd[0][1], brd[1][0]:-brd[1][1]]
        bc = _block_center(truncated_img, fact=fact)
        bc_image = Image.fromarray(bc).resize((truncated_img.shape[1], truncated_img.shape[0]))
        plt.imshow(bc_image, cmap='gray')
        plt.axis('off')
        plt.title(f'filtered with sigma {sig} then downsampled using block center')
        plt.show()


def _up_sample(ar, fact):
    final = []
    for row in ar:
        big_row = np.ones(shape=[row.shape[0], fact], dtype=row.dtype)
        ex_row = np.expand_dims(row, axis=-1)
        big_row = big_row * ex_row
        big_row_flat = big_row.reshape((row.shape[0] * fact), order='C')
        for _ in range(fact):
            final.append(big_row_flat)
    return np.stack(final)


def item_d_pre():
    x = np.arange(9).reshape(3, 3)
    for fact in [2, 3, 4, 5]:
        print(_up_sample(x, fact))


# effect on the resolution? need to read
def item_d():
    img = np.array(Image.open('./data/skin.jpg').convert('L'))
    for fact in [2, 3, 4, 5]:
        up = _up_sample(img, fact)
        plt.imshow(up, cmap='gray')
        plt.axis('off')
        plt.title(f'upsampled image by repeating {fact} times, new shape: {up.shape}')
        plt.show()


if __name__ == "__main__":
    # item_a()
    # item_b()
    # item_c()
    # item_d_pre()
    item_d()
