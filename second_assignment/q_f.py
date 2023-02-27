from scipy.signal import butter, filtfilt
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import hilbert
import matplotlib.ticker as ticker


def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter(N=order, Wn=[lowcut, highcut], fs=fs, btype='band')
    y = filtfilt(b, a, data)
    return y


def bandpass_filter(rf_file, fs, lowcut, highcut):
    L = rf_file.shape[1]
    img = np.empty_like(rf_file)
    for line in range(L):
        img[:, line] = butter_bandpass_filter(rf_file[:, line], lowcut, highcut, fs, order=4)
    return img


def _call_plot_stuff_i():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    scale_x = 20
    make_it_millimeters = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x / scale_x))
    ax.xaxis.set_major_formatter(make_it_millimeters)
    ax.yaxis.set_major_formatter(make_it_millimeters)
    plt.xlabel('Lateral, millimeters')
    plt.ylabel('Depth, millimeters')


def item_i(rf, fs, lowcut, highcut):
    img = bandpass_filter(rf_file=rf, fs=fs, lowcut=lowcut, highcut=highcut)
    analytical = img + 1.j * hilbert(img, axis=0)
    env = np.abs(analytical)
    scaled_image = env / env.max()
    log_image = 20 * np.log10(scaled_image)

    img = Image.fromarray(log_image).resize((400, 600))
    mine, maxe = np.min(img), np.max(img)
    _call_plot_stuff_i()
    plt.title('Raw resulting image')
    plt.imshow(img, cmap='gray')
    plt.show()

    print(np.min(img), np.mean(img), np.max(img))
    img = (255 * (img - mine) / (maxe - mine)).astype(int)
    print(np.min(img), np.mean(img), np.max(img))
    _call_plot_stuff_i()
    plt.title('Normalized to 0-255')
    plt.imshow(img, cmap='gray')
    plt.show()


if __name__ == "__main__":
    # settings

    fs = 500 * 10 ** 6
    lowcut = 5 * 10 ** 6
    highcut = 50 * 10 ** 6
    rf_file = sio.loadmat('./data/mousekidney.mat')['RawRF']

    # item calls

    item_i(rf=rf_file, fs=fs, lowcut=lowcut, highcut=highcut)
