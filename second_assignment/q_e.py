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


def _call_plot_stuff_ii(ax):
    scale_x = 10 ** 6
    make_it_millimeters = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x * scale_x))
    ax.xaxis.set_major_formatter(make_it_millimeters)
    plt.grid()
    plt.xlabel('Depth (millimeters)')
    plt.ylabel('Amplitude')
    plt.legend(loc='upper right')


def item_ii(rf, fs, lowcut, highcut, line=100):
    N = rf.shape[0]
    c = 1540
    depth = np.arange(0, N * c / (fs * 1000), c / (fs * 1000))
    ax = plt.subplot(3, 1, 1)
    plt.plot(depth, rf[:, line], color='silver', label='Original')
    _call_plot_stuff_ii(ax)

    img = bandpass_filter(rf_file=rf, fs=fs, lowcut=lowcut, highcut=highcut)
    ax = plt.subplot(3, 1, 2)
    plt.plot(depth, img[:, line], color='#3465a4', label='Filtered')
    _call_plot_stuff_ii(ax)

    env = np.abs(hilbert(img[:, line]))
    ax = plt.subplot(3, 1, 3)
    plt.plot(depth, env, color='red', label='Envelop')
    _call_plot_stuff_ii(ax)
    plt.show()


def _call_plot_stuff_iii():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    scale_x = 20
    make_it_millimeters = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x / scale_x))
    ax.xaxis.set_major_formatter(make_it_millimeters)
    ax.yaxis.set_major_formatter(make_it_millimeters)
    plt.xlabel('Lateral, millimeters')
    plt.ylabel('Depth, millimeters')


def item_iii(rf, fs, lowcut, highcut):
    img = bandpass_filter(rf_file=rf, fs=fs, lowcut=lowcut, highcut=highcut)
    env = np.abs(hilbert(img, axis=0))
    img = Image.fromarray(env).resize((400, 600))
    mine, maxe = np.min(img), np.max(img)
    _call_plot_stuff_iii()
    plt.title('Raw resulting image')
    plt.imshow(img, cmap='gray')
    plt.show()

    print(np.min(img), np.mean(img), np.max(img))
    img = (255 * (img - mine) / (maxe - mine)).astype(int)
    print(np.min(img), np.mean(img), np.max(img))
    _call_plot_stuff_iii()
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

    # item_ii(rf=rf_file, fs=fs, lowcut=lowcut, highcut=highcut)
    item_iii(rf=rf_file, fs=fs, lowcut=lowcut, highcut=highcut)
