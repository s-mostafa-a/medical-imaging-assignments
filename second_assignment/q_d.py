from scipy.signal import butter, filtfilt
import scipy.io as sio
from numpy.fft import rfftfreq, rfft
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz
from PIL import Image
import matplotlib.ticker as ticker


def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter(N=order, Wn=[lowcut, highcut], fs=fs, btype='band')
    y = filtfilt(b, a, data)
    return y


def item_i(fs, lowcut, highcut):
    plt.figure(1)
    plt.clf()
    order = 4
    b, a = butter(N=order, Wn=[lowcut, highcut], fs=fs, btype='band')
    w, h = freqz(b, a, fs=fs, worN=2000)
    plt.plot(w / (10 ** 6), abs(h), label="order = %d" % order)
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Gain')
    plt.grid(True)
    plt.legend(loc='best')
    plt.show()


def _call_plot_stuff_ii():
    plt.grid()
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Amplitude')
    plt.legend(loc='upper right')


def item_ii(rf, fs, lowcut, highcut, line=100):
    N = rf.shape[0]
    frequencies = rfftfreq(N, 1. / fs) / (10 ** 6)
    d_sig = rf[:, line]
    one_sided_power_spectrum = abs(rfft(d_sig))
    filtered_signal = butter_bandpass_filter(d_sig, lowcut, highcut, fs, order=4)
    one_sided_filtered = abs(rfft(filtered_signal))
    plt.subplot(3, 1, 1)
    plt.plot(frequencies, one_sided_power_spectrum, label='Original signal')
    plt.title('Signal power spectrum for a line')
    _call_plot_stuff_ii()

    plt.subplot(3, 1, 2)
    plt.plot(frequencies, one_sided_filtered, label='Filtered signal')
    _call_plot_stuff_ii()

    plt.subplot(3, 1, 3)
    plt.plot(frequencies, one_sided_power_spectrum, label='Original signal')
    plt.plot(frequencies, one_sided_filtered, label='Filtered signal')
    _call_plot_stuff_ii()

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
    L = rf.shape[1]
    img = np.empty_like(rf)
    for line in range(L):
        img[:, line] = butter_bandpass_filter(rf[:, line], lowcut, highcut, fs, order=4)
    img = Image.fromarray(img).resize((400, 600))
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

    rf_file = sio.loadmat('./data/mousekidney.mat')['RawRF']
    fs = 500 * 10 ** 6
    lowcut = 5 * 10 ** 6
    highcut = 50 * 10 ** 6

    # item calls

    # item_i(fs=fs, lowcut=lowcut, highcut=highcut)
    # item_ii(rf=rf_file, fs=fs, lowcut=lowcut, highcut=highcut)
    item_iii(rf=rf_file, fs=fs, lowcut=lowcut, highcut=highcut)
