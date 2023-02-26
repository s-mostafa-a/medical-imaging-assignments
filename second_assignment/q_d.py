from scipy.signal import butter, filtfilt
import scipy.io as sio
from numpy.fft import rfftfreq, rfft
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz
from PIL import Image


def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter(N=order, Wn=[lowcut, highcut], fs=fs, btype='band')
    y = filtfilt(b, a, data)
    return y


def avali(fs, lowcut, highcut):
    # Plot the frequency response for a few different orders.
    plt.figure(1)
    plt.clf()
    for order in [4]:
        b, a = butter(N=order, Wn=[lowcut, highcut], fs=fs, btype='band')
        w, h = freqz(b, a, fs=fs, worN=2000)
        plt.plot(w / (10 ** 6), abs(h), label="order = %d" % order)
    plt.plot([0, 0.5 * fs / (10 ** 6)], [np.sqrt(0.5), np.sqrt(0.5)], '--', label='sqrt(0.5)')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Gain')
    plt.grid(True)
    plt.legend(loc='best')
    plt.show()


def dovomi(fs, lowcut, highcut):
    rf_file = sio.loadmat('./data/mousekidney.mat')['RawRF']
    N = rf_file.shape[0]
    frequencies = rfftfreq(N, 1. / fs) / (10 ** 6)
    line = 100
    d_sig = rf_file[:, line]
    one_sided_power_spectrum = abs(rfft(d_sig))
    filtered_signal = butter_bandpass_filter(d_sig, lowcut, highcut, fs, order=4)
    one_sided_filtered = abs(rfft(filtered_signal))

    plt.plot(frequencies, one_sided_power_spectrum, label='Original signal')
    plt.plot(frequencies, one_sided_filtered, label='Filtered signal')
    plt.grid()
    plt.xlabel('Frequency, mega hertz')
    plt.ylabel('Amplitude')
    plt.title('Signal power spectrum for a line')
    plt.legend(loc='upper right')
    plt.show()


def sevomi(fs, lowcut, highcut):
    rf_file = sio.loadmat('./data/mousekidney.mat')['RawRF']
    N = rf_file.shape[0]
    L = rf_file.shape[1]
    img = np.empty_like(rf_file)
    for line in range(L):
        img[:, line] = butter_bandpass_filter(rf_file[:, line], lowcut, highcut, fs, order=4)
    mine, maxe = np.min(img), np.max(img)
    img = Image.fromarray(img).resize((400, 600))
    img = (255 * (img - mine) / (maxe - mine))

    print(np.min(img), np.mean(img), np.max(img))
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    fs = 500 * 10 ** 6
    lowcut = 5 * 10 ** 6
    highcut = 50 * 10 ** 6
    # avali(fs=fs, lowcut=lowcut, highcut=highcut)
    # dovomi(fs=fs, lowcut=lowcut, highcut=highcut)
    sevomi(fs=fs, lowcut=lowcut, highcut=highcut)
