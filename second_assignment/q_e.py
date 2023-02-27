from scipy.signal import butter, filtfilt
import scipy.io as sio
from numpy.fft import rfftfreq, rfft
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz
from PIL import Image
from scipy.signal import hilbert


def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter(N=order, Wn=[lowcut, highcut], fs=fs, btype='band')
    y = filtfilt(b, a, data)
    return y


def bandpass_filter(rf_file, fs, lowcut, highcut):
    N = rf_file.shape[0]
    L = rf_file.shape[1]
    img = np.empty_like(rf_file)
    for line in range(L):
        img[:, line] = butter_bandpass_filter(rf_file[:, line], lowcut, highcut, fs, order=4)
    return img
    # mine, maxe = np.min(img), np.max(img)
    #
    # env = np.abs(hilbert(img, axis=0))
    #
    # img = Image.fromarray(img).resize((400, 600))
    # img = (255 * (img - mine) / (maxe - mine))
    #
    # print(np.min(img), np.mean(img), np.max(img))
    # plt.imshow(img, cmap='gray')
    # plt.axis('off')
    # plt.show()


def envelope_detection():
    fs = 500 * 10 ** 6
    lowcut = 5 * 10 ** 6
    highcut = 50 * 10 ** 6
    rf_file = sio.loadmat('./data/mousekidney.mat')['RawRF']
    N = rf_file.shape[0]
    L = rf_file.shape[1]
    img = bandpass_filter(rf_file=rf_file, fs=fs, lowcut=lowcut, highcut=highcut)
    env = hilbert(img[:, 100])
    t = np.arange(0, N / fs, 1 / fs)
    # plot(t, x)
    # plot(t, m_hat)
    plt.plot(t, img[:, 100], color='silver', label='Original')
    plt.plot(t, env, color='#3465a4', label='Envelop')
    plt.legend(loc='upper right')
    plt.show()
    print(env.shape)


if __name__ == "__main__":
    envelope_detection()
