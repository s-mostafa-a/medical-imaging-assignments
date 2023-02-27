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


def envelope_detection_single():
    line = 100

    fs = 500 * 10 ** 6
    lowcut = 5 * 10 ** 6
    highcut = 50 * 10 ** 6
    rf_file = sio.loadmat('./data/mousekidney.mat')['RawRF']
    N = rf_file.shape[0]
    L = rf_file.shape[1]
    c = 1540  # Speed of sound in m/s

    depth = np.arange(0, N * c / (fs * 1000), c / (fs * 1000))

    # plot(t, x)
    # plot(t, m_hat)
    plt.subplot(3, 1, 1)
    plt.plot(depth, rf_file[:, line], color='silver', label='Original')
    plt.legend(loc='upper right')

    img = bandpass_filter(rf_file=rf_file, fs=fs, lowcut=lowcut, highcut=highcut)
    plt.subplot(3, 1, 2)
    plt.plot(depth, img[:, line], color='#3465a4', label='Filtered')
    plt.legend(loc='upper right')

    env = np.abs(hilbert(img[:, line]))
    plt.subplot(3, 1, 3)
    plt.plot(depth, env, color='red', label='Envelop')

    plt.legend(loc='upper right')
    print(np.mean(img[:, line] - env))

    plt.show()


def envelope_detection_img():
    fs = 500 * 10 ** 6
    lowcut = 5 * 10 ** 6
    highcut = 50 * 10 ** 6
    rf_file = sio.loadmat('./data/mousekidney.mat')['RawRF']
    N = rf_file.shape[0]
    L = rf_file.shape[1]
    c = 1540  # Speed of sound in m/s
    depth = np.arange(0, N * c / (fs * 1000), c / (fs * 1000))

    img = bandpass_filter(rf_file=rf_file, fs=fs, lowcut=lowcut, highcut=highcut)
    env = np.log(np.abs(hilbert(img, axis=0)))
    img = Image.fromarray(env).resize((400, 600))
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()



if __name__ == "__main__":
    # envelope_detection_single()
    envelope_detection_img()
