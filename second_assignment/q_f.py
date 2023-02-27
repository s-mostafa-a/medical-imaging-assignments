from scipy.signal import butter, filtfilt
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
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


def envelop_and_log_compression():
    fs = 500 * 10 ** 6
    lowcut = 5 * 10 ** 6
    highcut = 50 * 10 ** 6
    rf_file = sio.loadmat('./data/mousekidney.mat')['RawRF']

    img = bandpass_filter(rf_file=rf_file, fs=fs, lowcut=lowcut, highcut=highcut)
    env_image = np.abs(hilbert(img, axis=0))

    scaled_image = env_image / env_image.max()
    log_image = 20 * np.log10(scaled_image)
    img = Image.fromarray(log_image).resize((400, 600))
    mine, maxe = np.min(img), np.max(img)
    img = (255 * (img - mine) / (maxe - mine))
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    envelop_and_log_compression()
