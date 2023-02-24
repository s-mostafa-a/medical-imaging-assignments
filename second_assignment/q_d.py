from scipy.signal import butter, lfilter, freqz
from numpy.random import randn
from scipy import signal
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from numpy.fft import rfftfreq, rfft


def item_i():
    pass


def item_ii():
    pass


def item_iii():
    pass


if __name__ == "__main__":
    rf_file = sio.loadmat('./data/mousekidney.mat')['RawRF']

    fs = 500 * 10 ** 6  # Sampling frequency in Hz
    c = 1540  # Speed of sound in m/s
    N = rf_file.shape[0]  # Number of samples
    L = rf_file.shape[1]  # Number of transducer elements
    lateral_dimension = 0.02  # Lateral dimension of the image in meters

    rf_file = sio.loadmat('./data/mousekidney.mat')['RawRF']
    frequencies = rfftfreq(N, 1. / fs) / (10 ** 6)
    print(frequencies.shape)
    line = 0
    # for line in range(L):
    d_sig = rf_file[:, line]
    one_sided_power_spectrum = abs(rfft(d_sig))
    plotted_one_of_them = True
    plt.plot(frequencies, one_sided_power_spectrum)
    plt.grid()
    plt.xlabel('Frequency, mega hertz')
    plt.ylabel('Amplitude')
    plt.title('Signal power spectrum for a line')
    plt.show()
    # break

    b, a = butter(N=4, Wn=0.03, analog=False)
    print(b, a)
    # Show that frequency response is the same

    # Applies filter forward and backward in time
    imp_ff = signal.filtfilt(b, a, d_sig)

    # Applies filter forward in time twice (for same frequency response)

    plt.subplot(2, 1, 1)
    plt.semilogx(20 * np.log10(np.abs(rfft(imp_ff))))
    plt.ylim(-100, 20)
    plt.grid(True, which='both')
    plt.title('filtfilt')

    sig_ff = signal.filtfilt(b, a, d_sig)
    plt.subplot(2, 1, 2)
    plt.plot(d_sig, color='silver', label='Original')
    plt.plot(sig_ff, color='#3465a4', label='filtfilt')
    plt.grid(True, which='both')
    plt.legend(loc="best")
    plt.show()
