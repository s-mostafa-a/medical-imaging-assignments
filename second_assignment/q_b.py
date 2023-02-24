import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from numpy.fft import rfftfreq, rfft


def get_sample_num_for_depth(sampling_frequency, sound_velocity, d):
    return int(d * sampling_frequency / sound_velocity)


if __name__ == "__main__":
    rf_file = sio.loadmat('./data/mousekidney.mat')['RawRF']

    fs = 500 * 10 ** 6  # Sampling frequency in Hz
    c = 1540  # Speed of sound in m/s
    N = rf_file.shape[0]  # Number of samples
    L = rf_file.shape[1]  # Number of transducer elements
    lateral_dimension = 0.02  # Lateral dimension of the image in meters

    depths = {0: {'left': 0.003,
                  'right': 0.004},
              1: {'left': 0.0065,
                  'right': 0.008},
              2: {'left': 0.008,
                  'right': 0.015}
              }
    plotted_one_of_them = False
    for kd in depths.keys():
        left = get_sample_num_for_depth(sampling_frequency=fs, sound_velocity=c, d=depths[kd]['left'])
        right = get_sample_num_for_depth(sampling_frequency=fs, sound_velocity=c, d=depths[kd]['right'])
        all_spectrum = np.empty(shape=(L, int(1 + (right - left) / 2)))

        frequencies = rfftfreq(right - left, 1. / fs) / (10 ** 6)

        for line in range(L):
            d_sig = rf_file[left:right, line]
            one_sided_power_spectrum = abs(rfft(d_sig))
            # one_sided_power_spectrum = one_sided_power_spectrum / np.max(one_sided_power_spectrum)
            all_spectrum[line, :] = one_sided_power_spectrum[:]
            if not plotted_one_of_them:
                plotted_one_of_them = True
                plt.plot(frequencies, one_sided_power_spectrum)
                plt.grid()
                plt.xlabel('Frequency, mega hertz')
                plt.ylabel('Amplitude')
                plt.title('Signal power spectrum for a line')
                plt.show()

        mean_of_all = np.mean(all_spectrum, axis=0)
        mean_of_all = mean_of_all / np.max(mean_of_all)
        plt.plot(frequencies, mean_of_all)
        plt.grid()
        plt.plot(frequencies[np.argmax(mean_of_all)], mean_of_all[np.argmax(mean_of_all)], 'ok')
        plt.xlabel('Frequency, mega hertz')
        plt.ylabel('Amplitude, normalized')
        plt.title(f"Normalized averaged power spectrum for depth {depths[kd]['left'], depths[kd]['right']}")
        print(f"Depth {depths[kd]['left'], depths[kd]['right']}\tcenter: {frequencies[np.argmax(mean_of_all)]:.3f} MHz")
        plt.show()
