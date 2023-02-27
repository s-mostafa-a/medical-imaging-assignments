import scipy.io as sio

if __name__ == "__main__":
    rf_file = sio.loadmat('./data/mousekidney.mat')['RawRF']

    fs = 500 * 10 ** 6  # Sampling frequency in Hz
    c = 1540  # Speed of sound in m/s
    N = rf_file.shape[0]  # Number of samples
    L = rf_file.shape[1]  # Number of transducer elements
    lateral_dimension = 0.02  # Lateral dimension of the image in meters

    # Each line has 10000 samples and the sampling frequency is 500 * 10^6. So, sampling time is:
    sampling_time = N / fs
    # And we have to see in that time, how much the sound signal can travel?
    max_depth = sampling_time * c
    print(f'Depth: {max_depth} meters')
