import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

if __name__ == "__main__":
    rf_file = sio.loadmat('./data/mousekidney.mat')['RawRF']

    fs = 500 * 10**6  # Sampling frequency in Hz
    c = 1540  # Speed of sound in m/s
    f0 = 7e6  # Center frequency in Hz
    N = rf_file.shape[0]  # Number of samples
    L = rf_file.shape[1]  # Number of transducer elements
    depth = 40e-3  # Depth of image in meters
    print(rf_file.shape)
