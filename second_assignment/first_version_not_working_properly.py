import numpy as np
from scipy.signal import hilbert
import scipy.io as sio
import matplotlib.pyplot as plt

data = sio.loadmat('./data/mousekidney.mat')
rf = data['RawRF']
fs = 500 * 10 ** 6  # Sampling frequency in Hz
c = 1540  # Speed of sound in m/s
f0 = 7e6  # Center frequency in Hz
N = rf.shape[0]  # Number of samples
L = rf.shape[1]  # Number of transducer elements
depth = 0.0308  # Depth of image in meters
t = np.arange(0, N / fs, 1 / fs)
z = np.arange(0, depth, depth / L)
env = np.abs(hilbert(rf, axis=0))
log_env = 20 * np.log10(env / np.max(env))
image = np.zeros((len(z), len(t)))
for i in range(L):
    image += np.outer(np.ones(len(z)), log_env[:, i] ** 2) * (t * c / 2) * (i - L / 2)

plt.imshow(log_env, cmap='gray', aspect='auto', extent=[t[0], t[-1], z[-1], z[0]])
plt.xlabel('Time (s)')
plt.ylabel('Depth (m)')
plt.title('Ultrasound Image')
plt.show()
