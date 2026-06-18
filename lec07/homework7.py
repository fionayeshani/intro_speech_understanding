import numpy as np

def major_chord(f, Fs):
    '''
    Generate a one-half-second major chord.
    '''
    N = int(0.5 * Fs)
    n = np.arange(N)

    f_root = f
    f_third = f * (2 ** (4/12))
    f_fifth = f * (2 ** (7/12))

    x = (
        np.cos(2 * np.pi * f_root * n / Fs) +
        np.cos(2 * np.pi * f_third * n / Fs) +
        np.cos(2 * np.pi * f_fifth * n / Fs)
    )

    return x

def dft_matrix(N):
    '''
    Create a DFT transform matrix, W, of size N.
    '''
    n = np.arange(N)
    k = n.reshape((N, 1))

    W = np.exp(-2j * np.pi * k * n / N)

    return W

def spectral_analysis(x, Fs):
    '''
    Find the three loudest frequencies in x.
    '''
    X = np.fft.fft(x)
    magnitude = np.abs(X)

    freqs = np.fft.fftfreq(len(x), d=1/Fs)

    positive = freqs >= 0
    freqs = freqs[positive]
    magnitude = magnitude[positive]

    indices = np.argsort(magnitude)[-3:]
    loudest = np.sort(freqs[indices])

    f1, f2, f3 = loudest

    return f1, f2, f3