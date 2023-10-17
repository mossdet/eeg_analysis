from joblib import Parallel, delayed
from multiprocessing import cpu_count
import numpy as np
import matplotlib.pyplot as plt


def get_complex_morlet_wavelet(fs, inst_freq, nr_cycles):
    kernel_len = round((nr_cycles/inst_freq)*fs)*2
    if np.mod(kernel_len, 2) == 0:
        kernel_len += 1

    t = np.arange(-1*(kernel_len-1)/2, (kernel_len-1)/2+1, 1)/fs

    s = nr_cycles/(2*np.pi*inst_freq)
    A = 1 / np.sqrt(s*np.sqrt(np.pi))
    theta = (2*np.pi*inst_freq*t)

    gauss_wave = np.exp((-1*np.power(t, 2)) / (2*np.power(s, 2)))
    complex_sinus = np.cos(theta) + 1j*np.sin(theta)

    cmw = np.multiply(gauss_wave, complex_sinus)
    cmw = np.multiply(A, cmw)

    # cmw = np.subtract(cmw, np.mean(cmw))

    # plt.plot(t, cmw)
    # plt.show()

    return cmw


def cmwt_serial(signal, fs, freqs, nr_cycles=7):

    if type(nr_cycles) == int:
        nr_cycles = np.full(len(freqs), nr_cycles)

    cmwtm = np.zeros((len(freqs), len(signal)), np.float64)
    for i in np.arange(len(freqs)):
        cmw_kernel = get_complex_morlet_wavelet(fs, freqs[i], nr_cycles[i])
        cmwt = np.convolve(signal, cmw_kernel, mode='same')
        cmwt = np.multiply(cmwt, np.conj(cmwt))
        cmwtm[i, :] = cmwt

    return freqs, cmwtm


def cmwt_loop(sampling_rate, sig, freq, cycles):

    cmw_kernel = get_complex_morlet_wavelet(sampling_rate, freq, cycles)
    cmwt = np.convolve(sig, cmw_kernel, mode='same')
    cmwt = np.multiply(cmwt, np.conj(cmwt))

    return cmwt


def dcmwt(signal, fs, freqs, nr_cycles=7):

    if type(nr_cycles) == int:
        nr_cycles = np.full(len(freqs), nr_cycles)

    cmwtm = np.zeros((len(freqs), len(signal)), np.float64)

    # threads, processes
    cmwtm_pl = Parallel(n_jobs=int(cpu_count()), prefer='threads')(
        delayed(cmwt_loop)(sampling_rate=fs, sig=signal,
                           freq=freqs[i], cycles=nr_cycles[i])
        for i in range(len(freqs))
    )

    for i in range(len(cmwtm_pl)):
        cmwtm[i] = cmwtm_pl[i]
    return freqs, cmwtm
