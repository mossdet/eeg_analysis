import numpy as np
from scipy.ndimage import uniform_filter1d, median_filter
from scipy.signal import lfilter, firwin, filtfilt, find_peaks
import scipy.signal as sig
import scipy
import matplotlib.pyplot as plt
import timeit


# Symmetric moving average filter
def movmean(sig_vec, wdw_len):
    movAvgSig = uniform_filter1d(sig_vec, size=wdw_len)
    return movAvgSig


def movmedian(sig_vec, wdw_len):
    movmed_sig = median_filter(sig_vec, size=wdw_len, mode='wrap')
    return movmed_sig


def fir_bp_filter(sampling_rate, sig_vec, lowcut, highcut, fir_order):
    ntaps = int(round(fir_order/2))
    wdwType = 'hamming'  # 'hamming','boxcar', 'blackmanharris'
    taps = firwin(numtaps=ntaps, cutoff=[lowcut, highcut], fs=sampling_rate,
                  window=wdwType, pass_zero=False, scale=True)

    '''
    fltrd_sig = sig.lfilter(taps, 1.0, np.flip(sig_vec))
    fltrd_sig[0:ntaps-1] = fltrd_sig[ntaps]
    fltrd_sig[-1*ntaps:-1] = fltrd_sig[-1*ntaps-1]

    fltrd_sig = sig.lfilter(taps, 1.0, np.flip(fltrd_sig))
    fltrd_sig[0:ntaps-1] = fltrd_sig[ntaps]
    fltrd_sig[-1*ntaps:-1] = fltrd_sig[-1*ntaps-1]
    '''

    fltrd_sig = sig.filtfilt(taps, 1.0, sig_vec)

    return fltrd_sig


def iir_bp_filter(sampling_rate, sig_vec, lowcut, highcut, order):
    b, a = sig.iirfilter(
        order, [lowcut, highcut], btype='band', analog=False, ftype='cheby2')

    fltrd_sig = filtfilt(b, a, sig_vec)


def butter_bp_filter(sampling_rate, sig_vec, lowcut, highcut, order=5):
    sos = sig.butter(lowcut, 15, 'hp', fs=sampling_rate, output='sos')

    sig.sosfiltfilt(sos, sig_vec)

    nyq = sampling_rate/2
    low = lowcut / nyq
    high = highcut / nyq
    b, a = sig.butter(order, [low, high], btype='bandpass',
                      analog=False)

    fltrd_sig = filtfilt(b, a, sig_vec)

    return fltrd_sig


def butter_bp_filter_empatica_test(sampling_rate, sig_vec, lowcut, highcut, order=5):
    nyq = sampling_rate/2
    low = lowcut / nyq
    high = highcut / nyq
    b, a = sig.butter(order, [low, high], btype='bandpass',
                      analog=False)

    fltrd_sig = filtfilt(b, a, sig_vec)

    return fltrd_sig


def butter_bp_filter_empatica(sampling_rate, sig_vec, lowcut, highcut, order=5):
    nyq = sampling_rate/2
    low = lowcut / nyq
    high = highcut / nyq
    b, a = sig.butter(order, [low, high], btype='bandpass',
                      analog=False)

    # fltrd_sig = filtfilt(b, a, sig_vec)
    fltrd_sig = sig.lfilter(b, a, np.flip(sig_vec))
    fltrd_sig = sig.lfilter(b, a, np.flip(fltrd_sig))

    return fltrd_sig


def find_these_peaks(signal_vec):
    # return peak locations sorted in ascending order
    peaks_locs = find_peaks(signal_vec, distance=3, prominence=3)[0]
    peaks_locs = np.sort(peaks_locs)
    return peaks_locs


def find_these_peaks_rf(signal_vec):
    # return peak locations sorted in ascending order
    peaks_locs = find_peaks(signal_vec)[0]
    peaks_locs = np.sort(peaks_locs)
    return peaks_locs


def compose_signal(fs, pattern_freq, pattern_attenuation=0.8):

    # Generate pattern
    nr_cycles = 12
    pattern = get_real_morlet_wvlt(fs, pattern_freq, nr_cycles)
    pattern = pattern[int(fs*6/pattern_freq):-1*int(fs*6/pattern_freq)]
    pattern /= np.max(np.abs(pattern))
    pattern -= np.mean(pattern)
    pattern *= (1-pattern_attenuation)
    patt_len = len(pattern)

    # Compose signal
    signal_len = patt_len*10
    signal = np.random.random(signal_len)
    pattern += np.mean(signal)

    # location to plant pattern
    plant_loc = int(round(signal_len/2 - patt_len/2))
    sel = np.array([np.arange(plant_loc, plant_loc+patt_len, 1)])
    signal[sel] = np.add(signal[sel], pattern)/2

    return signal


def convolve_real(signal, kernel):
    kernel = np.flip(kernel)
    kernel_len = len(kernel)
    half_kl = int(np.round(kernel_len/2))
    zero_pad = np.zeros((kernel_len-1)*1, dtype=float)
    zero_padded_sig = np.append(zero_pad, np.append(signal, zero_pad))

    convo_result = np.zeros(len(zero_padded_sig), dtype=float)

    for i in range(len(zero_padded_sig)-kernel_len+1):
        sig_conv_seg = zero_padded_sig[i:i+kernel_len]
        conv_val = np.dot(sig_conv_seg, kernel)
        loc = i + half_kl
        convo_result[loc] = conv_val

    convo_result = convo_result[len(zero_pad):-1*len(zero_pad)]
    return convo_result


def convolve_complex(signal, kernel):
    kernel = np.flip(kernel)
    kernel_len = len(kernel)
    half_kl = int(np.round(kernel_len/2))
    zero_pad = np.zeros((kernel_len-1)*1, dtype=float)*1j
    zero_padded_sig = np.append(zero_pad, np.append(signal, zero_pad))

    convo_result = np.zeros(len(zero_padded_sig), dtype=float)*1j

    for i in range(len(zero_padded_sig)-kernel_len+1):
        sig_conv_seg = zero_padded_sig[i:i+kernel_len]
        conv_val = np.dot(sig_conv_seg, kernel)
        loc = i + half_kl
        convo_result[loc] = conv_val

    convo_result = convo_result[len(zero_pad):-1*len(zero_pad)]
    return convo_result


def get_gaussian(fs, kernel_len):

    time_step = 1/fs
    period = kernel_len/fs
    x_axis = np.arange(-1*(period/2), (period/2), time_step)
    mean = np.mean(x_axis)
    sd = np.std(x_axis)/2
    gaussian_vec = scipy.stats.norm.pdf(x_axis, mean, sd)

    return gaussian_vec


def get_sine_wave(fs, sine_freq, nr_cycles):
    start_time = 0
    end_time = nr_cycles/sine_freq
    time_step = 1/fs
    time = np.arange(0, end_time+time_step, time_step)

    theta = 0   # phase
    amplitude = 1

    sine_wave = amplitude * np.sin(2 * np.pi * sine_freq * time + theta)

    return sine_wave


def get_cosine_wave(fs, sine_freq, nr_cycles):
    start_time = 0
    end_time = nr_cycles/sine_freq
    time_step = 1/fs
    time = np.arange(0, end_time+time_step, time_step)

    theta = 0   # phase
    amplitude = 1

    sine_wave = amplitude * np.cos(2 * np.pi * sine_freq * time + theta)

    return sine_wave


def get_fourier_transform(fs, signal):
    N = len(signal)     # length of sequence
    freq_prec = fs/N

    # time starts at 0; dividing by N normalizes to 1
    t = np.arange(N)/fs

    # frequency bins
    freq_bins = np.arange(0, (fs/2)+freq_prec, freq_prec)

    # initialize Fourier coefficients
    fourier = 1j*np.zeros(len(freq_bins))

    # Fourier transform
    for fi in np.arange(len(freq_bins)):
        freq_bin_val = freq_bins[fi]
        # create sine wave
        sine_wave = np.exp(np.multiply(-1j*2*np.pi*(freq_bin_val), t))
        # compute dot product between sine wave and data
        fourier[fi] = np.sum(np.multiply(sine_wave, signal))

    return freq_bins, fourier


def get_inverse_fft(freq_bins, fourier, sig_len, fs):
    t = np.arange(sig_len)/fs
    signal = []
    # Inverse Fourier transform
    for i in np.arange(len(freq_bins)):
        f = freq_bins[i]
        sine_wave = np.exp(np.multiply(1j*2*np.pi*(f), t))
        sine_wave = np.multiply(fourier[i], sine_wave)
        if len(signal) == 0:
            signal = sine_wave
        else:
            signal += sine_wave
    signal /= len(freq_bins)

    return signal


def get_inverse_fft_range(freq_bins, fourier, time):
    signal = []
    # Inverse Fourier transform
    for i in np.arange(0, len(freq_bins), 2):
        f = freq_bins[i]
        sine_wave = np.exp(np.multiply(1j*2*np.pi*(f), time))
        sine_wave = np.multiply(fourier[i], sine_wave)
        if len(signal) == 0:
            signal = sine_wave
        else:
            signal += sine_wave
    signal /= len(freq_bins)

    return signal


def get_real_sine_wdw(fs, sine_freq, sig_len):
    t = np.arange(sig_len)/fs
    # sine_wave = np.exp(np.multiply(-1j*2*np.pi*(sine_freq), t))
    theta = 0   # phase
    amplitude = 1
    sine_wave = amplitude * np.sin(2 * np.pi * sine_freq * t + theta)

    return sine_wave


def get_real_cosine_wdw(fs, inst_freq, sig_len):
    t = np.arange(sig_len)/fs
    # sine_wave = np.exp(np.multiply(-1j*2*np.pi*(sine_freq), t))
    theta = 0   # phase
    amplitude = 1
    cos_wave = amplitude * np.cos(2 * np.pi * inst_freq * t + theta)

    return cos_wave


def get_real_gauss_wdw(fs, gauss_freq, sig_len, nr_cycles=7):
    a = 1
    m = 0
    t = np.arange(sig_len)/fs
    t -= np.mean(t)
    s = nr_cycles/(2*np.pi*gauss_freq)
    gauss_wave = a * np.exp((-1*(t-m)**2) / (2*s**2))
    return gauss_wave


def get_real_morlet_wvlt(fs, inst_freq, nr_cycles):
    kernel_len = round((nr_cycles/inst_freq)*fs)*2
    if np.mod(kernel_len, 2) == 0:
        kernel_len += 1

    t = np.arange(kernel_len)/fs
    t -= np.mean(t)

    s = nr_cycles/(2*np.pi*inst_freq)
    A = 1 / np.sqrt(s*np.sqrt(np.pi))
    theta = (2*np.pi*inst_freq*t)

    gauss_wave = np.exp((-1*np.power(t, 2)) / (2*np.power(s, 2)))
    complex_sinus = np.cos(theta) + np.sin(theta)

    rmw = np.multiply(gauss_wave, complex_sinus)
    rmw = np.multiply(A, rmw)

    return rmw


def get_complex_morlet_wavelet(fs, inst_freq, nr_cycles):
    kernel_len = round((nr_cycles/inst_freq)*fs)*2
    if np.mod(kernel_len, 2) == 0:
        kernel_len += 1

    t = np.arange(kernel_len)/fs
    t -= np.mean(t)

    s = nr_cycles/(2*np.pi*inst_freq)
    A = 1 / np.sqrt(s*np.sqrt(np.pi))
    theta = (2*np.pi*inst_freq*t)

    gauss_wave = np.exp((-1*np.power(t, 2)) / (2*np.power(s, 2)))
    complex_sinus = np.cos(theta) + 1j*np.sin(theta)

    cmw = np.multiply(gauss_wave, complex_sinus)
    cmw = np.multiply(A, cmw)

    return cmw


def get_cmwt(signal, fs, freqs, nr_cycles=7):
    cmwt_mat = np.zeros((len(freqs), len(signal)), np.float64)
    for i in np.arange(len(freqs)):
        inst_freq = freqs[i]
        cmw_kernel = get_complex_morlet_wavelet(fs, inst_freq, nr_cycles)
        cmwt = np.convolve(signal, cmw_kernel, mode='same')

        # power can be extracted either by squaring the length of the complex vector
        cmwt = np.power(cmwt, 2)
        # or by multiplying the complex vector by its conjugate, this is faster!
        cmwt = np.multiply(cmwt, np.conj(cmwt))

        cmwt_mat[i, :] = cmwt

    return freqs, cmwt_mat
