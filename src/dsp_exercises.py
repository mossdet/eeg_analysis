import numpy as np
import scipy.signal as sig
import scipy.fft
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import timeit
from time import sleep, time
import mne

from cwt import dcmwt, cmwt_serial
from dsp_tools import *
from datapaths import *


def test_convolution():

    fs = 2000

    # Compose Kernel
    # pattern = get_sine_wave(fs, sine_freq=10, nr_cycles=6)
    pattern = get_cosine_wave(fs, sine_freq=10, nr_cycles=6)

    kernel = get_gaussian(fs, len(pattern))
    # kernel = kernel/np.max(kernel)
    # pattern = pattern/np.max(pattern)
    kernel = kernel*pattern

    # Compose signal
    start_time = 0
    end_time = 10
    time_step = 1/fs
    time = np.arange(0, end_time+time_step, time_step)

    signal_len = len(time)
    # np.random.seed(25)
    signal = np.random.random(signal_len)
    loc = int(signal_len/2 - int(len(pattern)/2))  # location to plant pattern
    sel = np.array([np.arange(loc, loc+len(pattern), 1)])
    signal[sel] = np.multiply(signal[sel], (pattern+1))

    # Perform Convolution
    kernel = kernel - np.mean(kernel)
    conv_sig = convolve(signal, np.flip(kernel))
    conv_sig = np.power(conv_sig, 6)
    conv_sig = conv_sig / np.max(conv_sig)

    np_convolve = np.convolve(signal, kernel, mode='valid')
    np_convolve = np.convolve(signal, kernel, mode='same')

    fig, axs = plt.subplots(4, 1, figsize=(16, 9))
    fig.suptitle("Convolution Test")

    ax = axs[0]
    ax.plot(kernel, linewidth=1)
    ax.set_xlim(0, len(kernel))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Kernel")

    ax = axs[1]
    ax.plot(time, signal, linewidth=1)
    ax.set_xlim(np.min(time), np.max(time))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Random signal")

    ax = axs[2]
    ax.plot(time, conv_sig, linewidth=1)
    ax.set_xlim(np.min(time), np.max(time))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Convoluted signal")

    ax = axs[3]
    new_time = [(step_i*time_step) for step_i in np.arange(len(np_convolve))]
    ax.plot(new_time, np_convolve, linewidth=1)
    ax.set_xlim(np.min(new_time), np.max(new_time))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Numpy Convoluted signal")

    plt.tight_layout()
    plt.show()


def test_fourier_transform():
    # Construct test signal
    fs = 200
    sig_dur_s = 5.0
    sine_freqs = np.array([10, 15, 21, 27, 46, 58, 80])
    sine_freqs = np.array([10, 15, 20])
    signal = []
    rng = np.random.default_rng()
    for i in np.arange(len(sine_freqs)):
        nr_cycles = sig_dur_s*sine_freqs[i]
        sine_sig = get_sine_wave(fs, sine_freqs[i], nr_cycles)
        rand_sig = rng.random(len(sine_sig))     # random numbers
        if len(signal) == 0:
            # signal = np.add(sine_sig, rand_sig)
            signal = sine_sig
        else:
            # signal += np.add(sine_sig, rand_sig)
            signal += sine_sig

    signal /= len(sine_freqs)
    signal -= np.mean(signal)

    # Perform Fourier Transform
    t_0 = timeit.default_timer()  # record start time
    freq_bins, fourier = get_fourier_transform(fs, signal)
    fourier_abs = np.abs(fourier)  # get only amplitude of fourier series
    time = np.arange(len(signal))/fs
    t_1 = timeit.default_timer()
    elapsed_time = round((t_1 - t_0) * 10 ** 3, 4)
    own_dur_str = f"Own Fourier Transform duration : {elapsed_time} ms"

    # SciPy fft
    t_0 = timeit.default_timer()  # record start time
    scipy_fourier = scipy.fft.fft(signal)
    scipy_fourier_abs = scipy_fourier[0:len(freq_bins)]
    scipy_fourier_abs = abs(scipy_fourier_abs)
    t_1 = timeit.default_timer()
    elapsed_time = round((t_1 - t_0) * 10 ** 3, 3)
    fft_dur_str = f"FFT duration : {elapsed_time} ms"

    fig, axs = plt.subplots(3, 2, figsize=(16, 9))
    fig.suptitle("Fourier Transform Test")

    ax = axs[0, 0]
    ax.plot(time, signal, linewidth=1)
    ax.set_xlim(np.min(time), np.max(time))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Signal")

    ax = axs[1, 0]
    ax.plot(freq_bins, fourier_abs, linewidth=1)
    ax.set_xlim(np.min(freq_bins), np.max(freq_bins))
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")
    ax.set_title("Fourier Transform\n" + own_dur_str)

    ax = axs[2, 0]
    inv_fft_sig = get_inverse_fft(freq_bins, fourier, len(signal), fs)
    ax.plot(time, inv_fft_sig, linewidth=1)
    ax.set_xlim(np.min(time), np.max(time))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Inverse FFT Signal")

    ax = axs[0, 1]
    ax.plot(time, signal, linewidth=1)
    ax.set_xlim(np.min(time), np.max(time))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Signal")

    ax = axs[1, 1]
    ax.plot(freq_bins, scipy_fourier_abs, linewidth=1)
    ax.set_xlim(np.min(freq_bins), np.max(freq_bins))
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")
    ax.set_title("Scipy FFT\n" + fft_dur_str)

    ax = axs[2, 1]
    scipy_ifft_sig = scipy.fft.ifft(scipy_fourier)
    ax.plot(time, scipy_ifft_sig, linewidth=1)
    ax.set_xlim(np.min(time), np.max(time))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Scipy iFFT")

    plt.tight_layout()
    plt.show()


def test_real_wavelet_compose():
    fs = 2000
    inst_freq = 120
    nr_cycles = 6
    sig_len = (nr_cycles/inst_freq)*fs
    gauss_wdw = get_real_gauss_wdw(fs, inst_freq, sig_len)
    sine_wdw = get_real_sine_wdw(fs, inst_freq, sig_len)
    morlet_wdw = np.multiply(gauss_wdw, sine_wdw)

    # Construct test signal
    sig_dur_s = 5.0
    sine_freqs = np.array([10, 15, 20])
    signal = []
    rng = np.random.default_rng()
    for i in np.arange(len(sine_freqs)):
        nr_cycles = sig_dur_s*sine_freqs[i]
        sine_sig = get_sine_wave(fs, sine_freqs[i], nr_cycles)
        rand_sig = rng.random(len(sine_sig))     # random numbers
        if len(signal) == 0:
            # signal = np.add(sine_sig, rand_sig)
            signal = sine_sig
        else:
            # signal += np.add(sine_sig, rand_sig)
            signal += sine_sig

    signal /= len(sine_freqs)
    # signal -= np.mean(signal)
    time = np.arange(sig_len)/fs
    time -= np.mean(time)

    fig, axs = plt.subplots(3, 1, figsize=(16, 9))
    fig.suptitle("Real Wavelet Transform")

    ax = axs[0]
    ax.plot(time, signal, linewidth=1)
    ax.set_xlim(np.min(time), np.max(time))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Signal")

    ax = axs[1]
    ax.plot(time, sine_wdw, linewidth=1)
    ax.set_xlim(np.min(time), np.max(time))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Sine Wdw")

    ax = axs[2]
    ax.plot(time, morlet_wdw, linewidth=1)
    ax.set_xlim(np.min(time), np.max(time))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Morlet Wavelet")

    plt.tight_layout()
    plt.show()


def test_rmw_transform():
    fs = 2000
    inst_freq = 120
    nr_cycles = 6
    sig_len = (nr_cycles/inst_freq)*fs
    gauss_wdw = get_real_gauss_wdw(fs, inst_freq, sig_len)
    sine_wdw = get_real_sine_wdw(fs, inst_freq, sig_len)
    morlet_wdw = np.multiply(gauss_wdw, sine_wdw)

    time = np.arange(sig_len)/fs
    time -= np.mean(time)

    fig, axs = plt.subplots(3, 1, figsize=(16, 9))
    fig.suptitle("Morlet Wavelet Creation")

    ax = axs[0]
    ax.plot(time, gauss_wdw, linewidth=1)
    ax.set_xlim(np.min(time), np.max(time))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Gauss Wdw")

    ax = axs[1]
    ax.plot(time, sine_wdw, linewidth=1)
    ax.set_xlim(np.min(time), np.max(time))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Sine Wdw")

    ax = axs[2]
    ax.plot(time, morlet_wdw, linewidth=1)
    ax.set_xlim(np.min(time), np.max(time))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Morlet Wavelet")

    plt.tight_layout()
    plt.show()


def test_complex_morlet_wavelet():
    fs = 2000
    inst_freq = 120
    nr_cycles = 7
    cmw_kernel = get_complex_morlet_wavelet(fs, inst_freq, nr_cycles)

    pattern_nr_cycles = 10
    rmw_kernel = get_real_morlet_wvlt(fs, inst_freq, pattern_nr_cycles)
    pattern = rmw_kernel

    # Compose signal
    start_time = 0
    end_time = 1
    time_step = 1/fs
    time = np.arange(start_time, end_time+time_step, time_step)
    time -= np.mean(time)
    signal_len = len(time)
    signal = np.random.random(signal_len)
    # Plant pattern
    loc = int(signal_len/2 - int(len(pattern)/2))  # location to plant pattern
    sel = np.array([np.arange(loc, loc+len(pattern), 1)])
    signal[sel] = np.add(signal[sel], (pattern))

    # Perform real wavelet transform for a single frequency
    rmwt = convolve_complex(signal, rmw_kernel)
    # rmwt = np.convolve(signal, rmw_kernel, mode='same')
    rmwt = np.absolute(rmwt)
    rmwt *= rmwt
    # rmwt /= np.max(rmwt)

    # Perform complex wavelet transform for a single frequency
    cmwt = convolve_complex(signal, cmw_kernel)
    # cmwt = np.convolve(signal, cmw_kernel, mode='same')
    cmwt = np.multiply(cmwt, np.conj(cmwt))
    # cmwt = np.absolute(cmwt)
    # cmwt *= cmwt
    # cmwt /= np.max(cmwt)

    fig, axs = plt.subplots(6, 1, figsize=(16, 9))
    fig.suptitle("Morlet Wavelet Creation")

    ax = axs[0]
    ax.plot(time, signal, linewidth=1)
    ax.set_xlim(np.min(time), np.max(time))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Simulated Signal")

    ax = axs[1]
    kernel_time = np.arange(len(rmw_kernel))/len(rmw_kernel)
    kernel_time -= np.mean(kernel_time)
    ax.plot(kernel_time, rmw_kernel, linewidth=1)
    ax.set_xlim(np.min(kernel_time), np.max(kernel_time))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Real  Morlet Wavelet Kernel")

    ax = axs[2]
    kernel_time = np.arange(len(cmw_kernel))/len(cmw_kernel)
    kernel_time -= np.mean(kernel_time)
    ax.plot(kernel_time, cmw_kernel, linewidth=1)
    ax.set_xlim(np.min(kernel_time), np.max(kernel_time))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Complex Morlet Wavelet Kernel")

    ax = axs[3]
    ax.plot(time, rmwt, linewidth=1)
    ax.set_xlim(np.min(time), np.max(time))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Power($\\mu V^2$)")
    ax.set_title("Real Morlet Wavelet Transform")

    ax = axs[4]
    ax.plot(time, cmwt, linewidth=1)
    ax.set_xlim(np.min(time), np.max(time))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Power($\\mu V^2$)")
    ax.set_title("Complex Morlet Wavelet Transform")

    ax = axs[5]
    # ax.pcolormesh(time, freqs, _mat, cmap='viridis', shading='gouraud')

    plt.tight_layout()
    plt.show()


def test_cmwt():
    fs = 2000

    pattern_freqs = [60, 80, 100, 150, 200, 250, 350, 400, 450, 500]
    sig_duration = 10
    time, signal = compose_signal(fs, sig_duration, pattern_freqs)

    cmwt_freqs = np.arange(60, 600, 10)
    # cmwt_freqs = np.insert(cmwt_freqs, 0, 1)
    freqs, cmwtm = cmwt(signal, fs, cmwt_freqs, nr_cycles=7)
    cmwtm /= np.max(cmwtm)

    plot_ok = True
    if plot_ok:
        fig, axs = plt.subplots(2, 1, figsize=(16, 9))
        fig.suptitle("Morlet Wavelet Transform")

        ax = axs[0]
        ax.plot(time, signal, linewidth=1)
        ax.set_xlim(np.min(time), np.max(time))
        ax.set_xlabel("Time (s)")
        ax.set_xticks(np.arange(0, max(time)+0.5, 0.5))
        ax.set_ylabel("Amplitude")
        ax.set_title("Simulated Signal ($\\mu V$)")

        ax = axs[1]
        im = ax.pcolormesh(time, freqs, cmwtm,
                           cmap='viridis', shading='gouraud')
        ax.set_xticks(np.arange(0, max(time)+0.5, 0.5))
        # fig.colorbar(im, ax=ax)

        title_str = f"{pattern_freqs}"
        # ax.set_title(title_str)
        print(title_str)

        plt.tight_layout()
        plt.show()


def test_hilbert_transform():
    fs = 2000
    foi = 100
    nyq_fs = np.floor(fs/2)

    # Get signal
    pattern_freqs = np.arange(50, 550, 50)
    attenuation = 0.2
    signal = []
    for f in pattern_freqs:
        s = compose_signal(fs, f, attenuation)
        if len(signal) == 0:
            signal = s
        else:
            signal = np.append(signal, s)
    time = np.arange(len(signal))/fs

    ##########################################
    # Raw Signal
    # Manually obtain Hilbert Transform
    freq_bins = scipy.fft.fftfreq(len(signal), 1/fs)
    fft_vals = scipy.fft.fft(signal)
    pos_freqs_sel = np.logical_and(freq_bins > 0, freq_bins < nyq_fs)
    neg_freqs_sel = freq_bins < 0
    # Rotating by -90 and 90 degrees (-j and j or -pi/2 and pi/2)
    # fft_copy[pos_freqs_sel] += (1j*-np.pi/2)
    # fft_copy[neg_freqs_sel] += (1j*np.pi/2)
    # The rotation is also achieved by doubling the real part and zeroing the imaginary part
    fft_vals[pos_freqs_sel] *= 2
    fft_vals[neg_freqs_sel] -= fft_vals[neg_freqs_sel]
    hil = scipy.fft.ifft(fft_vals)
    hil_envelope = np.abs(hil)
    hil_envelope /= np.max(hil_envelope)
    # Scipy Hilbert Transform
    analytic_signal = scipy.signal.hilbert(signal)
    amplitude_envelope = np.abs(analytic_signal)
    amplitude_envelope /= np.max(amplitude_envelope)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = (np.diff(instantaneous_phase) / (2.0*np.pi) * fs)

    ##########################################
    # Bandpassed Signal
    bp_signal = fir_bp_filter(fs, signal, foi-5, foi+5, 1024)
    # Manually obtain Hilbert Transform
    freq_bins = scipy.fft.fftfreq(len(bp_signal), 1/fs)
    fft_vals = scipy.fft.fft(bp_signal)
    pos_freqs_sel = np.logical_and(freq_bins > 0, freq_bins < nyq_fs)
    neg_freqs_sel = freq_bins < 0
    fft_vals[pos_freqs_sel] *= 2
    fft_vals[neg_freqs_sel] -= fft_vals[neg_freqs_sel]
    hil_bp = scipy.fft.ifft(fft_vals)
    hil_envelope_bp = np.abs(hil_bp)
    # hil_envelope_bp = hil_envelope_bp**10
    hil_envelope_bp /= np.max(hil_envelope_bp)
    # Scipy Hilbert Transform
    analytic_signal_bp = scipy.signal.hilbert(bp_signal)
    amplitude_envelope_bp = np.abs(analytic_signal_bp)
    # amplitude_envelope_bp = amplitude_envelope_bp**10
    amplitude_envelope_bp /= np.max(amplitude_envelope_bp)
    instantaneous_phase_bp = np.unwrap(np.angle(analytic_signal_bp))
    instantaneous_frequency_bp = (
        np.diff(instantaneous_phase_bp) / (2.0*np.pi) * fs)

    plot_ok = True
    if plot_ok:
        fig, axs = plt.subplots(2, 3, figsize=(16, 9))
        fig.suptitle("Hilbert Transform")

        # Raw Signal
        ax = axs[0, 0]
        ax.plot(time, signal, linewidth=1)
        ax.plot(time, hil_envelope, linewidth=1)
        ax.set_xlim(np.min(time), np.max(time))
        ax.set_xlabel("Time (s)")
        ax.set_xticks(np.arange(0, max(time)+0.5, 0.5))
        ax.set_ylabel("Amplitude ($\\mu V$)")
        ax.set_title("Raw Signal Hilbert Transform\n(Manual Implementation)")

        ax = axs[1, 0]
        ax.plot(time, signal, linewidth=1)
        ax.plot(time, amplitude_envelope, linewidth=1)
        ax.set_xlim(np.min(time), np.max(time))
        ax.set_xlabel("Time (s)")
        ax.set_xticks(np.arange(0, max(time)+0.5, 0.5))
        ax.set_ylabel("Amplitude")
        ax.set_title("Raw Signal Hilbert Transform\n(Scipy Implementation)")

        # Bandpassed Signal
        ax = axs[0, 1]
        ax.plot(time, bp_signal, linewidth=1)
        ax.plot(time, hil_envelope_bp, linewidth=1)
        ax.set_xlim(np.min(time), np.max(time))
        ax.set_xlabel("Time (s)")
        ax.set_xticks(np.arange(0, max(time)+0.5, 0.5))
        ax.set_ylabel("Amplitude ($\\mu V$)")
        ax.set_title(
            "Bandpassed Signal Hilbert Transform\n(Manual Implementation)")

        ax = axs[1, 1]
        ax.plot(time, bp_signal, linewidth=1)
        ax.plot(time, amplitude_envelope_bp, linewidth=1)
        ax.set_xlim(np.min(time), np.max(time))
        ax.set_xlabel("Time (s)")
        ax.set_xticks(np.arange(0, max(time)+0.5, 0.5))
        ax.set_ylabel("Amplitude")
        ax.set_title(
            "Bandpassed Signal Hilbert Transform\n(Scipy Implementation)")

        # Bandpassed Signal Power
        ax = axs[0, 2]
        ax.plot(time, bp_signal, linewidth=1)
        ax.plot(time, bp_signal**2/np.max(bp_signal**2), linewidth=1)
        ax.set_xlim(np.min(time), np.max(time))
        ax.set_xlabel("Time (s)")
        ax.set_xticks(np.arange(0, max(time)+0.5, 0.5))
        ax.set_ylabel("Amplitude ($\\mu V$)")
        ax.set_title(
            "Bandpassed Signal Power")

        ax = axs[1, 2]
        ax.plot(time, bp_signal, linewidth=1)
        ax.plot(time, bp_signal**2/np.max(bp_signal**2), linewidth=1)
        ax.set_xlim(np.min(time), np.max(time))
        ax.set_xlabel("Time (s)")
        ax.set_xticks(np.arange(0, max(time)+0.5, 0.5))
        ax.set_ylabel("Amplitude")
        ax.set_title(
            "Bandpassed Signal Power")

        plt.tight_layout()
        wm = plt.get_current_fig_manager()
        wm.window.state('zoomed')
        plt.show()


def test_fft_for_hfo():
    fs = 1000
    # Get signal
    pattern_freqs = np.arange(100, 510, 50)
    # pattern_freqs = np.tile(pattern_freqs, 2)
    attenuation = -0.5
    rand_signal = np.random.random(2*fs)
    signal = rand_signal
    for f in pattern_freqs:
        s = compose_signal(fs, f, attenuation)
        if len(signal) == 0:
            signal = s
        else:
            signal = np.append(signal, s)

        signal = np.append(signal, np.random.random(2*fs))
    time = np.arange(len(signal))/fs

    freq_bins = scipy.fft.fftfreq(len(signal), 1/fs)
    fft_res = scipy.fft.fft(signal)
    real_freqs = freq_bins[freq_bins > 0]
    real_ampl = abs(fft_res[freq_bins > 0])
    ifft_sig = scipy.fft.ifft(fft_res)
    err_sig = np.abs((signal-ifft_sig)/signal)*100

    plot_period = 1
    init_samps = np.arange(0, np.max(time)-plot_period-1, plot_period)
    for ss in init_samps:

        seg_sel = np.logical_and(time >= ss, time <= ss+plot_period)
        time_seg = time[seg_sel]
        sig_seg = signal[seg_sel]
        freq_bins_seg = scipy.fft.fftfreq(len(sig_seg), 1/fs)
        fft_res_seg = scipy.fft.fft(sig_seg)
        real_freqs_seg = freq_bins_seg[freq_bins_seg > 0]
        real_ampl_seg = abs(fft_res_seg[freq_bins_seg > 0])
        ifft_sig_seg = scipy.fft.ifft(fft_res_seg)
        err_sig_seg = np.abs((sig_seg-ifft_sig_seg)/sig_seg)*100

        plot_ok = True
        if plot_ok:
            fig, axs = plt.subplot_mosaic([['A', 'B', 'E', 'F'], ['A', 'C', 'E', 'G'], ['A', 'D', 'E', 'H']],
                                          layout='constrained')
            fig.suptitle("FFT Test")
            trans = mtransforms.ScaledTranslation(
                10/72, -5/72, fig.dpi_scale_trans)

            label = 'A'
            ax = axs[label]
            ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
                    fontsize='medium', verticalalignment='top', fontfamily='serif',
                    bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0))

            ax.plot(real_freqs, real_ampl, linewidth=1)
            ax.set_xlim(np.min(real_freqs), np.max(real_freqs))
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Power ($\\mu V^2$)")
            ax.set_title("FFTs")

            label = 'B'
            ax = axs[label]
            ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
                    fontsize='medium', verticalalignment='top', fontfamily='serif',
                    bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0))
            ax.plot(time, signal, linewidth=1)
            ax.set_xlim(np.min(time), np.max(time))
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude ($\\mu V$)")
            ax.set_title("Raw Signal")

            label = 'C'
            ax = axs[label]
            ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
                    fontsize='medium', verticalalignment='top', fontfamily='serif',
                    bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0))
            ax.plot(time, ifft_sig, linewidth=1)
            ax.set_xlim(np.min(time), np.max(time))
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude ($\\mu V$)")
            ax.set_title("Inverse FFT")

            label = 'D'
            ax = axs[label]
            ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
                    fontsize='medium', verticalalignment='top', fontfamily='serif',
                    bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0))
            ax.plot(time, err_sig, linewidth=1)
            ax.set_xlim(np.min(time), np.max(time))
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Error (%)")
            ax.set_title("FFTs Diff")

            label = 'E'
            ax = axs[label]
            ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
                    fontsize='medium', verticalalignment='top', fontfamily='serif',
                    bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0))
            ax.plot(real_freqs_seg, real_ampl_seg, linewidth=1)
            ax.set_xlim(np.min(real_freqs_seg), np.max(real_freqs_seg))
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Power ($\\mu V^2$)")
            ax.set_title(
                f"FFT Segment ({np.min(time_seg)}-{np.max(time_seg)} s)")

            label = 'F'
            ax = axs[label]
            ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
                    fontsize='medium', verticalalignment='top', fontfamily='serif',
                    bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0))
            ax.plot(time_seg, sig_seg, linewidth=1)
            ax.set_xlim(np.min(time_seg), np.max(time_seg))
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude ($\\mu V$)")
            ax.set_title(
                f"Signal Segment({np.min(time_seg)}-{np.max(time_seg)} s)")

            label = 'G'
            ax = axs[label]
            ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
                    fontsize='medium', verticalalignment='top', fontfamily='serif',
                    bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0))
            ax.plot(time_seg, ifft_sig_seg, linewidth=1)
            ax.set_xlim(np.min(time_seg), np.max(time_seg))
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude ($\\mu V$)")
            ax.set_title(
                f"iFFT Reconstructed Segment({np.min(time_seg)}-{np.max(time_seg)} s)")

            label = 'H'
            ax = axs[label]
            ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
                    fontsize='medium', verticalalignment='top', fontfamily='serif',
                    bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0))
            ax.plot(time_seg, err_sig_seg, linewidth=1)
            ax.set_xlim(np.min(time_seg), np.max(time_seg))
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Error (%)")
            ax.set_title(
                f"Signal Segment and Reconstructed Error %({np.min(time_seg)}-{np.max(time_seg)} s)")

            plt.tight_layout()
            wm = plt.get_current_fig_manager()
            wm.window.state('zoomed')
            plt.show(block=True)
            plt.close()


def plt_ax_label(ax, trans, label):
    ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
            fontsize='medium', verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0))


def test_stft():
    fs = 2000

    # Get signal
    pattern_freqs = np.arange(100, 510, 50)
    # pattern_freqs = np.tile(pattern_freqs, 2)
    attenuation = 0.5
    rand_signal = np.random.random(2*fs)
    signal = rand_signal
    for f in pattern_freqs:
        s = compose_signal(fs, f, attenuation)
        if len(signal) == 0:
            signal = s
        else:
            signal = np.append(signal, s)

        signal = np.append(signal, np.random.random(2*fs))

    time = np.arange(len(signal))/fs

    # Short Time Fourier Transform
    seglen = int(round(fs*0.05))
    overlaplen = seglen/2
    f, t, zx = sig.stft(signal, fs, 'hamming', seglen, overlaplen,
                        nfft=seglen*2, detrend='linear')
    zx = np.abs(zx)

    cmwt_freqs = np.arange(50, 1000, 10)
    cmwt_freqs, cmwtm = cmwt(signal, fs, cmwt_freqs, nr_cycles=11)

    # Plot signals
    fig, axs = plt.subplot_mosaic(
        [['A'], ['A'], ['B'], ['B'], ['B'], ['C'], ['C'], ['C']], layout='constrained')
    trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
    fig.suptitle("STFT Test")

    label = 'A'
    ax = axs[label]
    plt_ax_label(ax, trans, label)
    ax.plot(time, signal, linewidth=1)
    ax.set_xlim(np.min(time), np.max(time))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude ($\\mu V$)")
    ax.set_title(f"Signal")

    label = 'B'
    ax = axs[label]
    plt_ax_label(ax, trans, label)
    ax.pcolormesh(t, f, zx,
                  cmap='viridis', shading='gouraud')
    ax.set_xlim(np.min(time), np.max(time))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz))")
    ax.set_ylim(50, np.max(f))
    ax.set_title(f"STFT")

    label = 'C'
    ax = axs[label]
    plt_ax_label(ax, trans, label)
    ax.pcolormesh(time, cmwt_freqs, cmwtm,
                  cmap='viridis', shading='gouraud')
    ax.set_xlim(np.min(time), np.max(time))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz))")
    ax.set_ylim(50, np.max(cmwt_freqs))
    ax.set_title(f"CMWT")

    plt.tight_layout()
    wm = plt.get_current_fig_manager()
    wm.window.state('zoomed')
    plt.show(block=True)
    plt.close()


def test_multitaper():
    fs = 1000

    # Get signal
    pattern_freqs = np.arange(100, 510, 50)
    # pattern_freqs = np.tile(pattern_freqs, 2)
    attenuation = -0.4
    rand_signal = np.random.random(2*fs)
    signal = rand_signal
    for f in pattern_freqs:
        s = compose_signal(fs, f, attenuation)
        if len(signal) == 0:
            signal = s
        else:
            signal = np.append(signal, s)

        signal = np.append(signal, np.random.random(2*fs))

    time = np.arange(len(signal))/fs

    # Short Time Fourier Transform
    seglen = int(round(fs*0.05))
    overlaplen = int(seglen-1)
    f, t, zx = sig.stft(signal, fs, 'hamming', seglen, overlaplen,
                        nfft=seglen*2, detrend='linear')
    zx = np.abs(zx)
    zx /= np.max(zx)

    # Wavelet Transform
    cmwt_freqs = np.arange(50, 500, 5)
    n_cycles_vec = np.round(cmwt_freqs/100)*2+9

    cmwt_freqs, cmwtm = dcmwt(signal, fs, cmwt_freqs, nr_cycles=n_cycles_vec)
    cmwtm /= np.max(cmwtm)

    # Multitaper
    # signal = np.array([signal])
    n_channels = 1
    ch_names = ["Fp1"]
    ch_types = ["eeg"]
    info = mne.create_info(ch_names, ch_types=ch_types, sfreq=fs)
    simulated_raw = mne.io.RawArray([signal], info)
    signal_epoch = mne.EpochsArray(
        np.array([np.array([np.array(signal)])]), info)
    # simulated_raw.plot(show_scrollbars=False, show_scalebars=False)

    mt_freqs = cmwt_freqs

    mt_data = mne.time_frequency.tfr_array_multitaper(
        np.array([np.array([np.array(signal)])]), fs, mt_freqs, n_cycles_vec, zero_mean=True, time_bandwidth=2.0, use_fft=True, decim=1, output='power', n_jobs=-1, verbose=None)
    mt_data = mt_data[0, 0, :, :]
    mt_data /= np.max(mt_data)

    # stw_data = mne.time_frequency.tfr_stockwell(signal_epoch, fmin=np.min(cmwt_freqs), fmax=np.max(cmwt_freqs), width=0.2)

    mne_dwt_data = mne.time_frequency.tfr_array_morlet(
        np.array([np.array([np.array(signal)])]), fs, cmwt_freqs, n_cycles_vec, zero_mean=True, use_fft=True, decim=1, output='power', n_jobs=-1, verbose=None)
    mne_dwt_data = mne_dwt_data[0, 0, :, :]

    # combined_data = np.multiply(cmwtm, mt_data)
    combined_data = np.multiply(cmwtm, mt_data)

    # Plot signals
    fig, axs = plt.subplot_mosaic(
        [['A'], ['B'], ['B'], ['C'], ['C'], ['D'], ['D'], ['E'], ['E']], layout='constrained')
    trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
    fig.suptitle("STFT Test")

    label = 'A'
    ax = axs[label]
    plt_ax_label(ax, trans, label)
    ax.plot(time, signal, linewidth=1)
    ax.set_xlim(np.min(time), np.max(time))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude ($\\mu V$)")
    ax.set_title(f"Signal")

    label = 'B'
    ax = axs[label]
    plt_ax_label(ax, trans, label)
    ax.pcolormesh(t, f, zx,
                  cmap='viridis', shading='gouraud')
    ax.set_xlim(np.min(time), np.max(time))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz))")
    ax.set_ylim(50, np.max(f))
    ax.set_title(f"STFT")

    label = 'C'
    ax = axs[label]
    plt_ax_label(ax, trans, label)
    ax.pcolormesh(time, cmwt_freqs, cmwtm,
                  cmap='viridis', shading='gouraud')
    ax.set_xlim(np.min(time), np.max(time))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz))")
    ax.set_ylim(50, np.max(cmwt_freqs))
    ax.set_title(f"DCWT")

    label = 'D'
    ax = axs[label]
    plt_ax_label(ax, trans, label)
    ax.pcolormesh(time, mt_freqs, mne_dwt_data,  # mne_dwt_data,
                  cmap='viridis', shading='gouraud')
    ax.set_xlim(np.min(time), np.max(time))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz))")
    ax.set_ylim(np.min(mt_freqs), np.max(mt_freqs))
    ax.set_title(f"MNE Python DWT")

    label = 'E'
    ax = axs[label]
    plt_ax_label(ax, trans, label)
    ax.pcolormesh(time, mt_freqs, mt_data,
                  cmap='viridis', shading='gouraud')
    ax.set_xlim(np.min(time), np.max(time))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz))")
    ax.set_ylim(np.min(mt_freqs), np.max(mt_freqs))
    ax.set_title(f"Multitaper")

    plt.tight_layout()
    fig_name = images_path + "Frequency_Analysis.png"
    plt.savefig(fig_name, bbox_inches='tight', dpi=2000)
    plt.show(block=True)
    plt.close()


# test_convolution()
# test_fourier_transform()
# test_wavelet()
# test_real_wavelet_compose()
# test_rmw_transform()
# test_complex_morlet_wavelet()
# test_cmwt()
# test_hilbert_transform()
# test_fft_for_hfo()
# test_stft()
test_multitaper()
stop = 1
