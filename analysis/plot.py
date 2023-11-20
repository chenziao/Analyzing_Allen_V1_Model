import numpy as np
import pandas as pd
import xarray as xr
import scipy.signal as ss
import matplotlib.pyplot as plt

import pywt
from fooof import FOOOF
from fooof.sim.gen import gen_aperiodic, gen_model


pop_color = {'e': 'red', 'Pvalb': 'blue', 'Sst': 'green', 'Htr3a': 'purple'}
pop_color = {p: 'tab:' + clr for p, clr in pop_color.items()}
pop_names = list(pop_color.keys())
t_stop = 3.0

def raster(pop_spike, pop_color, id_column='node_ids', s=0.1, ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    ymin, ymax = [], []
    for p, spike_df in pop_spike.items():
        if len(spike_df) > 0:
            ids = spike_df[id_column].values
            ax.scatter(spike_df['timestamps'], ids, c=pop_color[p], s=s, label=p)
            ymin.append(ids.min())
            ymax.append(ids.max())
    ax.set_xlim(left=0.)
    ax.set_ylim([np.min(ymin) - 1, np.max(ymax) + 1])
    ax.set_title('Spike Raster Plot')
    ax.legend(loc='upper right', framealpha=0.9, markerfirst=False)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Cell ID')
    return ax


def firing_rate_histogram(pop_fr, pop_color, bins=30, min_fr=None,
                          logscale=False, stacked=True, ax=None):
    if logscale and min_fr is not None:
        pop_fr = {p: np.fmax(fr, min_fr) for p, fr in pop_fr.items()}
    fr = np.concatenate(list(pop_fr.values()))
    if logscale:
        fr = fr[fr > 0]
        bins = np.geomspace(fr.min(), fr.max(), bins + 1)
    else:
        bins = np.linspace(fr.min(), fr.max(), bins + 1)
    pop_names = list(pop_fr.keys())
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if stacked:
        ax.hist(pop_fr.values(), bins=bins, label=pop_names,
                color=[pop_color[p] for p in pop_names], stacked=True)
    else:
        for p, fr in pop_fr.items():
            ax.hist(fr, bins=bins, label=p, color=pop_color[p], alpha=0.5)
    if logscale:
        ax.set_xscale('log')
        plt.draw()
        xt = ax.get_xticks()
        xtl = [x.get_text() for x in ax.get_xticklabels()]
        xt = np.append(xt, min_fr)
        xtl.append('0')
        ax.set_xticks(xt)
        ax.set_xticklabels(xtl)
    ax.set_xlim(bins[0], bins[-1])
    ax.legend(loc='upper right')
    ax.set_title('Firing Rate Histogram')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Count')
    return ax


def firing_rate(spikes_df, num_cells=None, time_windows=(0.,), frequency=True):
    """
    Count number of spikes for each cell.
    spikes_df: dataframe of node id and spike times (ms)
    num_cells: number of cells (that determines maximum node id)
    time_windows: list of time windows for counting spikes (second)
    frequency: whether return firing frequency in Hz or just number of spikes
    """
    if not spikes_df['timestamps'].is_monotonic:
        spikes_df = spikes_df.sort_values(by='timestamps')
    if num_cells is None:
        num_cells = spikes_df['node_ids'].max() + 1
    time_windows = 1000. * np.asarray(time_windows).ravel()
    if time_windows.size % 2:
        time_windows = np.append(time_windows, spikes_df['timestamps'].max())
    nspk = np.zeros(num_cells, dtype=int)
    n, N = 0, time_windows.size
    count = False
    for t, i in zip(spikes_df['timestamps'], spikes_df['node_ids']):
        while n < N and t > time_windows[n]:
            n += 1
            count = not count
        if count:
            nspk[i] = nspk[i] + 1
    if frequency:
        nspk = nspk / (total_duration(time_windows) / 1000)
    return nspk


def total_duration(time_windows):
    return np.diff(np.reshape(time_windows, (-1, 2)), axis=1).sum()


def pop_spike_rate(spike_times, time, frequeny=False):
    t = np.arange(*time)
    t = np.append(t, t[-1] + time[2])
    spike_rate, _ = np.histogram(np.asarray(spike_times), t)
    if frequeny:
        spike_rate = 1000 / time[2] * spike_rate
    return spike_rate


def xcorr_coeff(x, y, max_lag=None, dt=1., plot=True, ax=None):
    x = np.asarray(x)
    y = np.asarray(y)
    xcorr = ss.correlate(x, y) / x.size / x.std() / y.std()
    xcorr_lags = ss.correlation_lags(x.size, y.size)
    if max_lag is not None:
        lag_idx = np.nonzero(np.abs(xcorr_lags) <= max_lag / dt)[0]
        xcorr = xcorr[lag_idx]
        xcorr_lags = xcorr_lags[lag_idx]

    if plot:
        if ax is None:
            _, ax = plt.subplots(1, 1)
        ax.plot(dt * xcorr_lags, xcorr)
        ax.set_xlabel('Lag')
        ax.set_ylabel('Cross Correlation')
    return xcorr, xcorr_lags


def get_stim_cycle(on_time=0.5, off_time=0.5, t_start=0., t_stop=t_stop):
    """Get burst input stimulus parameters, (duration, number) of cycles"""
    t_cycle = on_time + off_time
    n_cycle = int(np.floor((t_stop + off_time - t_start) / t_cycle))
    return t_cycle, n_cycle


def get_seg_on_stimulus(x, fs, on_time, off_time,
                        t_start, t=t_stop, tseg=None):
    x = np.asarray(x)
    in_dim = x.ndim
    if in_dim == 1:
        x = x.reshape(1, x.size)
    t = np.asarray(t)
    t_stop = t.size / fs if t.ndim else t
    if tseg is None:
        tseg = on_time # time segment length for PSD (second)
    t_cycle, n_cycle = get_stim_cycle(on_time, off_time, t_start, t_stop)

    nfft = int(tseg * fs) # steps per segment
    i_start = int(t_start * fs)
    i_on = int(on_time * fs)
    i_cycle = int(t_cycle * fs)
    nseg_cycle = int(np.ceil(i_on / nfft))
    x_on = np.zeros((x.shape[0], n_cycle * nseg_cycle * nfft))

    for i in range(n_cycle):
        m = i_start + i * i_cycle
        for j in range(nseg_cycle):
            xx = x[:, m + j * nfft:m + min((j + 1) * nfft, i_on)]
            n = (i * nseg_cycle + j) * nfft
            x_on[:, n:n + xx.shape[1]] = xx
    if in_dim == 1:
        x_on = x_on.ravel()

    stim_cycle = {'t_cycle': t_cycle, 'n_cycle': n_cycle,
                  't_start': t_start, 'on_time': on_time,
                  'i_start': i_start, 'i_cycle': i_cycle}
    return x_on, nfft, stim_cycle


def get_psd_on_stimulus(x, fs, on_time, off_time,
                        t_start, t=t_stop, tseg=None):
    x_on, nfft, stim_cycle = get_seg_on_stimulus(
        x, fs, on_time, off_time, t_start, t=t, tseg=tseg)
    f, pxx = ss.welch(x_on, fs=fs, window='boxcar', nperseg=nfft, noverlap=0)
    return f, pxx, stim_cycle


def get_coh_on_stimulus(x, y, fs, on_time, off_time,
                        t_start, t=t_stop, tseg=None):
    xy = np.array([x, y])
    xy_on, nfft, _ = get_seg_on_stimulus(
        xy, fs, on_time, off_time, t_start, t=t, tseg=tseg)
    f, cxy = ss.coherence(xy_on[0], xy_on[1], fs=fs,
        window='boxcar', nperseg=nfft, noverlap=0)
    return f, cxy


def plot_stimulus_cycles(t, x, stim_cycle, dv_n_sigma=5.,
                         var_label='LFP (mV)', ax=None):
    t_cycle, n_cycle = stim_cycle['t_cycle'], stim_cycle['n_cycle']
    t_start, on_time = stim_cycle['t_start'], stim_cycle['on_time']
    i_start, i_cycle = stim_cycle['i_start'], stim_cycle['i_cycle']
    dv = dv_n_sigma * np.std(x[i_start:])
    
    if ax is None:
        _, ax = plt.subplots(1, 1)
    ax.plot(t[:i_start], x[:i_start] + n_cycle * dv, 'k', label='pre stimulus')
    for i in range(n_cycle):
        m = i_start + i * i_cycle
        xx = x[m:m + i_cycle] + (n_cycle - i - 1) * dv
        ax.plot(t[:len(xx)], xx, label=f'stimulus {i + 1:d}')
    ax.axvline(on_time * 1000, color='gray', label='stimulus off')
    ax.set_xlim(0, 1000 * max(1.25 * t_cycle, t_start))
    ax.set_ylim(np.array((-2, n_cycle + 2)) * dv)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel(var_label)
    ax.legend(loc='lower right', frameon=False)


def fit_fooof(f, pxx, aperiodic_mode='fixed', dB_threshold=3., max_n_peaks=10,
              freq_range=None, peak_width_limits=None, report=False,
              plot=False, plt_log=False, plt_range=None, figsize=None):
    if aperiodic_mode != 'knee':
        aperiodic_mode = 'fixed'
    def set_range(x, upper=f[-1]):        
        x = np.array(upper) if x is None else np.array(x)
        return [f[2], x.item()] if x.size == 1 else x.tolist()
    freq_range = set_range(freq_range)
    peak_width_limits = set_range(peak_width_limits, np.inf)

    # Initialize a FOOOF object
    fm = FOOOF(peak_width_limits=peak_width_limits, min_peak_height=dB_threshold / 10,
               peak_threshold=0., max_n_peaks=max_n_peaks, aperiodic_mode=aperiodic_mode)
    # Fit the model
    fm.fit(f, pxx, freq_range)
    results = fm.get_results()

    if report:
        fm.print_results()
        if aperiodic_mode=='knee':
            ap_params = results.aperiodic_params
            if ap_params[1] <= 0:
                print('Negative value of knee parameter occurred. Suggestion: Fit without knee parameter.')
            knee_freq = np.abs(ap_params[1]) ** (1 / ap_params[2])
            print(f'Knee location: {knee_freq:.2f} Hz')
    if plot:
        plt_range = set_range(plt_range)
        fm.plot(plt_log=plt_log)
        plt.xlim(np.log10(plt_range) if plt_log else plt_range)
        if figsize:
            plt.gcf().set_size_inches(figsize)
        plt.show()
    return results, fm


def plot_channel_psd(psd, channel_id=None, freq_range=200., plt_range=(0, 100.), figsize=(5, 4),
                 aperiodic_mode='knee', dB_threshold=3., max_n_peaks=10, plt_log=True):
    """Plot PSD at given chennel with FOOOF results"""
    plt_range = np.array(plt_range)
    if plt_range.size == 1:
        plt_range = (0, plt_range.item())
    psd_plt = psd.sel(frequency=slice(*plt_range))
    plt.rcParams['axes.prop_cycle'] = plt.cycler('color',
        plt.cm.get_cmap('plasma')(np.linspace(0, 1, psd.coords['channel'].size)))
    plt.figure(figsize=figsize)
    plt.plot(psd_plt.frequency, psd_plt.values.T, label=psd_plt.channel.values)
    plt.xlim(plt_range)
    plt.yscale('log')
    plt.legend(loc='upper right', framealpha=0.2, title='channel ID')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    fig1 = plt.gcf()

    if channel_id is None:
        channel_id = psd.channel[0]
    print(f'Channel: {channel_id: d}')
    psd_plt = psd.sel(channel=channel_id)
    results = fit_fooof(psd_plt.frequency.values, psd_plt.values,
                        aperiodic_mode=aperiodic_mode, dB_threshold=dB_threshold, max_n_peaks=max_n_peaks,
                        freq_range=freq_range, peak_width_limits=None, report=True,
                        plot=True, plt_log=plt_log, plt_range=plt_range[1], figsize=figsize)
    fig2 = plt.gcf()
    return results, fig1, fig2


def psd_residual(f, pxx, fooof_result, plot=False, plt_log=False, plt_range=None, ax=None):
    full_fit, _, ap_fit = gen_model(f[1:], fooof_result.aperiodic_params,
                                    fooof_result.gaussian_params, return_components=True)
    full_fit, ap_fit = 10 ** full_fit, 10 ** ap_fit

    res_psd = np.insert(pxx[1:] - ap_fit, 0, 0.)
    res_fit = np.insert(full_fit - ap_fit, 0, 0.)

    if plot:
        if ax is None:
            _, ax = plt.subplots(1, 1)
        plt_range = np.array(f[-1]) if plt_range is None else np.array(plt_range)
        if plt_range.size == 1:
            plt_range = [f[1] if plt_log else 0., plt_range.item()]
        f_idx = (f >= plt_range[0]) & (f <= plt_range[1])
        ax.plot(f[f_idx], res_psd[f_idx], 'b', label='residual')
        ax.plot(f[f_idx], res_fit[f_idx], 'r', label='fit')
        if plt_log:
            ax.set_xscale('log')
        ax.set_xlim(plt_range)
        ax.legend(loc='upper right')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('PSD Residual')
    return res_psd, res_fit


# cone of influence in frequency for cmorxx-1.0 wavelet
f0 = 2 * np.pi
CMOR_COI = 2 ** -0.5
CMOR_FLAMBDA = 4 * np.pi / (f0 + (2 + f0 ** 2) ** 0.5)
COI_FREQ = 1 / (CMOR_COI * CMOR_FLAMBDA)

def cwt_spectrogram(x, fs, nNotes=6, nOctaves=np.inf, freq_range=(0, np.inf),
                    bandwidth=1.0, axis=-1, detrend=False, normalize=False):
    """Calculate spectrogram using continuous wavelet transform"""
    x = np.asarray(x)
    N = x.shape[axis]
    times = np.arange(N) / fs
    # detrend and normalize
    if detrend:
        x = ss.detrend(x, axis=axis, type='linear')
    if normalize:
        x = x / x.std()
    # Define some parameters of our wavelet analysis. 
    # range of scales (in time) that makes sense
    # min = 2 (Nyquist frequency)
    # max = np.floor(N/2)
    nOctaves = min(nOctaves, np.log2(2 * np.floor(N / 2)))
    scales = 2 ** (1 + np.arange(np.floor(nOctaves * nNotes)) / nNotes)
    # cwt and the frequencies used. 
    # Use the complex morelet with bw=2*bandwidth and center frequency of 1.0
    # bandwidth is the sigma of the gaussian envelope
    wavelet = 'cmor' + str(2 * bandwidth) + '-1.0'
    frequencies = pywt.scale2frequency(wavelet, scales) * fs
    scales = scales[(frequencies >= freq_range[0]) & (frequencies <= freq_range[1])]
    coef, frequencies = pywt.cwt(x, scales[::-1], wavelet=wavelet, sampling_period=1 / fs, axis=axis)
    power = np.real(coef * np.conj(coef)) # equivalent to power = np.abs(coef)**2
    # cone of influence in terms of wavelength
    coi = N / 2 - np.abs(np.arange(N) - (N - 1) / 2)
    # cone of influence in terms of frequency
    coif = COI_FREQ * fs / coi
    return power, times, frequencies, coif


def cwt_spectrogram_xarray(x, fs, axis=-1, downsample_fs=None, channel_coords=None, **cwt_kwargs):
    """Calculate spectrogram using continuous wavelet transform and return an xarray.Dataset
    x: input array
    fs: sampling frequency
    axis: dimension index of time axis in x
    downsample_fs: downsample to the frequency if specified
    channel_coords: dictionary of {coordinate name: index} for channels
    cwt_kwargs: keyword arguments for cwt_spectrogram()
    """
    x = np.asarray(x)
    T = x.shape[axis] # number of time points
    t = np.arange(T) / fs
    if downsample_fs is None or downsample_fs >= fs:
        downsample_fs = fs
        downsampled = x
    else:
        num = int(T * downsample_fs / fs)
        downsample_fs = num / T * fs
        downsampled, t = ss.resample(x, num=num, t=t, axis=axis)
    downsampled = np.moveaxis(downsampled, axis, -1)
    sxx, _, f, coif = cwt_spectrogram(downsampled, downsample_fs, **cwt_kwargs)
    sxx = np.moveaxis(sxx, 0, -2) # shape (... , freq, time)
    if channel_coords is None:
        channel_coords = {f'dim_{i:d}': range(d) for i, d in enumerate(sxx.shape[:-2])}
    sxx = xr.DataArray(sxx, coords={**channel_coords, 'frequency': f, 'time': t}).to_dataset(name='PSD')
    sxx = sxx.assign(cone_of_influence_frequency=xr.DataArray(coif, coords={'time': t}))
    return sxx

def plot_spectrogram(sxx_xarray, remove_aperiodic=None, plt_log=False,
                     plt_range=None, clr_freq_range=None, ax=None):
    """Plot spectrogram. Determine color limits using value in frequency band clr_freq_range"""
    sxx = sxx_xarray.PSD.values.copy()
    t = sxx_xarray.time.values.copy()
    f = sxx_xarray.frequency.values.copy()

    cbar_label = 'PSD' if remove_aperiodic is None else 'PSD Residual'
    if plt_log:
        with np.errstate(divide='ignore'):
            sxx = np.log10(sxx)
        cbar_label += ' log(power)'

    if remove_aperiodic is not None:
        f1_idx = 0 if f[0] else 1
        ap_fit = gen_aperiodic(f[f1_idx:], remove_aperiodic.aperiodic_params)
        sxx[f1_idx:, :] -= (ap_fit if plt_log else 10 ** ap_fit)[:, None]
        sxx[:f1_idx, :] = 0.

    if ax is None:
        _, ax = plt.subplots(1, 1)
    plt_range = np.array(f[-1]) if plt_range is None else np.array(plt_range)
    if plt_range.size == 1:
        plt_range = [f[0 if f[0] else 1] if plt_log else 0., plt_range.item()]
    f_idx = (f >= plt_range[0]) & (f <= plt_range[1])
    if clr_freq_range is None:
        vmin, vmax = None, None
    else:
        c_idx = (f >= clr_freq_range[0]) & (f <= clr_freq_range[1])
        vmin, vmax = sxx[c_idx, :].min(), sxx[c_idx, :].max()

    f = f[f_idx]
    pcm = ax.pcolormesh(t, f, sxx[f_idx, :], shading='gouraud', vmin=vmin, vmax=vmax)
    if 'cone_of_influence_frequency' in sxx_xarray:
        coif = sxx_xarray.cone_of_influence_frequency
        ax.plot(t, coif)
        ax.fill_between(t, coif, step='mid', alpha=0.2)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(f[0], f[-1])
    plt.colorbar(mappable=pcm, ax=ax, label=cbar_label)
    ax.set_xlabel('Time (sec)')
    ax.set_ylabel('Frequency (Hz)')
    return sxx
