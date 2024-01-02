import numpy as np
import pandas as pd


def df2node_id(df):
    """Get node ids from a node dataframe into a list"""
    return df.index.tolist()


def get_pop(node_df, pop_name):
    """Get nodes with given population name from the nodes dataframe"""
    return node_df.loc[node_df['pop_name'] == pop_name]


def get_pop_id(node_df, pop_name):
    """Get node ids with given population name from the nodes dataframe"""
    return df2node_id(get_pop(node_df, pop_name))


def get_populations(node_df, pop_names, only_id=False):
    """Get node dataframes of multiple populations from the nodes dataframe"""
    func = get_pop_id if only_id else get_pop
    return {p: func(node_df, p) for p in pop_names}


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


def population_statistics(data, stats={'mean': np.mean, 'stdev': np.std}):
    """Get population statistics in data
    data: dict('pop_name': data_array). Dictionary of data for populations.
    stats: dict('statistics': function). Dictionary of statistics name and function pairs.
    """
    df = pd.DataFrame({name: map(func, data.values()) for name, func in stats.items()},
                      index=pd.Index(data.keys(), name='pop_name'))
    return df
