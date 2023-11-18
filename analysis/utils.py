
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


def population_statistics(data, stats={'mean': np.mean, 'stdev': np.std}):
    """Get population statistics in data
    data: dict('pop_name': data_array). Dictionary of data for populations.
    stats: dict('statistics': function). Dictionary of statistics name and function pairs.
    """
    df = pd.DataFrame({name: map(func, data.values()) for name, func in stats.items()},
                      index=pd.Index(data.keys(), name='pop_name'))
    return df
