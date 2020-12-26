import pandas as pd
import os.path as osp


def load_data_csv(file_path, col_prefix=None, date_col='Date'):
    """
    Loads the data by the regular format and takes relevant columns
    :param file_path: relative or absolute path of csv
    :param col_prefix: if want to add some prefix before each column name
    :param date_col: name of the date column
    :return: tne loaded df
    """
    df = pd.read_csv(file_path, parse_dates=[date_col])
    df = df.set_index(date_col)
    df.columns = [col.strip() for col in df.columns]
    df = df[['hs', 'dir', 'Tm']]
    df = df.astype('float')
    if col_prefix is not None:
        df.columns = [col_prefix+'_'+col for col in df.columns]
    return df


def load_all_data(file_names_and_prefix_list, data_dir):
    """
    Load all data files, and merge them into one dataframe
    assumes - "Date" is the time column. and takes only times that
    have data for all (takes the intersection)
    :param file_names_and_prefix_list: tuples of file name and the
    appropriate prefix to column of that file's data. since the data from all files
    is similar, we use the prefixes to know the origin and data source
    :param data_dir: path to data files
    :return: dataframe of all loaded data, with prefixes. records appearing are only those that had data in all
    desired files
    """
    dfs = []
    for file_name, col_prefix in file_names_and_prefix_list:
        dfs.append(load_data_csv(osp.join(data_dir, file_name), col_prefix, date_col='Date'))
    df = pd.concat(dfs, join='inner', axis=1)
    return df


def get_df_for_model(data_df, col_names_and_offsets):
    """
    Prepare data so it will include wanted columns and be aligned as desired
    :param data_df: the original df after loading, before choosing and aligning columns. it is assumed that data is
    ordered chronologically (by date column)
    :param col_names_and_offsets: series, in which the index indicate which columns should be included
    and the value for each is the desired alignment offset.
    The value the desired delta between records, when the "unit" is a sample/record (and not a time offset).
    The translation between that and the matching time offset should be a prior calculation, taking into account
    the sampling resolution
    :return: The resulting df after alignment
    """
    srs = []
    max_offset = col_names_and_offsets.max()
    # if minimum offset is not 0 this is weird, we should just reduce the minimum from all
    assert (0 == col_names_and_offsets.min())
    # if max offset is more than shape than we cannot realign in desired offset
    assert (max_offset < data_df.shape[0])
    # verify chronological order, assuming index has dates
    assert(list(data_df.index) == sorted(list(data_df.index)))

    # if no realignment needed, just return the dataframe with desired columns
    if max_offset == 0:
        df = data_df[list(col_names_and_offsets.index)]
    # if realignment needed, go column by column and realign accordingly
    else:
        for data_col, offset in zip(col_names_and_offsets.index, col_names_and_offsets):
            offset = int(offset)
            if 0 == offset:
                # take dates_index from the original, at offset 0
                # assertion above makes sure there is at least on like that so it will necessarily be initialized
                dates_index = data_df.iloc[:-max_offset].index
                srs.append(data_df[data_col].iloc[:-max_offset].reset_index(drop=True))
            elif max_offset == offset:
                srs.append(data_df[data_col].iloc[max_offset:].reset_index(drop=True))
            else:  # some offset between 0 and max
                srs.append(data_df[data_col].iloc[offset:-(max_offset - offset)].reset_index(drop=True))
        df = pd.concat(srs, axis=1)
        df.index = dates_index

    return df
