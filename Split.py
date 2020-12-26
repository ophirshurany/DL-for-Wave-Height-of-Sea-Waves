import math


def split_train_test_val(df, train_ratio=0.8, test_ratio=0.1, phys_target=None):
    """
    split df into train, val, test according to ratios.
    split takes last part, makes it test, before last is val, and train is first part
    phys_target series for test data area is also returned if optional parameter is received
    :param df: data
    :param train_ratio: out of 1 - how much of data is train
    :param test_ratio: out of 1 - how much of data is test
    :param phys_target: series of physical model data (WW3), prediction for the specific time
    (original prediction, before changing alignments. kept separately for evaluation purposes)
    :return: tuple with df for each of the relevant parts (and physical_model series for test data also, if received.
    otherwise None)
    """
    phys_test = None
    assert(train_ratio + test_ratio <= 1)
    num_samples = df.shape[0]
    train_size = round(num_samples * train_ratio)
    test_size = round(num_samples * test_ratio)
    val_size = num_samples - (train_size + test_size)
    train = df.iloc[:train_size]
    val = df.iloc[train_size:train_size + val_size]
    test = df.iloc[train_size + val_size:]
    if phys_target is not None:
        phys_test = phys_target.iloc[train_size + val_size:]
    return train, val, test, phys_test


def kfold_split_train_test(df, fold_num, k=5, phys_target=None):
    """
    Split df into train, val, test.
    The data is split into "k" (num of total folds) distinct parts.
    Then, "fold_num" index indicates which part is the one to be chosen as the data for the val and test.
    this part is split equally into val and test, and all the rest is returned as training data.
    if fold_num is 0 or k-1 then train data will be returned as one df. otherwise (0 < fold_num < k-1),
    means that train data is not continuous and thus a list of two items
    (df of train data before val and test data fold, and df of data after them) will be returned.
    :param df: data
    :param fold_num: index of desired option/configuration (train/val/test split) out of the k folds
    fold_num has to be between 0 and k-1
    :param k: num of folds the data split calculation considers (meaning there will be k options for
    folds without intersection)
    :param phys_target: series of physical model data (WW3), prediction for the specific time
    (original prediction, before changing alignments. kept separately for evaluation purposes)
    :return: tuple with df for each of the relevant parts (and physical_model series for test data and val data also,
    if received. otherwise None) train_data will be list of two dfs, if of consists of 2 discontinuous parts, otherwise
    single df.
    """
    num_samples = df.shape[0]
    part_size = math.floor(num_samples / k)
    val_start = part_size * fold_num
    val = df.iloc[val_start:val_start + math.floor(part_size / 2)]
    test = df.iloc[val_start + math.floor(part_size / 2):val_start + part_size]
    train_1 = df.iloc[:val_start]
    train_2 = df.iloc[val_start + part_size:]
    train = [t for t in [train_1, train_2] if t.shape[0] > 0]
    # if actually have only one train_df, return it as is, not as list
    if 1 == len(train):
        train = train[0]
    if phys_target is not None:
        phys_test = phys_target.iloc[val_start + math.floor(part_size / 2):val_start + part_size]
        phys_val = phys_target.iloc[val_start:val_start + math.floor(part_size / 2)]
    else:
        phys_val = None
        phys_test = None
    return train, val, test, phys_val, phys_test
