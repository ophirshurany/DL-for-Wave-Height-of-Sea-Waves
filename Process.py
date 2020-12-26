from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import numpy as np

import itertools as it


# noinspection SpellCheckingInspection
class PreprocessData:
    """
    All feature preprocessing - sequence building from streams, feature normalization, wave direction engineering etc.
    is implemented here. to be used both for training transformations and for transforming test data,
    after class is built on train data (and constants are saved internally for these purposes)
    """

    def __init__(self, **kwargs):
        """
        initializing the scalers that will be saved and the stream constants
        :param kwargs: the arguments for the sequence building, that will be sent to the internal PredConsts class
        """
        self.scaler = None
        self.target_scaler = None
        self.c = self.PredConsts(**kwargs)

    class PredConsts:
        def __init__(self, steps_back=1, y_length=1, step_size=1, gap_forward=0):
            """
            Constants for sequences for model
            :param steps_back: how many samples back is each prediction fed
            :param y_length: length of forecast (does each prediction return single forecast or stream of size > 1)
            currently only y_length=1 supported
            :param step_size: how many data points to jump between every sample fed to model
            :param gap_forward: how far into the future is the forecast desired for (unit is number of samples forward)
            """
            self.look_back = steps_back
            if y_length > 1:
                raise NotImplementedError
            self.y_length = y_length
            self.step_size = step_size
            self.gap_fwd = gap_forward

    def get_forward_pred_num_steps(self):
        """
        :return: number of steps forward for which the prediction is given
        (unit is umber of samples forward, so translation to time should be according to data sampling resolutions)
        """
        return self.c.gap_fwd

    def create_seq_from_stream(self, df, target_data=None):
        """
        This function takes sequential data and builds sequences from it -
        essentially converting an array of values into a dataset matrix
        if used for training or checking model, it will build sequences and corresponding predictions, according to
        gap_forward
        If used for giving predictions, it will build only X_data, and predictions will be given from trained model
        :param df: the original data points
        :param target_data: the data for the prediction target
        :return: Tuple of X_data, y_data, target time index:
                X_data - input data for prediction (built into sequences according to PredConsts class),
                y_data - target data for prediction (if given as input), built into the relevant sample dimensions,
                target time index - index (of dates) which is aligned by the prediction target data
        """
        y_data = None
        # initializing to zero, if no target necessary should stay zero
        gap_fwd_for_pred = 0
        # for working with the sequences in a numpy array format
        data = df.values

        # note that might be more efficient memory-wise to switch to np.array(df.index). (not sure)
        target_dates_index = df.index

        if target_data is not None:
            # if no need for target data, can predict till end of data,
            # no need to save buffer forward, of size "gap_fwd" representing prediction look ahead
            gap_fwd_for_pred = self.c.gap_fwd

        # how much of data range window is "used" by the relevant lengths (look_back is used prior to actual sample
        # and y_length and gap_fwd used forward. no need to count the initial step of gap_fwd and y_length twice,
        # that's why the "-1" after the gap_fwd_for_pred
        distance_needed = self.c.look_back + self.c.y_length + (gap_fwd_for_pred - 1)
        # since we use it later for "range" function, both are numbers on a count based 1 index (max_seq_opt_range,
        # which is calculated based on len(data))
        max_seq_opts_range = int(len(data) - distance_needed)

        # number of sequences is all that fit for starting in that range, given the chosen step size
        num_seq_opts = len(range(0, max_seq_opts_range, self.c.step_size))

        X_data = [data[i:(i + self.c.look_back)] for i in range(0, max_seq_opts_range, self.c.step_size)]
        # reshape input to be [samples, look_back, features_dimension] (look_back is number of time steps back)
        X_data = np.reshape(X_data, (num_seq_opts, self.c.look_back, data.shape[1]))

        # if given target data, form the relevant data order as well
        if target_data is not None:
            # we assume target_data is time-aligned like the X_data, and thus the target data is taken starting from
            # gap_fwd steps forward of the last sample in the X_data
            # note that if y_length  != 1 some reshaping will probably be needed afterwards, like for the X_data.
            # currently we just use "squeeze" instead since when y_length=1, we just have 1 sample for every time
            y_data = [target_data[(i + self.c.look_back + self.c.gap_fwd):
                                  (i + self.c.look_back + self.c.gap_fwd + self.c.y_length)].squeeze()
                      for i in range(0, max_seq_opts_range, self.c.step_size)]
            # as written above currently only length of 1 is supported
            assert(1 == self.c.y_length)
            y_data = np.array(y_data).squeeze()

        # saving the time indices, to later be able to align (time-wise) the predictions
        target_dates_index = [target_dates_index[(i + self.c.look_back + self.c.gap_fwd):
                                                 (i + self.c.look_back + self.c.gap_fwd + self.c.y_length)].values[0]
                              for i in range(0, max_seq_opts_range, self.c.step_size)]

        return X_data, y_data, target_dates_index

    def scale_hs(self, df, height_cols):
        """
        Normalize the significant wave height data
        :param df: df with data
        :param height_cols: the names of the relevant columns, with the Hs, for normalizing
        :return: normalized data
        """
        return self.scaler.transform(df[height_cols])

    def inverse_scale_hs(self, df, height_cols):
        """
        Inverse normalization of the significant wave height data
        :param df: df with data
        :param height_cols: the names of the relevant columns, with the Hs, for normalizing
        :return: data in original scale (inverse the normalization)
        """
        return self.scaler.inverse_transform(df[height_cols])

    def scale_target(self, df, col_num=None):
        """
        Normalize target data (significant wave height for target)
        :param df: input data. can be given in the form of a series, or a dataframe with one column (essentially similar
        to a series, but code-wise treated in a different manner, or as a column within a whole df, which will then
        need to be accompanied by the index of the target data column)
        :param col_num: optional, index of target data column in the df, if a df with additional data is given
        :return: normalized data
        """
        if pd.Series == type(df):
            return self.target_scaler.transform(df.to_frame())
        elif df.shape[1] == 1:
            return self.target_scaler.transform(df)
        else:
            if col_num:
                return self.target_scaler.transform(df.iloc[:, [col_num]])
            else:
                raise IndexError

    def inverse_scale_target(self, df, col_num=None):
        """
        Inverse normalization of target data (significant wave height for target)
        Normalize target data (significant wave height for target)
        :param df: input data. can be given in the form of a series, or a dataframe with one column (essentially similar
        to a series, but code-wise treated in a different manner, or as a column within a whole df, which will then
        need to be accompanied by the index of the target data column)
        :param col_num: optional, index of target data column in the df, if a df with additional data is given
        :return: data in original scale (inverse the normalization)
        """
        if pd.Series == type(df):
            return self.target_scaler.inverse_transform(df.to_frame())
        elif df.shape[1] == 1:
            return self.target_scaler.inverse_transform(df)
        else:
            if col_num:
                return self.target_scaler.inverse_transform(df.iloc[:, [col_num]])
            else:
                raise IndexError

    def fit(self, data, target_data):
        """
        fit the preprocessor - thus - initializes and fits the data scalers (built for normalization purposes)
        assumes that the names of columns containing the significant wave height data end with "hs"
        handles dataframe or list of dataframes, and similarly series or list of series
        :param data: a dataframe of data for prediction (input data), or a list of dataframes
        :param target_data: a Series of the prediction target labels,
        or a list of series (corresponding to the data format)
        :return: None (just fits the preprocessor)
        """

        # if input data is given in a list (will happen when input data is split into two - before and after
        # validation and test data. a patch used for cross validation when don't have a lot of data, explained
        # elaborately in project documentation
        if list == type(data):
            assert(len(data) == len(target_data))
            # just for scaling purposes, flatten out list to one whole data frame
            data = pd.concat(data, axis=0)
            # same for target data
            target_data = pd.concat(target_data, axis=0)

        # save names of columns with Hs data
        wave_cols = [col for col in data.columns if col.endswith("hs")]
        if 0 != len(wave_cols):
            self.scaler = MinMaxScaler(feature_range=(-0.5, 0.5))
            self.scaler.fit(data[wave_cols])

        # initialized target data separately (might be or not be in input data, more general to save a separate scaler)
        # and the overhead is negligible
        self.target_scaler = MinMaxScaler(feature_range=(-0.5, 0.5))
        self.target_scaler.fit(target_data)

    def fold_dir_and_normalize(self, df_dir, dir_cols):
        """
        Create "folded" angle representation. angles will be transformed to represent their distance from 180
        (to conserved relative relations. for example angle of 160 and 200 will have the same number and same for
        10 and 350. Calculate and normalize the new angle. and add additional binary feature representing what was
        the initial direction (below or above 180 degrees)
        :param df_dir: df with wave direction data
        :param dir_cols: list of names of columns with direction (angle) data
        :return: A df with the new direction features, for the "folded" angle and binary direction
        representation. The new columns are prefixed with 'angle_' and 'direction_'.
        """
        ANG_PREF = 'angle_'
        DIR_PREF = 'direction_'

        for col in dir_cols:
            # fold (angles up to 180 stay the same, and between 180 and 360 are represented by their distance from 360)
            # nd normalize - between -0.5 and 0.5
            df_dir[ANG_PREF + col] = df_dir[col].apply(lambda x: (min(x, 360 - x)/180) - 0.5)
            # binary representation of which was chosen (was angle between 0 and 180 or above)
            df_dir[DIR_PREF + col] = np.select([df_dir[col] <= 180, df_dir[col] > 180], [1, 0], default=np.nan)
        # make a list for column reordering - to be ordered by original angle data (for each one - angle and binary
        # direction representation)
        col_names_order = list(it.chain.from_iterable([(ANG_PREF + col, DIR_PREF + col) for col in dir_cols]))
        # save only the relevant columns (without the original angles)
        df_dir = df_dir[col_names_order]
        return df_dir

    def apply_sin_and_cosine_to_dir(self, df_dir, dir_cols):
        """
        Transform direction angle and calculate from it two features - representing sin and cos of angle.
        :param df_dir: df with wave direction data
        :param dir_cols: list of names of columns with direction (angle) data
        :return: A df with the new direction features, for the sin and cos of the angles. The new columns
        are prefixed with 'sin_' and 'cos_'.
        """
        SIN_PREF = 'sin_'
        COS_PREF = 'cos_'

        # numpy sin and cos accept degree in radians
        df_dir[[SIN_PREF + col for col in dir_cols]] = df_dir[dir_cols].apply(
            lambda x: np.sin(np.deg2rad(x)))
        df_dir[[COS_PREF + col for col in dir_cols]] = df_dir[dir_cols].apply(
            lambda x: np.cos(np.deg2rad(x)))

        col_names_order = list(it.chain.from_iterable([(SIN_PREF + col, COS_PREF + col) for col in dir_cols]))
        df_dir = df_dir[col_names_order]
        return df_dir

    def apply_dir_transform(self, df_dir, dir_cols, is_fold_transform=True):
        """
        Choose which transformation to apply on direction angle, and call relevant function
        :param df_dir: df with wave direction data
        :param dir_cols: list of names of columns with direction (angle) data
        :param is_fold_transform: True if prefer the "linear fold" one ("fold around 180"), and False if want
        sin-cos transformation
        :return: A dataframe with the new engineered columns representing the angle
        (two new columns per original angle column)
        their order in the df
        """
        if is_fold_transform:
            return self.fold_dir_and_normalize(df_dir, dir_cols)
        else:
            return self.apply_sin_and_cosine_to_dir(df_dir, dir_cols)

    def transform_single_df(self, df, target_data=None, is_angle_fold_transform=True):
        """
        Scales waves, transforms directions and builds sequences, from single df
        :param df: the input data df for training/inference
        :param target_data: optional, the prediction target data, given as a series (if for training, otherwise None)
        :param is_angle_fold_transform: True for "fold" transform on angles, False for sin/cos transform
        :return: a tuple - X_data, y_data, times_data. X_data is a numpy array of dimensions
        [samples, look_back, features_dimension], y_data is labels which will be returned if target_data is given
        (otherwise returned as None), and times_data is a index of datetime objects representing the times for which
        the sequences should return the predictions
        """

        # assumes these are the suffixes for the significant wave height and wave direction data columns
        height_cols = [col for col in df.columns if col.endswith("hs")]
        dir_cols = [col for col in df.columns if col.endswith("dir")]

        # if we want transform inplace we can delete this. its here in order not to change original data
        df = df.copy()

        # suppress SettingWithCopyWarning since it's a false and irrelevant warning in our scenario
        original_warning_setting = pd.options.mode.chained_assignment
        pd.options.mode.chained_assignment = None

        columns_for_model = []
        if target_data is not None:
            target_data = self.scale_target(target_data)

        # scale Hs columns if there are any
        if 0 != len(height_cols):
            df[height_cols] = self.scale_hs(df, height_cols)
            columns_for_model += height_cols
        # transform direction colums if there are any
        if 0 != len(dir_cols):
            df_dir = self.apply_dir_transform(df[dir_cols], dir_cols, is_angle_fold_transform)
            # add new columns to df
            dir_new_columns = list(df_dir.columns)
            df[dir_new_columns] = df_dir
            # add columns to those that will be used
            columns_for_model += dir_new_columns

        # return original SettingWithCopyWarning
        pd.options.mode.chained_assignment = original_warning_setting

        # take only desired columns, and by this order
        df = df[columns_for_model]

        # get the data in the structure that will be useful for model
        X_data, y_data, dates_y = self.create_seq_from_stream(df, target_data)
        return X_data, y_data, dates_y

    def transform(self, data, target_data=None, is_angle_fold_transform=True):
        """
        Transform original df with data samples and target data to X and y ready for training
        (This includes: transform angles, normalize data, build sequences, and if necessary - handle df inputs as list
        of dfs). This is a wrapper function for transform_single_df, which handles also df lists if needed
        :param data: the input data for the model. Can be given as single dataframe, if all samples are
        continuous and consecutive in time. or as a list of df's each for the relevant time span
        :param target_data: a series with the labels (if transform used for training), otherwise None.
        Should be aligned with "data" param, meaning each sample represents the data of the prediction target for the
        corresponding data sample in "data" (and likewise can be given as single series or list of series).
        :param is_angle_fold_transform: True for "fold" transform on angles, False for sin/cos transform
        :return: a tuple - X_data, y_data, times_data. X_data is a numpy array of dimensions
        [samples, look_back, features_dimension], y_data is labels which will be returned if target_data is given
        (otherwise returned as None), and times_data is a index of datetime objects representing the times for which
        the sequences should return the predictions
        """
        all_data = []
        if list == type(data):
            if target_data is not None:
                assert (len(data) == len(target_data))
            else:
                target_data = [None]*len(data)
            for df, target_df in zip(data, target_data):
                all_data.append(self.transform_single_df(df, target_df))
            X_data = np.concatenate([d[0] for d in all_data], axis=0)
            if target_data is not None:
                y_data = np.concatenate([d[1] for d in all_data], axis=0)
            else:
                y_data = None
            # concat just the times with data (and not empty list, cause then concat failes in this case)
            times_data = np.concatenate([d[2] for d in all_data if len(d[2]) > 0], axis=0)
        else:
            X_data, y_data, times_data = self.transform_single_df(data, target_data, is_angle_fold_transform)
        return X_data, y_data, times_data

    def fit_transform(self, data, target_data, is_angle_fold_transform=True):
        """
        fit and transform, relevant for train data.
        all preprocessing class parameters will be saved by fit
        and transformed version of data returned, ready for model
        :param data: a dataframe of data for prediction (input data), or a list of dataframes
        :param target_data: a Series of the prediction target labels,
        or a list of series (corresponding to the data format)
        :param is_angle_fold_transform: True for "fold" transform on angles, False for sin/cos transform
        :return: transformed data, ready for model
        """
        self.fit(data, target_data)
        return self.transform(data, target_data, is_angle_fold_transform)
