import pandas as pd
import tensorflow as tf

import Load
import Split
import Process
import Models
import Eval
import TestInstanceParams

def downsample_data(data, downsampling_factor=1):
    """
    returns a downsamples dataframe - for increasing the time difference between every two samples
    :param data: dataframe
    :param downsampling_factor: new dataframe will have about 1/downsampling_factor samples
    default is 1 - thus not changing the original dataframe
    :return: downsampled dataframe
    """
    return data.iloc[range(0, data.shape[0], downsampling_factor)]

def get_feature_and_target_data(data, target_col_name, is_target_in_features=True):
    """
    :param data: dataframe with all relevant columns for model, including target column,
    or list of several dataframes (that are not continuous in time thus given as a list)
    :param target_col_name: name of target column (for forecasting)
    :param is_target_in_features: binary - is the data of the target location also given as input to the model
    :return: tuple - first item being the dataframe for training (without target data if shouldn't be there),
    and second item is the target
    if data was given as list then both will return as lists, if given as is then will return directly
    """
    if type(data) == list:
        target = [d[[target_col_name]] for d in data]
        if not is_target_in_features:
            data = [d.drop(target_col_name, axis=1) for d in data]
    else:
        target = data[[target_col_name]]
        if not is_target_in_features:
            data = data.drop(target_col_name, axis=1)
    return data, target


def run_single_fold_train_test(df, phys_target, run_params, pre, curr_fold_num):
    """
    Train, predict, and calculate eval metrics, for model. on a single data fold.
    This function receives the data, takes care of splitting it to folds, trains the model and returns the results
    for the fold (index) it ran on.
    :param df: dataframe or list of dataframes. all columns are those that will be used for training
    :param phys_target: series with the physical model of the target data (for eval purposes)
    :param run_params: instance of type TestInstanceParams class, holds relevant model configurations
    :param pre: instance of type Process - for data preprocessing
    :param curr_fold_num: The index of the relevant fold number. has to be an int between (and including)
    0 and run_params.k -1.
    :return: a dictionary with the model, the predicitons and ground truth for the test, validation and train datasets,
    and for the validation and test also the physical model predictions
    and a dataframe summarizing the evaluation metrics for the fold, for the train, validation and test sets
    """
    fold_dict = {}
    fold_dict["fold_num"] = curr_fold_num
    train, val, test, phys_val, phys_test = Split.kfold_split_train_test(df, curr_fold_num,
                                                                         k=run_params.k, phys_target=phys_target)
    pre.fit(*get_feature_and_target_data(
        train, run_params.target_col, run_params.is_target_in_input))
    fold_dict["preprocess"] = pre
    X_train, y_train, dates_y_train = pre.transform(
        *get_feature_and_target_data(train, run_params.target_col, run_params.is_target_in_input))
    X_val, y_val, dates_y_val = pre.transform(
        *get_feature_and_target_data(val, run_params.target_col, run_params.is_target_in_input))
    X_test, y_test, dates_y_test = pre.transform(
        *get_feature_and_target_data(test, run_params.target_col, run_params.is_target_in_input))
    input_dim = X_train.shape[2]
    model_structure_args = {"look_back": run_params.train_steps, "input_dimension": input_dim,
                            "build_config_description": run_params.desc_str + "_f{}".format(curr_fold_num)}

    fold_dict["train"] = {}
    fold_dict["val"] = {}
    fold_dict["test"] = {}

    fold_dict["train"]["dates"] = dates_y_train
    fold_dict["val"]["dates"] = dates_y_val
    fold_dict["test"]["dates"] = dates_y_test

    with tf.device("/cpu:0"):
        curr_model = run_params.model_class(**model_structure_args)

    with tf.device("/cpu:0"):
        # train model (and save it, if this was implemented in model class)
        curr_model = curr_model.fit(X_train, y_train, val_data=(X_val, y_val), **run_params.model_args)

    fold_dict["model"] = curr_model

    fold_dict["test"]["pred"] = pre.inverse_scale_target(fold_dict["model"].predict(X_test))
    fold_dict["test"]["true"] = pre.inverse_scale_target(y_test.reshape(-1, 1))
    fold_dict["test"]["ww3"] = phys_test.iloc[run_params.train_steps + run_params.pred_forward:].values.reshape(-1, 1)

    fold_dict["val"]["pred"] = pre.inverse_scale_target(fold_dict["model"].predict(X_val))
    fold_dict["val"]["true"] = pre.inverse_scale_target(y_val.reshape(-1, 1))
    fold_dict["val"]["ww3"] = phys_val.iloc[run_params.train_steps + run_params.pred_forward:].values.reshape(-1, 1)

    fold_dict["train"]["pred"] = pre.inverse_scale_target(fold_dict["model"].predict(X_train))
    fold_dict["train"]["true"] = pre.inverse_scale_target(y_train.reshape(-1, 1))

    fold_dict["results_test"] = Eval.eval_pred_phys_const(fold_dict["test"], pre)
    fold_dict["results_val"] = Eval.eval_pred_phys_const(fold_dict["val"], pre)
    # for train we don't look at ww3 model or const guess. these metrics are interesting
    # only for checking overfit in training
    train_eval = Eval.eval_model(
        fold_dict["train"]["true"], fold_dict["train"]["pred"])
    fold_dict["results_train"] = pd.Series(train_eval, name="ML")
    return fold_dict


def run_kfold_train_test(df, phys_target, run_params, pre):
    """
    Runs all relevant folds (number given in run_params)
    :param df: ready dataframe with columns for training
    :param phys_target: series of physical model data
    :param run_params: instance of TestInstanceParams class for all model configurations
    :param pre: Initialized (but not yet fitted) instance of Processing class, for data preprocessing
    :return: dictionary with results for each of the folds (as returned by run_single_fold_train_test
    and combined dataframe with metric evaluation results for test, val, train for all folds
    """
    folds_run_data = {}
    folds_run_data["run_params"] = run_params
    results_test = []
    results_val = []
    results_train = []
    folds_to_run_on = list(range(run_params.k))
    if run_params.num_folds_to_run:
        # if num_folds_to_run < k, prefer running on last folds
        folds_to_run_on = folds_to_run_on[-run_params.num_folds_to_run:]
    folds_run_data["folds_dict"] = {}
    for i in folds_to_run_on:
        print("##### Running on fold {} #####".format(i))
        curr_fold_results = run_single_fold_train_test(df, phys_target, run_params, pre, i)
        folds_run_data["folds_dict"][i] = curr_fold_results
        results_test.append(folds_run_data["folds_dict"][i]["results_test"].assign(fold=i))
        results_val.append(folds_run_data["folds_dict"][i]["results_val"].assign(fold=i))
        results_train.append(folds_run_data["folds_dict"][i]["results_train"].to_frame().assign(fold=i))
    results_test = pd.concat(results_test)
    results_val = pd.concat(results_val)
    results_train = pd.concat(results_train)
    results_test = results_test.set_index(['fold', results_test.index])
    results_val = results_val.set_index(['fold', results_val.index])
    results_train = results_train.set_index(['fold', results_train.index])
    folds_run_data["results_test"] = results_test
    folds_run_data["results_val"] = results_val
    folds_run_data["results_train"] = results_train
    return folds_run_data


def prepare_and_run(data, *, target_col, col_names_and_offsets, input_data_str_repr,
                   pred_forward_hrs=4, look_back_hrs=12, time_sample_res_minutes=10,
                   num_folds_to_run=5, k=5):
    """
    E2E functionality: given the data and the desired configurations -
    runs end to end process (prepares data, trains model, uses model to predict and evaluates performance, for
    desired data folds)
    The result is the dict with results (predictions and metrics), for all of the folds,
    as returned by kfold_train_test function
    :param data: dataframe with all relevant data (can include also unecessary columns)
    (like Load.load_all_data function returns)
    :param target_col: the name of the column for which we are forecasting
    :param col_names_and_offsets: series with names of all columns that should be in training data, and desired
    offset in hours (if one is desired, otherwise 0)
    :param input_data_str_repr: string representation of the input data (representing the index
    in col_names_and_offsets). this will be used in the filenames when saving the model, results, etc
    :param pred_forward_hrs: how many hours forward are we forecasting "target_col" for
    :param look_back_hrs: how many hours of data backwards is the model fed
    :param time_sample_res_minutes: what is the desired sampling resolution of the data
    (currently the files we have are every 10 minutes)
    :param num_folds_to_run: for the cross validation, how many out of the k folds that the data is split into should
    be used
    :param k: how many folds to split the data to
    :return: the dictionary with all the results (both the numbers and the metrics), as returned from
    run_kfold_train_test function
    """

    assert(num_folds_to_run > 0 and num_folds_to_run <= k)
    # this is currently hard coded here, can of course also be changed and then received as part of model input
    model_train_args = {"num_epochs": 16, "batch_size": 50}
    model_str_repr = 'lstm1'
    model_class = Models.LSTMModel
    # model_class = Models.FCNNModel
    # model_class = Models.RandomForestModel

    col_names_and_offsets = col_names_and_offsets * int(60 / time_sample_res_minutes)

    is_target_in_input = True
    if target_col not in col_names_and_offsets.index:
        is_target_in_input = False
        col_names_and_offsets[target_col] = 0

    run_params = TestInstanceParams.TestInstanceParams(input_data_str_repr=input_data_str_repr, \
                                                       model_str_repr=model_str_repr, target_col=target_col, \
                                                       is_target_in_input=is_target_in_input,
                                                       pred_forward_hrs=pred_forward_hrs, \
                                                       look_back_hrs=look_back_hrs,
                                                       time_sample_res_minutes=time_sample_res_minutes, \
                                                       k=k, num_folds_to_run=num_folds_to_run, \
                                                       model_class=model_class, model_args=model_train_args, \
                                                       desc_str_addition='')

    data = downsample_data(data, run_params.downsample_ratio)
    df = Load.get_df_for_model(data, col_names_and_offsets)
    phys_target = df[run_params.phys_col]

    pre = Process.PreprocessData(steps_back=run_params.train_steps, \
                                 y_length=1, step_size=1, \
                                 gap_forward=run_params.pred_forward)

    run_folds_dict = run_kfold_train_test(df, phys_target, run_params, pre)

    pd.options.display.float_format = '{:,.3f}'.format
    print(run_params.desc_str)
    return run_folds_dict