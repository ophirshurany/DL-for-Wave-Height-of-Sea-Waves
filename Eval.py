import math
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def calc_weighted_error(y_true, y_compare):
    """
    A self-developed error, weighted such that the smaller the wave, the more significant the error added
    current choice is to normalize and then divide by maximum between 0.5 and normed_true. This assumes that
    then inverse normalization, and calculate the "weighted rmse"
    (the 0.5 addition was chosen such that between the highest wave and lowest, significance
    of mistake drops down by a factor of 3 [the denominator is between 0.5 and 1.5])
    This can be changed of course to accommodate any desired relation and according to the max wave heights
    :param y_true: measurements
    :param y_compare: predictions
    :return:
    """
    weighted_error = (((y_true - y_compare)/np.maximum(y_true, 0.5*np.ones(y_true.shape)))**2).mean()
    root_weighted_error = math.sqrt(weighted_error)
    return root_weighted_error


def calc_si_normed(y_true, y_compare):
    """
    calculates the scatter index
    :param y_true: measurements
    :param y_compare: predictions
    :return: scatter index result
    """
    ''' '''
    return math.sqrt((((y_true-y_true.mean()) -
                       (y_compare - y_compare.mean()))**2).mean())*100 / y_true.mean()


def eval_model(y_true, y_compare, desired_metrics=None):
    """
    Calculates performance metrics
     Receives the y_true - ,
    and y_compare -
    desired_metrics - optional parameter for returning only some of them.
    At this stage it's not more efficient, still calculates all of them but
    just doesn't return them
    :param y_true: the ground truth measurements
    :param y_compare: be it the ML model preds, WW3 model, or any other
    forecast we'd like to evaluate
    :param desired_metrics: subset of values to calculate, out of the current list:
    rmse, r2 (r squared), si (two versions), mae, bias, max_error, my_weighted_rmse
    :return: dictionary with values for each metric
    """

    eval_results = {}
    eval_results['rmse'] = math.sqrt(mean_squared_error(y_true, y_compare))
    eval_results['r2'] = r2_score(y_true, y_compare)
    eval_results['si_simple'] = eval_results['rmse']/y_compare.mean()
    eval_results['si'] = calc_si_normed(y_true, y_compare)
    eval_results['mae'] = mean_absolute_error(y_true, y_compare)
    eval_results['bias'] = (y_true-y_compare).mean()
    eval_results['max_error'] = abs(y_true - y_compare).max()
    eval_results['my_weighted_rmse'] = calc_weighted_error(y_true, y_compare)

    if desired_metrics:
        # can change code to calculate only the desired ones and skip the rest
        assert(set(desired_metrics).issubset(set(eval_results.keys())))
        eval_results = dict((k, eval_results[k]) for k in desired_metrics)

    return eval_results


def eval_pred_phys_const(y_dict, pre):
    """
    Evaluates the predictions of the ML model, and of the desired baseline:
    the "const guess" prediction, and the WW3 model predictions
    :param y_dict - dictionary with key 'true' with ground truth measurements,
    key 'pred' with ML model predictions and key 'ww3' with WW3 model or any other physical model predictions
    :param pre: preprocessing object
    :return: a dataframe with all metrics per each forecast. df index is ML, WW3, Const_Guess
    """

    y_test = y_dict["true"]
    y_preds = y_dict["pred"]
    y_phys = y_dict["ww3"]

    wanted_metrics = ['rmse', 'r2', 'si', 'si_simple', 'mae', 'bias', 'max_error', 'my_weighted_rmse']
    o = eval_model(y_test, y_preds)
    p = eval_model(y_test, y_phys)
    gap = pre.get_forward_pred_num_steps()
    # calculate the metrics for the "const guess" option:
    # do so by considering the const guess "predictions" to be a shift back of the ground truth, in accordance with
    # num of steps forward for the forecast
    c = eval_model(y_test[:-gap], y_test[gap:])
    results = pd.DataFrame([o, p, c], index=['ML', 'WW3', 'Const_Guess'])
    return results[wanted_metrics]
