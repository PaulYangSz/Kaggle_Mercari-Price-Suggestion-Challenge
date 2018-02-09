# For NN model do not appropriate for fold-CV and leave some small validation dataset can also save time
from pprint import pprint

import pandas as pd
from itertools import product

import time
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def time_measure(section, start, elapsed):
    lap = time.time() - start - elapsed
    elapsed = time.time() - start
    print("{:60}: {:15.2f}[sec]{:15.2f}[sec]".format(section, lap, elapsed))
    return elapsed


def make_all_scoring_cols(scoring_cols, n_valid):
    mean_cols = ["mean_{}".format(scoring) for scoring in scoring_cols]
    all_cols = []
    for scoring in scoring_cols:
        all_cols.extend([scoring+"_{}".format(i+1) for i in range(n_valid)])
    return mean_cols, all_cols


def score_validation(ml_model, valid_X, valid_y, clf_or_reg, result_df, i, i_valid):
    index = 'params_{}'.format(i+1)
    pred_y = ml_model.predict(valid_X)
    if clf_or_reg == 'clf_2':
        result_df.loc[index, "accuracy_{}".format(i_valid+1)] = accuracy_score(y_true=valid_y, y_pred=pred_y)
        result_df.loc[index, "f1_{}".format(i_valid+1)] = f1_score(y_true=valid_y, y_pred=pred_y)
    elif clf_or_reg == 'clf_n':
        result_df.loc[index, "accuracy_{}".format(i_valid+1)] = accuracy_score(y_true=valid_y, y_pred=pred_y)
        result_df.loc[index, "f1_macro_{}".format(i_valid+1)] = f1_score(y_true=valid_y, y_pred=pred_y, average='macro')
        result_df.loc[index, "f1_weighted_{}".format(i_valid+1)] = f1_score(y_true=valid_y, y_pred=pred_y, average='weighted')
    else:
        result_df.loc[index, "mean_squared_{}".format(i_valid+1)] = mean_squared_error(y_true=valid_y, y_pred=pred_y)
        result_df.loc[index, "r2_{}".format(i_valid+1)] = r2_score(y_true=valid_y, y_pred=pred_y)


def get_mean_score(result_df, i, cols_score, n_valid, key_score_col):
    index = 'params_{}'.format(i+1)
    key_score = -99999.0
    for score in cols_score:
        extend_score_cols = [score+"_{}".format(i+1) for i in range(n_valid)]
        result_df.loc[index, 'mean_{}'.format(score)] = result_df.loc[index, extend_score_cols].mean()
        if score == key_score_col:
            key_score = result_df.loc[index, 'mean_{}'.format(score)]
    return key_score


def show_current_selec_param(current_para_dict, search_param_list):
    tuning_params_dict = dict()
    for param in search_param_list:
        tuning_params_dict[param] = current_para_dict[param]
    pprint('leave_1_validation：当前选择的参数为\n{}'.format(tuning_params_dict))


def leave_1_validation(model_class, tuning_params, all_data_df, n_valid, test_ratio, y_col, clf_or_reg):
    """
    Input model and params output the tuning result(best_para_dict, result_df)
    :param model_class: Class of model
    :param tuning_params: {'param1': [value], 'tuning_param2': [value1, ... ,]}
    :param all_data_df:
    :param n_valid: number of validation
    :param test_ratio:
    :param y_col: y-column name
    :param clf_or_reg: 'clf_2'(Bi-Classification) or 'clf_n'(Mul-Classification) or 'reg'(regression)
    """
    # Get all need tuning params
    search_param_list = []
    for k, v in tuning_params.items():
        if len(v) > 1:
            search_param_list.append(k)
    search_param_list.sort()

    # Define scoring
    if clf_or_reg == 'clf_2':
        scoring_cols = ['accuracy', 'f1']  # , 'roc_auc' need support predict_proba()
        key_score_col = 'f1'
    elif clf_or_reg == 'clf_n':
        scoring_cols = ['accuracy', 'f1_macro', 'f1_weighted']
        key_score_col = 'f1_weighted'
    else:
        scoring_cols = ['mean_squared', 'r2']
        key_score_col = 'r2'

    # Define the result df
    mean_scoring_cols, extend_scoring_cols = make_all_scoring_cols(scoring_cols, n_valid)
    result_df = pd.DataFrame(columns=mean_scoring_cols+search_param_list+extend_scoring_cols)

    # loop every params combination
    best_key_score = -99999.0
    best_params_dict = dict()
    tuning_param_values = list(product(*[tuning_params[key] for key in search_param_list]))
    start = time.time()
    elapsed = 0
    for i in range(len(tuning_param_values)):
        print('~~~~~~~~~Now use params({}/{}) to train and scoring~~~~~~~~~~~'.format(i+1, len(tuning_param_values)))
        current_values = tuning_param_values[i]
        current_para_dict = dict()
        for key in tuning_params.keys():
            if key not in search_param_list:
                current_para_dict[key] = tuning_params[key][0]
            else:
                current_para_dict[key] = current_values[search_param_list.index(key)]
        show_current_selec_param(current_para_dict, search_param_list)

        for i_valid in range(n_valid):
            if n_valid == 1:
                rand_state = 123
            else:
                rand_state = None
            dsample, dvalid = train_test_split(all_data_df, test_size=test_ratio, random_state=rand_state)
            train_X = dsample.drop(y_col, axis=1)
            train_y = dsample[y_col].values
            valid_X = dvalid.drop(y_col, axis=1)
            valid_y = dvalid[y_col].values

            ml_model = model_class(**current_para_dict)
            ml_model.fit(train_X, train_y)

            # Use Trained model to predict the validation dataset
            # set [i, [v_i_s1, ... ,]]
            score_validation(ml_model, valid_X, valid_y, clf_or_reg, result_df, i, i_valid)
            if len(search_param_list) > 0:
                result_df.loc["params_{}".format(i+1), search_param_list] = list(current_values)
            del ml_model
            elapsed = time_measure("::::Once .fit() cost time::::", start, elapsed)

        # set [i, [mean_s1, ... , ]
        key_score = get_mean_score(result_df, i, scoring_cols, n_valid, key_score_col)
        if key_score > best_key_score:
            best_key_score = key_score
            best_params_dict = current_para_dict

    print('BestScore = {}'.format(best_key_score))
    pprint('Best Params = \n{}'.format(best_params_dict))
    result_df = result_df.sort_values(by=['mean_'.format(key_score_col)], ascending=False)
    return best_params_dict, result_df
