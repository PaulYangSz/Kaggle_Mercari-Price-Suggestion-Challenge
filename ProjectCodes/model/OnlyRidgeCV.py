#!/usr/bin/env python
# encoding: utf-8

"""
Use sklearn based API model to local run and tuning.
"""
import platform
import os
import sys
import pandas as pd
import numpy as np
import time

from functools import reduce
from sklearn.linear_model import RidgeCV
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, median_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from keras.layers import Input, Dropout, Dense, BatchNormalization, \
    Activation, concatenate, GRU, Embedding, Flatten
from keras.models import Model
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping#, TensorBoard
from keras import backend as K
from keras import optimizers
import logging
import logging.config
import lightgbm as lgb

np.random.seed(123)

if platform.system() == 'Windows':
    N_CORE = 1
    LOCAL_FLAG = True
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    # 有中文出现的情况，需要u'内容'
elif 's30' in platform.node():
    N_CORE = 1
    LOCAL_FLAG = True
else:
    LOCAL_FLAG = False

if LOCAL_FLAG:
    CURR_DIR_Path = os.path.abspath(os.path.dirname(__file__))
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_Path = CURR_DIR_Path.split('ProjectCodes')[0]
    sys.path.append(ROOT_Path)
    from ProjectCodes.model.DataReader import DataReader
    from ProjectCodes.model.DataReader import record_log
    RNN_VERBOSE = 10
else:
    RNN_VERBOSE = 1


if LOCAL_FLAG:
    def start_logging():
        # 加载前面的标准配置
        from ProjectCodes.logging_config import ConfigLogginfDict
        logging.config.dictConfig(ConfigLogginfDict(__file__).LOGGING)
        # 获取loggers其中的一个日志管理器
        logger = logging.getLogger("default")
        logger.info('\n\n#################\n~~~~~~Start~~~~~~\n#################')
        print(type(logger))
        return logger
    if 'Logger' not in dir():
        Logger = start_logging()

RECORD_LOG = lambda log_str: record_log(LOCAL_FLAG, log_str)


class CvGridParams(object):
    scoring = 'neg_mean_squared_error'  # 'r2'
    rand_state = 20180117

    def __init__(self, param_type:str='default'):
        if param_type == 'default':
            self.name = param_type
            self.all_params = {
                'fit_intercept': [True],
                'alphas': [[alp] for alp in np.linspace(0.01, 10, 50)],
                'normalize': [False],
                'cv': [2, 5],
                'scoring': ['neg_mean_squared_error'],
            }
        else:
            print("Construct CvGridParams with error param_type: " + param_type)

    def rm_list_dict_params(self):
        for key in self.all_params.keys():
            self.all_params[key] = self.all_params.get(key)[0]


def print_param(cv_grid_params:CvGridParams):
    RECORD_LOG('选取的模型参数为：')
    RECORD_LOG("param_name = '{}'".format(cv_grid_params.name))
    RECORD_LOG("regression loss = {}".format(cv_grid_params.scoring))
    RECORD_LOG("rand_state = {}".format(cv_grid_params.rand_state))
    RECORD_LOG("param_dict = {")
    search_param_list = []
    for k, v in cv_grid_params.all_params.items():
        RECORD_LOG("\t'{}' = {}".format(k, v))
        if len(v) > 1:
            search_param_list.append(k)
    RECORD_LOG("}")
    return search_param_list


def get_cv_result_df(cv_results_:dict, adjust_paras:list, n_cv):
    cols = ['mean_test_score', 'mean_train_score', 'mean_fit_time']
    for param_ in adjust_paras:
        cols.append('param_{}'.format(param_))
    for i in range(n_cv):
        cols.append('split{}_test_score'.format(i))
    for i in range(n_cv):
        cols.append('split{}_train_score'.format(i))
    return pd.DataFrame(data={key: cv_results_[key] for key in cols}, columns=cols)


def show_CV_result(reg:GridSearchCV, adjust_paras, classifi_scoring):
    # pprint(reg.cv_results_)
    RECORD_LOG('XXXXX查看CV的结果XXXXXX')
    RECORD_LOG(
        '{}: MAX of mean_test_score = {}'.format(classifi_scoring, reg.cv_results_.get('mean_test_score').max()))
    RECORD_LOG(
        '{}: MAX of mean_train_score = {}'.format(classifi_scoring, reg.cv_results_.get('mean_train_score').max()))
    cv_result_df = get_cv_result_df(reg.cv_results_, adjust_paras, reg.cv.n_splits)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None, 'display.height', None):
        RECORD_LOG('\n对各组调参参数的交叉训练验证细节为：\n{}'.format(cv_result_df))
    if len(adjust_paras) == 1 and platform.system() == 'Windows':
        every_para_score = pd.Series()
        every_para_score.name = adjust_paras[0]
    for i in range(len(reg.cv_results_.get('mean_test_score'))):
        # RECORD_LOG('+++++++++++')
        # RECORD_LOG('mean_test_score = {}'.format(reg.cv_results_.get('mean_test_score')[i]))
        # RECORD_LOG('mean_train_score = {}'.format(reg.cv_results_.get('mean_train_score')[i]))
        param_str = "{"
        for k in adjust_paras:
            param_str += "'{}': {}, ".format(k, reg.cv_results_.get('params')[i][k])
        param_str = param_str[:-2] + "}"
        # RECORD_LOG('params = {}'.format(param_str))
        if len(adjust_paras) == 1 and platform.system() == 'Windows':
            record_param_value = reg.cv_results_.get('params')[i].get(adjust_paras[0])
            if isinstance(record_param_value, tuple):
                record_param_value = '{}'.format(reduce(lambda n_h, n_h1: str(n_h) + '_' + str(n_h1), record_param_value))
            every_para_score.loc[record_param_value] = reg.cv_results_.get('mean_test_score')[i]
    print('best_score_ = {}'.format(reg.best_score_))
    RECORD_LOG('reg.best_score_: %f' % reg.best_score_)
    for param_name in sorted(reg.best_params_.keys()):
        if param_name in adjust_paras:
            RECORD_LOG("调参选择为%s: %r" % (param_name, reg.best_params_[param_name]))
    if len(adjust_paras) == 1 and platform.system() == 'Windows':
        every_para_score.plot(kind='line', title=u'模型参数{}和评分{}的变化图示'.format(adjust_paras[0], classifi_scoring),
                              style='o-')
        plt.show()


def selfregressor_predict_and_score(reg, last_valida_X, last_valida_y):
    print('对样本集中留出的验证集进行预测:')
    predict_ = reg.predict(last_valida_X)
    # print(predict_)
    explained_var_score = explained_variance_score(y_true=last_valida_y, y_pred=predict_)
    mean_abs_error = mean_absolute_error(y_true=last_valida_y, y_pred=predict_)
    mean_sqr_error = mean_squared_error(y_true=last_valida_y, y_pred=predict_)
    median_abs_error = median_absolute_error(y_true=last_valida_y, y_pred=predict_)
    r2score = r2_score(y_true=last_valida_y, y_pred=predict_)
    # RECORD_LOG('使用sklearn的打分评价得到explained_var_score={}, mean_abs_error={}, mean_sqr_error={}, median_abs_error={}, r2score={}'
    #             .format(explained_var_score, mean_abs_error, mean_sqr_error, median_abs_error, r2score))
    return predict_, [explained_var_score, mean_abs_error, mean_sqr_error, median_abs_error, r2score]


if __name__ == "__main__":
    start_time = time.time()
    # 1. Get sample and last validation data.
    # Get Data include some pre-process.
    # Initial get fillna dataframe
    # cat_fill_type= "fill_paulnull" or "base_name" or "base_brand"
    # brand_fill_type= "fill_paulnull" or "base_other_cols" or "base_NB" or "base_GRU"
    # item_desc_fill_type= 'fill_' or 'fill_paulnull' or 'base_name'
    data_reader = DataReader(local_flag=LOCAL_FLAG, cat_fill_type='fill_paulnull', brand_fill_type='base_other_cols', item_desc_fill_type='fill_')
    RECORD_LOG('[{:.4f}s] Finished handling missing data...'.format(time.time() - start_time))

    data_reader.tokenizer_text_col()  # For desc_word_len, name_word_len and desc_npc_cnt
    data_reader.del_redundant_cols()

    # FIT FEATURES TRANSFORMERS
    RECORD_LOG("Fitting features pipeline and get train and test ...")
    sample_X, last_valida_X, sample_y, last_valida_y, test_X = data_reader.get_split_sparse_data()
    RECORD_LOG('[{:.4f}s] Finished FIT FEATURES TRANSFORMERS & SPLIT...'.format(time.time() - start_time))
    print('sample_X.shape={}'.format(sample_X.shape))
    print('last_valida_X.shape={}'.format(last_valida_X.shape))

    # 2. Check self-made estimator
    # check_estimator(LocalRegressor)  # Can not pass because need default DataReader in __init__.

    # 3. Parameters of GridSearchCV use.
    cv_grid_params = CvGridParams()
    adjust_para_list = print_param(cv_grid_params)

    if LOCAL_FLAG and len(adjust_para_list) > 0:
        # 4. Use GridSearchCV to tuning model.
        print('Begin to train self-defined sklearn-API regressor.')
        regress_model = RidgeCV()
        reg = GridSearchCV(estimator=regress_model,
                           param_grid=cv_grid_params.all_params,
                           n_jobs=N_CORE,
                           cv=KFold(n_splits=3, shuffle=True, random_state=cv_grid_params.rand_state),
                           scoring=cv_grid_params.scoring,
                           verbose=2,
                           refit=True)
        reg.fit(sample_X, sample_y)
        RECORD_LOG('[{:.4f}s] Finished Grid Search and training.'.format(time.time() - start_time))

        # 5. See the CV result
        show_CV_result(reg, adjust_paras=adjust_para_list, classifi_scoring=cv_grid_params.scoring)

        # 6. Use Trained Regressor to predict the last validation dataset
        validation_scores = pd.DataFrame(columns=["explained_var_score", "mean_abs_error", "mean_sqr_error", "median_abs_error", "r2score"])
        predict_y, score_list = selfregressor_predict_and_score(reg, last_valida_X, last_valida_y)
        validation_scores.loc["last_valida_df"] = score_list
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None, 'display.height', None):
            RECORD_LOG("对于样本集中留出的验证集整体打分有：\n{}".format(validation_scores))
        # last_valida_df['predict'] = predict_y
        # analysis_predict_result(last_valida_df)

        # 7. Predict and submit
        test_preds = reg.predict(test_X)
        test_preds = np.expm1(test_preds)
        RECORD_LOG('[{:.4f}s] Finished predicting test set...'.format(time.time() - start_time))
        submission = data_reader.test_df[["test_id"]].copy()
        submission["price"] = test_preds
        submission.to_csv("./csv_output/self_regressor_r2score_{:.5f}.csv".format(validation_scores.loc["last_valida_df", "r2score"]), index=False)
        RECORD_LOG('[{:.4f}s] Finished submission...'.format(time.time() - start_time))
    else:
        assert len(adjust_para_list) == 0
        cv_grid_params.rm_list_dict_params()
        regress_model = RidgeCV(**cv_grid_params.all_params)
        regress_model.fit(sample_X, sample_y)

        # 6. Use Trained Regressor to predict the last validation dataset
        validation_scores = pd.DataFrame(
            columns=["explained_var_score", "mean_abs_error", "mean_sqr_error", "median_abs_error", "r2score"])
        predict_y, score_list = selfregressor_predict_and_score(regress_model, last_valida_X, last_valida_y)
        validation_scores.loc["last_valida_df"] = score_list
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None, 'display.height', None):
            RECORD_LOG("对于样本集中留出的验证集整体打分有：\n{}".format(validation_scores))
        # last_valida_df['predict'] = predict_y

        test_preds = regress_model.predict(X=test_X)
        test_preds = np.expm1(test_preds)
        RECORD_LOG('[{:.4f}s] Finished predicting test set...'.format(time.time() - start_time))
        submission = data_reader.test_df[["test_id"]].copy()
        submission["price"] = test_preds
        file_path = './csv_output/' if LOCAL_FLAG else './'
        submission.to_csv(file_path + "self_regressor_r2score_{:.5f}.csv".format(validation_scores.loc["last_valida_df", "r2score"]), index=False)
        RECORD_LOG('[{:.4f}s] Finished submission...'.format(time.time() - start_time))



