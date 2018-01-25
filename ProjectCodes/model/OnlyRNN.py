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
from sklearn.linear_model import Ridge
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, median_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
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


class SelfLocalRegressor(BaseEstimator, RegressorMixin):
    """ An sklearn-API regressor.
    Model 1: Embedding GRU ---- Embedding(text or cat) -> Concat[GRU(words) or Flatten(cat_vector)] ->  Dense -> Output
    Parameters
    ----------
    demo_param : All tuning parameters should be set in __init__()
        A parameter used for demonstation of how to pass and store paramters.
    Attributes
    ----------
    X_ : array, shape = [n_samples, n_features]
        The input passed during :meth:`fit`
    y_ : array, shape = [n_samples]
        The labels passed during :meth:`fit`
    """

    def __init__(self, data_reader:DataReader, name_emb_dim=20, item_desc_emb_dim=60, cat_name_emb_dim=20, brand_emb_dim=10,
                 cat_main_emb_dim=10, cat_sub_emb_dim=10, cat_sub2_emb_dim=10, item_cond_id_emb_dim=5, desc_len_dim=5, name_len_dim=5,
                 GRU_layers_out_dim=(8, 16), drop_out_layers=(0.25, 0.1), dense_layers_dim=(128, 64),
                 epochs=3, batch_size=512*3, lr_init=0.015, lr_final=0.007):
        self.data_reader = data_reader
        self.name_emb_dim = name_emb_dim
        self.item_desc_emb_dim = item_desc_emb_dim
        self.cat_name_emb_dim = cat_name_emb_dim
        self.brand_emb_dim = brand_emb_dim
        self.cat_main_emb_dim = cat_main_emb_dim
        self.cat_sub_emb_dim = cat_sub_emb_dim
        self.cat_sub2_emb_dim = cat_sub2_emb_dim
        self.item_cond_id_emb_dim = item_cond_id_emb_dim
        self.desc_len_dim = desc_len_dim
        self.name_len_dim = name_len_dim
        self.GRU_layers_out_dim = GRU_layers_out_dim
        assert len(drop_out_layers) == len(dense_layers_dim)
        self.drop_out_layers = drop_out_layers
        self.dense_layers_dim = dense_layers_dim
        self.emb_GRU_model = self.get_GRU_model(data_reader)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr_init = lr_init
        self.lr_final = lr_final

    def get_GRU_model(self, reader:DataReader):
        # Inputs
        name = Input(shape=[reader.name_seq_len], name="name")
        item_desc = Input(shape=[reader.item_desc_seq_len], name="item_desc")
        # category_name = Input(shape=[reader.cat_name_seq_len], name="category_name")
        item_condition = Input(shape=[1], name="item_condition")
        category_main = Input(shape=[1], name="category_main")
        category_sub = Input(shape=[1], name="category_sub")
        category_sub2 = Input(shape=[1], name="category_sub2")
        brand = Input(shape=[1], name="brand")
        num_vars = Input(shape=[1], name="num_vars")
        desc_len = Input(shape=[1], name="desc_len")
        name_len = Input(shape=[1], name="name_len")
        desc_W_len = Input(shape=[1], name="desc_W_len")

        # Embedding的作用是配置字典size和词向量len后，根据call参数的indices，返回词向量.
        #  类似TF的embedding_lookup
        #  name.shape=[None, MAX_NAME_SEQ] -> emb_name.shape=[None, MAX_NAME_SEQ, output_dim]
        emb_name = Embedding(input_dim=reader.n_text_dict_words, output_dim=self.name_emb_dim)(name)
        emb_item_desc = Embedding(reader.n_text_dict_words, self.item_desc_emb_dim)(item_desc)  # [None, MAX_ITEM_DESC_SEQ, emb_size]
        # emb_category_name = Embedding(reader.n_text_dict_words, self.cat_name_emb_dim)(category_name)
        emb_cond_id = Embedding(reader.n_condition_id, self.item_cond_id_emb_dim)(item_condition)
        emb_cat_main = Embedding(reader.n_cat_main, self.cat_main_emb_dim)(category_main)
        emb_cat_sub = Embedding(reader.n_cat_sub, self.cat_sub_emb_dim)(category_sub)
        emb_cat_sub2 = Embedding(reader.n_cat_sub2, self.cat_sub2_emb_dim)(category_sub2)
        emb_brand = Embedding(reader.n_brand, self.brand_emb_dim)(brand)
        emb_desc_len = Embedding(reader.n_desc_max_len, self.desc_len_dim)(desc_len)
        emb_name_len = Embedding(reader.n_name_max_len, self.name_len_dim)(name_len)
        emb_desc_W_len = Embedding(reader.n_desc_W_max_len, self.desc_len_dim)(desc_W_len)

        # GRU是配置一个cell输出的units长度后，根据call词向量入参,输出最后一个GRU cell的输出(因为默认return_sequences=False)
        rnn_layer_name = GRU(units=self.GRU_layers_out_dim[0])(emb_name)
        rnn_layer_item_desc = GRU(units=self.GRU_layers_out_dim[1])(emb_item_desc)  # rnn_layer_item_desc.shape=[None, 16]
        # rnn_layer_cat_name = GRU(units=self.GRU_layers_out_dim[2])(emb_category_name)

        # main layer
        # 连接列表中的Tensor，按照axis组成一个大的Tensor
        main_layer = concatenate([Flatten()(emb_brand),  # [None, 1, 10] -> [None, 10]
                                  Flatten()(emb_cat_main),
                                  Flatten()(emb_cat_sub),
                                  Flatten()(emb_cat_sub2),
                                  Flatten()(emb_cond_id),
                                  Flatten()(emb_desc_len),
                                  Flatten()(emb_name_len),
                                  Flatten()(emb_desc_W_len),
                                  rnn_layer_name,
                                  rnn_layer_item_desc,
                                  # rnn_layer_cat_name,
                                  num_vars])
        # Concat[all] -> Dense1 -> ... -> DenseN
        for i in range(len(self.dense_layers_dim)):
            main_layer = Dropout(self.drop_out_layers[i])(Dense(self.dense_layers_dim[i], activation='relu')(main_layer))

        # output
        output = Dense(1, activation="linear")(main_layer)

        # model
        model = Model(inputs=[name, item_desc, brand, category_main, category_sub, category_sub2, item_condition, num_vars, desc_len, name_len, desc_W_len],  # category_name
                      outputs=output)
        # optimizer = optimizers.RMSprop()
        optimizer = optimizers.Adam(lr=0.001, decay=0.0)
        model.compile(loss="mse", optimizer=optimizer)
        return model


    def fit(self, X, y):
        """A reference implementation of a fitting function for a regressor.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples]
            The target values. An array of float.
        Returns
        -------
        self : object
            Returns self.
        """
        # Check that X and y have correct shape
        # X, y = check_X_y(X, y)  # ValueError: setting an array element with a sequence. This is caused by "XXX_seq"

        self.X_ = X
        self.y_ = y

        # FITTING THE MODEL
        steps = int(X.shape[0] / self.batch_size) * self.epochs
        # final_lr=init_lr * (1/(1+decay))**(steps-1)
        exp_decay = lambda init, final, step_num: (init / final) ** (1 / (step_num - 1)) - 1
        lr_decay = exp_decay(self.lr_init, self.lr_final, steps)
        log_subdir = '_'.join(['ep', str(self.epochs),
                               'bs', str(self.batch_size),
                               'lrI', str(self.lr_init),
                               'lrF', str(self.lr_final)])
        K.set_value(self.emb_GRU_model.optimizer.lr, self.lr_init)
        K.set_value(self.emb_GRU_model.optimizer.decay, lr_decay)

        # print('~~~~~~~~~~~~In fit() type(X): {}'.format(type(X)))
        keras_X = self.data_reader.get_keras_dict_data(X)
        history = self.emb_GRU_model.fit(keras_X, y, epochs=self.epochs, batch_size=self.batch_size, validation_split=0., # 0.01
                                         # callbacks=[TensorBoard('./logs/'+log_subdir)],
                                         verbose=RNN_VERBOSE)

        # Return the regressor
        return self

    def predict(self, X):
        """ A reference implementation of a prediction for a regressor.
        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.
        Returns
        -------
        y : array of int of shape = [n_samples]
            The label for each sample is the label of the closest sample
            seen udring fit.
        """
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        # X = check_array(X)  # ValueError: setting an array element with a sequence. This is caused by "XXX_seq"

        keras_X = self.data_reader.get_keras_dict_data(X)
        gru_y = self.emb_GRU_model.predict(keras_X, batch_size=self.batch_size, verbose=RNN_VERBOSE)
        gru_y = gru_y.reshape(gru_y.shape[0])

        return gru_y


class CvGridParams(object):
    scoring = 'neg_mean_squared_error'  # 'r2'
    rand_state = 20180117

    def __init__(self, param_type:str='default'):
        if param_type == 'default':
            self.name = param_type
            self.all_params = {
                'name_emb_dim': [20],  # In name each word's vector length
                'item_desc_emb_dim': [60],
                'cat_name_emb_dim': [20],
                'brand_emb_dim': [10],
                'cat_main_emb_dim': [10],
                'cat_sub_emb_dim': [10],
                'cat_sub2_emb_dim': [10],
                'item_cond_id_emb_dim': [5],
                'desc_len_dim': [5],
                'name_len_dim': [5],
                'GRU_layers_out_dim': [(8, 16)],  # GRU hidden units
                'drop_out_layers': [(0.1, 0.1, 0.1, 0.1)],
                'dense_layers_dim': [(512, 256, 128, 64)],
                'epochs': [2],
                'batch_size': [512*3],
                'lr_init': [0.005],
                'lr_final': [0.001],
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


def train_model_with_gridsearch(regress_model:SelfLocalRegressor, sample_df, cv_grid_params):
    sample_X = sample_df.drop('target', axis=1)
    # sample_X = sample_X[['name_int_seq', 'desc_int_seq', 'brand_le', 'cat_main_le', 'cat_sub_le', 'cat_sub2_le', 'item_condition_id', 'shipping']]  # , 'cat_int_seq'
    sample_y = sample_df['target']

    # Check the list of available parameters with `estimator.get_params().keys()`
    print("keys are:::: {}".format(regress_model.get_params().keys()))

    reg = GridSearchCV(estimator=regress_model,
                       param_grid=cv_grid_params.all_params,
                       n_jobs=N_CORE,
                       cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=cv_grid_params.rand_state),
                       scoring=cv_grid_params.scoring,
                       verbose=2,
                       refit=True)
    reg.fit(sample_X, sample_y)
    return reg


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


def selfregressor_predict_and_score(reg, last_valida_df):
    print('对样本集中留出的验证集进行预测:')
    verify_X = last_valida_df.drop('target', axis=1)
    predict_ = reg.predict(verify_X)
    # print(predict_)
    verify_golden = last_valida_df['target'].values
    explained_var_score = explained_variance_score(y_true=verify_golden, y_pred=predict_)
    mean_abs_error = mean_absolute_error(y_true=verify_golden, y_pred=predict_)
    mean_sqr_error = mean_squared_error(y_true=verify_golden, y_pred=predict_)
    median_abs_error = median_absolute_error(y_true=verify_golden, y_pred=predict_)
    r2score = r2_score(y_true=verify_golden, y_pred=predict_)
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

    data_reader.del_redundant_cols()

    # PROCESS CATEGORICAL DATA
    RECORD_LOG("Handling categorical variables...")
    data_reader.le_encode()
    RECORD_LOG('[{:.4f}s] Finished PROCESSING CATEGORICAL DATA...'.format(time.time() - start_time))
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None,
                           'display.height', None):
        RECORD_LOG('\n{}'.format(data_reader.train_df.head(3)))

    # PROCESS TEXT: RAW
    RECORD_LOG("Text to seq process...")
    RECORD_LOG("   Fitting tokenizer...")
    data_reader.tokenizer_text_col()
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None,
                           'display.height', None):
        RECORD_LOG('\n{}'.format(data_reader.train_df.head(3)))
    RECORD_LOG('[{:.4f}s] Finished PROCESSING TEXT DATA...'.format(time.time() - start_time))

    # EMBEDDINGS MAX VALUE
    # Base on the histograms, we select the next lengths
    data_reader.ensure_fixed_value()
    RECORD_LOG('[{:.4f}s] Finished EMBEDDINGS MAX VALUE...'.format(time.time() - start_time))

    data_reader.del_redundant_cols()

    # EXTRACT DEVELOPMENT TEST
    sample_df, last_valida_df, test_df = data_reader.split_get_train_validation()
    last_valida_df.is_copy = None
    print(sample_df.shape)
    print(last_valida_df.shape)

    # 2. Check self-made estimator
    # check_estimator(LocalRegressor)  # Can not pass because need default DataReader in __init__.

    # 3. Parameters of GridSearchCV use.
    cv_grid_params = CvGridParams()
    adjust_para_list = print_param(cv_grid_params)

    if len(adjust_para_list) > 0:
        # 4. Use GridSearchCV to tuning model.
        regress_model = SelfLocalRegressor(data_reader=data_reader)
        print('Begin to train self-defined sklearn-API regressor.')
        reg = train_model_with_gridsearch(regress_model, sample_df, cv_grid_params)
        RECORD_LOG('[{:.4f}s] Finished Grid Search and training.'.format(time.time() - start_time))

        # 5. See the CV result
        show_CV_result(reg, adjust_paras=adjust_para_list, classifi_scoring=cv_grid_params.scoring)

        # 6. Use Trained Regressor to predict the last validation dataset
        validation_scores = pd.DataFrame(columns=["explained_var_score", "mean_abs_error", "mean_sqr_error", "median_abs_error", "r2score"])
        predict_y, score_list = selfregressor_predict_and_score(reg, last_valida_df)
        validation_scores.loc["last_valida_df"] = score_list
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None, 'display.height', None):
            RECORD_LOG("对于样本集中留出的验证集整体打分有：\n{}".format(validation_scores))
        last_valida_df['predict'] = predict_y
        # analysis_predict_result(last_valida_df)

        # 7. Predict and submit
        test_preds = reg.predict(test_df)
        test_preds = np.expm1(test_preds)
        RECORD_LOG('[{:.4f}s] Finished predicting test set...'.format(time.time() - start_time))
        submission = test_df[["test_id"]].copy()
        submission["price"] = test_preds
        submission.to_csv("./csv_output/self_regressor_r2score_{:.5f}.csv".format(validation_scores.loc["last_valida_df", "r2score"]), index=False)
        RECORD_LOG('[{:.4f}s] Finished submission...'.format(time.time() - start_time))
    else:
        cv_grid_params.rm_list_dict_params()
        regress_model = SelfLocalRegressor(data_reader=data_reader, **cv_grid_params.all_params)

        train_X = sample_df.drop('target', axis=1)
        train_y = sample_df['target'].values
        regress_model.fit(train_X, train_y)

        # 6. Use Trained Regressor to predict the last validation dataset
        validation_scores = pd.DataFrame(
            columns=["explained_var_score", "mean_abs_error", "mean_sqr_error", "median_abs_error", "r2score"])
        predict_y, score_list = selfregressor_predict_and_score(regress_model, last_valida_df)
        validation_scores.loc["last_valida_df"] = score_list
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None, 'display.height', None):
            RECORD_LOG("对于样本集中留出的验证集整体打分有：\n{}".format(validation_scores))
        last_valida_df['predict'] = predict_y

        test_preds = regress_model.predict(test_df)
        test_preds = np.expm1(test_preds)
        RECORD_LOG('[{:.4f}s] Finished predicting test set...'.format(time.time() - start_time))
        submission = test_df[["test_id"]].copy()
        submission["price"] = test_preds
        file_path = './csv_output/' if LOCAL_FLAG else './'
        submission.to_csv(file_path + "self_regressor_r2score_{:.5f}.csv".format(validation_scores.loc["last_valida_df", "r2score"]), index=False)
        RECORD_LOG('[{:.4f}s] Finished submission...'.format(time.time() - start_time))



