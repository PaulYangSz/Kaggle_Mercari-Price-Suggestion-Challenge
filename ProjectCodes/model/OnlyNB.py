#!/usr/bin/env python
# encoding: utf-8

"""
Use sklearn based API model to local run and tuning.
"""
import platform

import gc
from sklearn import metrics

import pandas as pd
import numpy as np
import time

from functools import reduce

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, median_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
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

import matplotlib.pyplot as plt

from ProjectCodes.model.DataReader import DataReader
from ProjectCodes.model.LocalCvModel import show_CV_result

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


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

data_reader = DataReader(local_flag=True, cat_fill_type='fill_paulnull', brand_fill_type='fill_paulnull', item_desc_fill_type='fill_')

predict_by_cols = ['name', 'item_description', 'cat_name_main', 'cat_name_sub', 'cat_name_sub2']


def construct_feature_text(row):
    text = ''
    for col in predict_by_cols:
        text += row[col] + ' '
    return text

all_df = pd.concat([data_reader.train_df, data_reader.test_df]).reset_index(drop=True).loc[:, data_reader.train_df.columns[1:]]
nb_use_df = all_df.loc[all_df['brand_name'] != 'paulnull', predict_by_cols + ['brand_name']].copy()
del all_df
gc.collect()
print('nb_use_df.shape = {}'.format(nb_use_df.shape))

start_time = time.time()

le = LabelEncoder()  # 给字符串或者其他对象编码, 从0开始编码
nb_use_df['brand_le'] = le.fit_transform(nb_use_df['brand_name'])
print('[{:.3f}] LabelEncoder brand_name finished.'.format(time.time() - start_time))

nb_use_df['text'] = nb_use_df.apply(construct_feature_text, axis=1)
print('[{:.3f}] Get text finished.'.format(time.time() - start_time))

dsample, dvalid = train_test_split(nb_use_df[['text', 'brand_le']], random_state=666, train_size=0.9)
print('dsample={}, dvalid={}'.format(dsample.shape, dvalid.shape))

vectorizer = TfidfVectorizer(input='content', stop_words='english', max_df=0.5, sublinear_tf=True)
vectorizer.fit(nb_use_df['text'])
dsample_X = vectorizer.transform(dsample['text'])
dvalid_X = vectorizer.transform(dvalid['text'])
print('dsample_X={}, dvalid_X={}'.format(dsample_X.shape, dvalid_X.shape))
print('[{:.3f}] TfidfVectorizer() finished.'.format(time.time() - start_time))


nb_model = MultinomialNB()
alpha_can = np.logspace(-3, 2, 10)
scoring='accuracy'
clf = GridSearchCV(nb_model,
                   param_grid={'alpha': alpha_can},
                   cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=666),
                   scoring=scoring)
clf.fit(dsample_X, dsample['brand_le'])
print('[{:.3f}] GridSearchCV(cv=3) finished.'.format(time.time() - start_time))
show_CV_result(clf, adjust_paras=['alpha'], classifi_scoring=scoring)

y_hat = clf.predict(dvalid_X)
acc = metrics.accuracy_score(dvalid['brand_le'], y_hat)
f1_micro = metrics.f1_score(dvalid['brand_le'], y_hat, average='micro')
print('validation dataset score: acc={:.4f}, f1={:.4f}'.format(acc, f1_micro))



