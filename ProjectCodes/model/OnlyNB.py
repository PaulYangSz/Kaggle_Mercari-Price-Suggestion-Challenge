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

from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, CountVectorizer
from sklearn.linear_model import Ridge
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, median_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import logging
import logging.config

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

predict_by_cols = ['name']  # , 'item_description', 'cat_name_main', 'cat_name_sub', 'cat_name_sub2'


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
brand_classes = list(range(len(le.classes_)))
print('[{:.3f}] LabelEncoder brand_name finished.'.format(time.time() - start_time))
del le

nb_use_df['text'] = nb_use_df['name']  # .apply(construct_feature_text, axis=1)
print('[{:.3f}] Get text finished.'.format(time.time() - start_time))

dsample, dvalid = train_test_split(nb_use_df[['text', 'brand_le']], random_state=666, test_size=0.8)
print('dsample={}, dvalid={}'.format(dsample.shape, dvalid.shape))

# vectorizer = TfidfVectorizer(decode_error='ignore', max_features=2 ** 18, token_pattern=r"\S+")  # ngram_range=(1, 1)
vectorizer = CountVectorizer(decode_error='ignore', max_features=2 ** 18, token_pattern=r"\S+")  # ngram_range=(1, 1) lowercase=True
vectorizer.fit(nb_use_df['text'])
dsample_X = vectorizer.transform(dsample['text'])
dvalid_X = vectorizer.transform(dvalid['text'])
print('dsample_X={}, dvalid_X={}'.format(dsample_X.shape, dvalid_X.shape))
print('[{:.3f}] XXX_Vectorizer() finished.'.format(time.time() - start_time))
# del vectorizer


def time_measure(section, start, elapsed):
    lap = time.time() - start - elapsed
    elapsed = time.time() - start
    print("{:60}: {:15.2f}[sec]{:15.2f}[sec]".format(section, lap, elapsed))
    return elapsed


alpha_can = np.logspace(-3, -1, 10)
scoring='accuracy'
# clf = GridSearchCV(nb_model,
#                    param_grid={'alpha': alpha_can},
#                    cv=KFold(n_splits=3, shuffle=True, random_state=666),
#                    scoring=scoring)
PART_SAMPLE_N = 10000
start = time.time()
elapsed = 0
best_acc = 0
best_f1 = 0
best_alpha = 0
for alpha in alpha_can:
    nb_model = MultinomialNB(alpha=alpha)
    def nb_partial_fit(model_nb, all_X, all_y, y_classes, this_elapsed):
        part_n = np.math.ceil(all_X.shape[0] / PART_SAMPLE_N)
        for i in range(part_n):
            end = all_X.shape[0] if i==part_n-1 else PART_SAMPLE_N*(i+1)
            model_nb.partial_fit(all_X[PART_SAMPLE_N*i:end], all_y[PART_SAMPLE_N*i:end], classes=y_classes)
            this_elapsed = time_measure("  >>>> Part[{}].partial_fit()".format(i), start, this_elapsed)
    nb_partial_fit(nb_model, dsample_X, dsample['brand_le'], brand_classes, elapsed)
    elapsed = time_measure("> alpha={} nb_partial_fit()".format(alpha), start, elapsed)

    y_hat = nb_model.predict(dvalid_X)
    y_proba = nb_model.predict_proba(dvalid_X)
    # print('y_hat:\n{}'.format(y_hat[:3]))
    # print('y_proba:\n{}'.format(y_proba[:3]))
    acc = metrics.accuracy_score(dvalid['brand_le'], y_hat)
    f1_micro = metrics.f1_score(dvalid['brand_le'], y_hat, average='micro')
    print('validation dataset score: acc={:.4f}, f1={:.4f}'.format(acc, f1_micro))
    if best_acc < acc and best_f1 < f1_micro:
        best_acc = acc
        best_f1 = f1_micro
        best_alpha = alpha

print('best_alpha={}, best_acc={}, best_f1={}'.format(best_alpha, best_acc, best_f1))



