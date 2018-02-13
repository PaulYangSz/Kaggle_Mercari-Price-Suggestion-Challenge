#!/usr/bin/env python
# encoding: utf-8

"""
Read data and do some pre-process.
"""
import gc
import pandas as pd
import numpy as np
import re
import logging
import logging.config
import platform

import time
from scipy.sparse import csr_matrix, hstack
from keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords

np.random.seed(123)


if platform.system() == 'Windows':
    N_CORE = 1
    LOCAL_FLAG = True
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    # 有中文出现的情况，需要u'内容'
elif 's30' in platform.node():
    N_CORE = 4
    LOCAL_FLAG = True
else:
    N_CORE = 1
    LOCAL_FLAG = False

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


def record_log(local_flag, str_log):
    if local_flag:
        Logger.info(str_log)
    else:
        print(str_log)


class DataReader():

    def __init__(self, local_flag:bool, cat_fill_type:str, brand_fill_type:str, item_desc_fill_type:str):
        record_log(local_flag, '\n构建数据DF时使用的参数：\n'
                    'local_flag={}, cat_fill_type={}, brand_fill_type={}, item_desc_fill_type={}'
                   .format(local_flag, cat_fill_type, brand_fill_type, item_desc_fill_type))
        TRAIN_FILE = "../input/train.tsv"
        TEST_FILE = "../input/test.tsv"
        self.local_flag = local_flag
        self.item_desc_fill_type = item_desc_fill_type

        if local_flag:
            train_df = pd.read_csv("../" + TRAIN_FILE, sep='\t', engine='python')#, nrows=10000)
            test_df = pd.read_csv("../" + TEST_FILE, sep='\t', engine='python')#, nrows=3000)
        else:
            train_df = pd.read_csv(TRAIN_FILE, sep='\t')
            test_df = pd.read_csv(TEST_FILE, sep='\t')

        record_log(local_flag, 'Remain price!=0 items')
        train_df = train_df[train_df['price'] != 0]
        # record_log(local_flag, 'drop_duplicates()')
        # train_df_no_id = train_df.drop("train_id", axis=1)
        # train_df_no_id = train_df_no_id.drop_duplicates()
        # train_df = train_df.loc[train_df_no_id.index]

        stopwords_list = stopwords.words('english')
        # stop_patten = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
        word_patten = re.compile(r"(\w+(-\w+)+|\w+(\.\w+)+|\w+'\w+|\w+|!+)")
        def normal_desc(desc):
            try:
                filter_words = []
                for tuple_words in word_patten.findall(desc):
                    word = tuple_words[0]
                    if word.lower() not in stopwords_list:
                        filter_words.append(word)
                normal_text = " ".join(filter_words)
                return normal_text
            except:
                return ''
        rm_2_jiage = re.compile(r"\[rm\]")
        no_mean = re.compile(r"(No description yet|No description)", re.I)  # |\[rm\]
        def fill_item_description_null(str_desc, replace):
            if pd.isnull(str_desc):
                return replace
            else:
                changeRM = re.sub(pattern=rm_2_jiage, repl='jiagejine', string=str_desc)
                left = re.sub(pattern=no_mean, repl=replace, string=changeRM)
                if len(left) > 2:
                    return normal_desc(left)
                else:
                    return replace
        if item_desc_fill_type == 'fill_':
            train_df.loc[:, 'item_description'] = train_df['item_description'].map(lambda x: fill_item_description_null(x, ''))
            test_df.loc[:, 'item_description'] = test_df['item_description'].map(lambda x: fill_item_description_null(x, ''))
        elif item_desc_fill_type == 'fill_None':
            train_df['item_description'].fillna(value="None", inplace=True)
            test_df['item_description'].fillna(value="None", inplace=True)
        elif item_desc_fill_type == 'base_name':
            train_df.loc[:, 'item_description'] = train_df[['item_description', 'name']].apply(lambda x: fill_item_description_null(x.iloc[0], x.iloc[1]), axis=1)
            test_df.loc[:, 'item_description'] = test_df[['item_description', 'name']].apply(lambda x: fill_item_description_null(x.iloc[0], x.iloc[1]), axis=1)
        else:
            print('【错误】：item_desc_fill_type should be: "fill_" or "fill_paulnull" or "base_name"')


        # 尝试下对name只做normal但是不去停止词
        def normal_name(name):
            try:
                normal_text = " ".join(list(map(lambda x: x[0], word_patten.findall(name))))
                return normal_text
            except:
                return ''

        # train_df.loc[:, 'name'] = train_df['name'].map(normal_name)
        # test_df.loc[:, 'name'] = test_df['name'].map(normal_name)


        # 统计下description中特殊字符的个数
        npc_patten = re.compile(r'!')  # '!+'
        def patten_count(text, patten_):
            try:
                # text = text.lower()
                return len(patten_.findall(text))
            except:
                return 0
        train_df['desc_npc_cnt'] = train_df['item_description'].map(lambda x: patten_count(x, npc_patten))
        test_df['desc_npc_cnt'] = test_df['item_description'].map(lambda x: patten_count(x, npc_patten))


        # [先把能补充确定的brand填充上，然后再find brand]
        if brand_fill_type == 'fill_missing':
            train_df['brand_name'].fillna(value="missing", inplace=True)
            test_df['brand_name'].fillna(value="missing", inplace=True)
        else:
            print('【错误】：brand_fill_type should be: "fill_paulnull" or "base_other_cols" or "base_NB" or "base_GRU" ')




        if cat_fill_type == 'fill_Other':
            train_df['category_name'].fillna(value="Other", inplace=True)
            test_df['category_name'].fillna(value="Other", inplace=True)
        else:
            print('【错误】：cat_fill_type should be: "fill_paulnull" others are too cost time: "base_name" or "base_brand"')


        self.train_df = train_df
        self.test_df = test_df

        self.name_seq_len = 0
        self.item_desc_seq_len = 0
        self.n_name_dict_words = 0
        self.n_desc_dict_words = 0
        self.n_category = 0
        self.n_brand = 0
        self.n_condition_id = 0
        self.n_desc_max_len = 0
        self.n_npc_max_cnt = 0

    def le_encode(self):
        le = LabelEncoder()  # 给字符串或者其他对象编码, 从0开始编码

        # LabelEncoder category_name
        le.fit(np.hstack([self.train_df['category_name'], self.test_df['category_name']]))
        self.train_df['category_le'] = le.transform(self.train_df['category_name'])
        self.test_df['category_le'] = le.transform(self.test_df['category_name'])

        # LabelEncoder brand_name
        le.fit(np.hstack([self.train_df['brand_name'], self.test_df['brand_name']]))
        self.train_df['brand_le'] = le.transform(self.train_df['brand_name'])
        self.test_df['brand_le'] = le.transform(self.test_df['brand_name'])
        del le#, self.train_df['brand_name'], self.test_df['brand_name']

        record_log(self.local_flag, "\nLabelEncoder之后train_df的列有{}".format(self.train_df.columns))
        record_log(self.local_flag, "\nLabelEncoder之后test_df的列有{}".format(self.test_df.columns))

    def tokenizer_text_col(self, len_2_bin_flag):
        """
        将文本列分词并转编码，构成编码list
        """
        # 分割文本成词，然后将词转成编码(先分词，后编码, 编码从1开始)
        name_tok_raw = Tokenizer(num_words=250000)
        desc_tok_raw = Tokenizer(num_words=600000)
        # 这里构成raw文本的时候没有加入test数据是因为就算test中有新出现的词也不会在后续训练中改变词向量, but in SEQ will change
        name_raw_text = np.hstack([self.train_df['name'].str.lower(), self.test_df['name'].str.lower()])
        desc_raw_text = np.hstack([self.train_df['item_description'].str.lower(), self.test_df['item_description'].str.lower()])
        name_tok_raw.fit_on_texts(name_raw_text)
        desc_tok_raw.fit_on_texts(desc_raw_text)
        self.n_name_dict_words = min(max(name_tok_raw.word_index.values()), name_tok_raw.num_words) + 2
        self.n_desc_dict_words = min(max(desc_tok_raw.word_index.values()), desc_tok_raw.num_words) + 2
        print("self.n_name_dict_words={}, self.n_desc_dict_words={}".format(self.n_name_dict_words, self.n_desc_dict_words))

        # self.train_df["cat_int_seq"] = tok_raw.texts_to_sequences(self.train_df.category_name.str.lower())
        # self.test_df["cat_int_seq"] = tok_raw.texts_to_sequences(self.test_df.category_name.str.lower())
        self.train_df["name_int_seq"] = name_tok_raw.texts_to_sequences(self.train_df.name.str.lower())
        self.test_df["name_int_seq"] = name_tok_raw.texts_to_sequences(self.test_df.name.str.lower())
        self.train_df["desc_int_seq"] = desc_tok_raw.texts_to_sequences(self.train_df.item_description.str.lower())
        self.test_df["desc_int_seq"] = desc_tok_raw.texts_to_sequences(self.test_df.item_description.str.lower())
        self.train_df['desc_len'] = self.train_df['desc_int_seq'].apply(len)
        self.test_df['desc_len'] = self.test_df['desc_int_seq'].apply(len)

        if len_2_bin_flag:
            def len_2_bins(length, bins):
                for i in range(len(bins)):
                    if length <= bins[i]:
                        return i
                return len(bins)
            # train_desc_bins = list(range(1, self.train_df['desc_len'].max()+5, 3))
            train_desc_bins = list(set(self.train_df['desc_len'].quantile([q/1000 for q in range(1, 1001)]).values))
            train_desc_bins.sort()
            self.train_df['desc_len'] = self.train_df['desc_len'].apply(lambda x: len_2_bins(x, train_desc_bins))
            self.test_df['desc_len'] = self.test_df['desc_len'].apply(lambda x: len_2_bins(x, train_desc_bins))
            train_npc_bins = list(set(self.train_df['desc_npc_cnt'].quantile([q/1000 for q in range(1, 1001)]).values))
            train_npc_bins.sort()
            self.train_df['desc_npc_cnt'] = self.train_df['desc_npc_cnt'].map(lambda x: len_2_bins(x, train_npc_bins))
            self.test_df['desc_npc_cnt'] = self.test_df['desc_npc_cnt'].map(lambda x: len_2_bins(x, train_npc_bins))

        del name_tok_raw, desc_tok_raw

        record_log(self.local_flag, "\ntexts_to_sequences之后train_df的列有{}".format(self.train_df.columns))
        record_log(self.local_flag, "\ntexts_to_sequences之后test_df的列有{}".format(self.test_df.columns))

    def get_dummies(self, dummilize_cols):
        record_log(self.local_flag, "Need dummilize cols are: {}".format(dummilize_cols))
        all_need_dummy_df = pd.concat([self.train_df[dummilize_cols], self.test_df[dummilize_cols]]).reset_index(drop=True)
        dummy_df = pd.get_dummies(all_need_dummy_df[dummilize_cols].astype(str))
        dummy_len = dummy_df.shape[1]
        record_log(self.local_flag, "Get_dummies output len={}".format(dummy_len))
        for i in range(dummy_len):
            self.train_df['dummy_{}'.format(i)] = dummy_df.iloc[:self.train_df.shape[0], i]
            self.test_df['dummy_{}'.format(i)] = dummy_df.iloc[self.train_df.shape[0]:, i]
        return dummy_len

    def ensure_fixed_value(self):
        # TODO: 序列长度参数可调
        self.name_seq_len = 10  # 最长17个词
        self.item_desc_seq_len = 75  # 最长269个词，90%在62个词以内
        self.n_category = np.max([self.train_df.category_le.max(), self.test_df.category_le.max()]) + 1  # LE编码后最大值+1
        self.n_brand = np.max([self.train_df.brand_le.max(), self.test_df.brand_le.max()])+1
        self.n_condition_id = np.max([self.train_df.item_condition_id.max(), self.test_df.item_condition_id.max()])+1
        self.n_desc_max_len = np.max([self.train_df.desc_len.max(), self.test_df.desc_len.max()]) + 1
        print("self.train_df.desc_npc_cnt.max() =", self.train_df.desc_npc_cnt.max())
        print("self.test_df.desc_npc_cnt.max() =", self.test_df.desc_npc_cnt.max())
        self.n_npc_max_cnt = np.max([self.train_df.desc_npc_cnt.max(), self.test_df.desc_npc_cnt.max()]) + 1
        print("n_category:{},n_brand:{},n_condition_id:{},n_desc_max_len:{}".format(self.n_category, self.n_brand, self.n_condition_id, self.n_desc_max_len))

    def split_get_train_validation(self):
        """
        Split the train_df -> sample and last_validation
        :return: sample, validation, test
        """
        self.train_df['target'] = np.log1p(self.train_df['price'])
        dsample, dvalid = train_test_split(self.train_df, random_state=123, test_size=0.01)
        record_log(self.local_flag, "train_test_split: sample={}, validation={}".format(dsample.shape, dvalid.shape))
        return dsample, dvalid, self.test_df

    def get_keras_dict_data(self, dataset):
        """
        KERAS DATA DEFINITION
        name:名字词编号pad列表, item_desc:描述词编号pad列表,
        brand:品牌编号, category:类别编号, category_name:类别词编号pad列表,
        item_condition: item_condition_id, num_vars: shipping
        :param dataset:
        :return:
        """
        X = {
            'name': pad_sequences(dataset['name_int_seq'], maxlen=self.name_seq_len),
            'item_desc': pad_sequences(dataset.desc_int_seq, maxlen=self.item_desc_seq_len),
            'brand': np.array(dataset.brand_le),
            'category': np.array(dataset.category_le),
            'item_condition': np.array(dataset.item_condition_id),
            'num_vars': np.array(dataset[['shipping']]),
            'desc_len': np.array(dataset[["desc_len"]]),
            'desc_npc_cnt': np.array(dataset[["desc_npc_cnt"]]),
        }
        return X

    def del_redundant_cols(self):
        useful_cols = ['train_id', 'test_id', 'name', 'item_condition_id', 'category_name', 'brand_name', 'price', 'shipping',
                       'item_description', 'category_le',
                       'brand_le', 'name_int_seq', 'desc_int_seq', 'desc_len', 'desc_npc_cnt']
        for col in self.train_df.columns:
            if col not in useful_cols:
                del self.train_df[col]
        for col in self.test_df.columns:
            if col not in useful_cols:
                del self.test_df[col]
        gc.collect()

    def get_split_sparse_data(self):
        """
        无法和Keras的数据在一个模型里共存，因为这里需要用稀疏矩阵存储，而且大家对原始特征数据的处理方式也有不同
        :return:
        """
        def cols_astype_to_str(dataset):
            dataset['shipping'] = dataset['shipping'].astype(str)
            dataset['item_condition_id'] = dataset['item_condition_id'].astype(str)
            dataset['desc_len'] = dataset['desc_len'].astype(str)
            dataset['desc_npc_cnt'] = dataset['desc_npc_cnt'].astype(str)
        cols_astype_to_str(self.train_df)
        cols_astype_to_str(self.test_df)

        merge_df = pd.concat([self.train_df, self.test_df]).reset_index(drop=True)[self.test_df.columns]
        default_preprocessor = CountVectorizer().build_preprocessor()
        def build_preprocessor(field):
            field_idx = list(self.test_df.columns).index(field)
            return lambda x: default_preprocessor(x[field_idx])

        feat_union = FeatureUnion([
            ('name', CountVectorizer(
                ngram_range=(1, 2),
                max_features=50000,
                preprocessor=build_preprocessor('name'))),
            ('category_name', CountVectorizer(
                token_pattern='.+',
                preprocessor=build_preprocessor('category_name'))),
            ('brand_name', CountVectorizer(
                token_pattern='.+',
                preprocessor=build_preprocessor('brand_name'))),
            ('shipping', CountVectorizer(
                token_pattern='\d+',
                preprocessor=build_preprocessor('shipping'))),
            ('item_condition_id', CountVectorizer(
                token_pattern='\d+',
                preprocessor=build_preprocessor('item_condition_id'))),
            ('desc_len', CountVectorizer(
                token_pattern='\d+',
                preprocessor=build_preprocessor('desc_len'))),
            ('desc_npc_cnt', CountVectorizer(
                token_pattern='\d+',
                preprocessor=build_preprocessor('desc_npc_cnt'))),
            ('item_description', TfidfVectorizer(
                # token_pattern=r"(?u)\S+",
                ngram_range=(1, 2),
                max_features=100000,
                preprocessor=build_preprocessor('item_description'))),
        ])
        feat_union_start = time.time()
        feat_union.fit(merge_df.values)
        # feat_union.fit(self.train_df.drop('price', axis=1).values)
        record_log(self.local_flag, 'FeatureUnion fit() cost {}s'.format(time.time() - feat_union_start))
        sparse_train_X = feat_union.transform(self.train_df.drop('price', axis=1).values)
        # sparse_train_X = hstack((sparse_train_X, self.train_df[['desc_len', 'name_len', 'desc_npc_cnt']].values), format='csr')
        if 'target' in self.train_df.columns:
            train_y = self.train_df['target']
        else:
            train_y = np.log1p(self.train_df['price'])
        sparse_test_X = feat_union.transform(self.test_df.values)
        # sparse_test_X = hstack((sparse_test_X, self.test_df[['desc_len', 'name_len', 'desc_npc_cnt']].values), format='csr')
        record_log(self.local_flag, 'FeatureUnion fit&transform() cost {}s'.format(time.time() - feat_union_start))

        X_train, X_test, y_train, y_test = train_test_split(sparse_train_X, train_y, random_state=123, test_size=0.01)
        record_log(self.local_flag, "train_test_split: X_train={}, X_test={}".format(X_train.shape, X_test.shape))
        return X_train, X_test, y_train, y_test, sparse_test_X




