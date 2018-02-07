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


def collect_W_char(str_from):
    if isinstance(str_from, str):
        W_finder = re.compile('\W')
        return set(W_finder.findall(str_from))
    else:
        return set()


def set_merge(set1, set2):
    return set1.union(set2)


def rm_regex_char(raw_str):
    raw_str = raw_str.replace('?', "\?")
    raw_str = raw_str.replace('*', "\*")
    raw_str = raw_str.replace('.', "\.")
    raw_str = raw_str.replace('|', "\|")
    raw_str = raw_str.replace('+', "\+")
    return raw_str


def recover_regex_char(raw_str):
    raw_str = raw_str.replace('\?', "?")
    raw_str = raw_str.replace('\*', "*")
    raw_str = raw_str.replace('\.', ".")
    raw_str = raw_str.replace('\|', "|")
    raw_str = raw_str.replace('\+', "+")
    return raw_str


def split_cat_name(name, str_class):
    sub_array = name.split('/')
    if str_class == 'main':
        return sub_array[0]
    elif str_class == 'sub':
        return sub_array[1]
    else:
        return '/'.join(sub_array[2:])


def get_brand_top_cat0_info_df(df_source:pd.DataFrame):
    """
    brand -> [top_cat0_name, count]
    :param df_source: train_df + test_df
    """
    df_source_have_cat = df_source[~df_source.category_name.isnull()].copy()
    df_source_have_cat.loc[:, 'cat_main'] = df_source_have_cat['category_name'].map(lambda x: split_cat_name(x, 'main'))
    group_brand = df_source_have_cat['cat_main'].groupby(df_source_have_cat['brand_name'])
    top_cat_main_ser = group_brand.apply(lambda x: x.value_counts().index[0])
    top_cat_main_ser.name = 'top_cat_main'
    top_cat_main_count_ser = group_brand.apply(lambda x: x.value_counts().iloc[0])
    top_cat_main_count_ser.name = 'item_count'
    index_brand = df_source_have_cat['brand_name'].value_counts().index
    ret_brand_top_cat0_info_df = pd.DataFrame({'top_cat_main': top_cat_main_ser, 'item_count': top_cat_main_count_ser},
                                              index=index_brand,
                                              columns=['top_cat_main', 'item_count'])
    return ret_brand_top_cat0_info_df, ret_brand_top_cat0_info_df['top_cat_main'].to_dict()


def base_other_cols_get_brand(brand_known_ordered_list:list, brand_top_cat0_dict:dict, row_ser:pd.Series):
    """
    通过前面的数据分析可以看到，name是不为空的，所以首先查看name中是否包含brand信息，找到匹配的brand集合
    其次使用item_description信息来缩小上述brand集合 (暂停使用)
    再次使用cat信息来看对应哪个brand最可能在这个cat上
    :param row_ser: 包含所需列的row
    :param brand_known_ordered_list: 按照商品个数有序的品牌list
    :param brand_top_cat0_info_df: brand对应top1主类别和item数目
    """
    if pd.isnull(row_ser['brand_name']):
        name = row_ser['name']
        brand_in_name = list()
        for brand in brand_known_ordered_list:
            # 在数据中，有的品牌名只会出现首个单词，不过看起来不像是大多数，所以索性还是从简单开始处理吧
            # 有的品牌名称和普通单词接近，比如Select，Complete之类，所以尽管有nike这样的小写存在，但是还是先不考虑小写了。
            rm_regex_brand = rm_regex_char(brand)
            brand_finder = re.compile(r'\b' + rm_regex_brand + r'\b')  # re.I
            if brand_finder.search(name):
                brand_in_name.append(brand)
        if len(brand_in_name) > 0:
            if pd.isnull(row_ser['category_name']):
                return brand_in_name[0]
            else:
                cat_main = row_ser['category_name'].split('/')[0]
                for brand in brand_in_name:
                    if brand in brand_top_cat0_dict and brand_top_cat0_dict[brand] == cat_main:
                        return brand
                return 'paulnull'
        else:
            return 'paulnull'
            # 暂停使用description来查找brand
            # desc = row_ser['item_description']
            # brand_in_desc = list()
            # if not pd.isnull(desc):
            #     for brand in brand_known_ordered_list:
            #         brand = rm_regex_char(brand)
            #         brand_finder = re.compile(r'\b' + brand + r'\b')  # re.I
            #         if brand_finder.search(desc):
            #             brand_in_desc.append(brand)

    #     if len(brand_in_name) == 0:
    #         brand_select = brand_in_desc
    #     else:
    #         if len(brand_in_desc) == 0:
    #             brand_select = brand_in_name
    #         else:
    #             brand_inter = [brand_ for brand_ in brand_in_name if brand_ in brand_in_desc]
    #             brand_select = brand_inter if len(brand_inter) > 0 else brand_in_name
    #
    #     if len(brand_select) == 1:
    #         return brand_select[0]
    #     elif len(brand_select) == 0:
    #         return 'paulnull'
    #     else:
    #         if pd.isnull(row_ser['category_name']):
    #             return brand_select[0]
    #         else:
    #             max_count = 0
    #             ret_brand = ''
    #             cat_main = row_ser['category_name'].split('/')[0]
    #             for brand in brand_select:
    #                 if brand_top_cat0_info_df.loc[brand, 'top_cat_main'] == cat_main:
    #                     this_count = brand_top_cat0_info_df.loc[brand, 'item_count']
    #                     if this_count >= max_count:
    #                         max_count = this_count
    #                         ret_brand = brand
    #             if max_count == 0:
    #                 return 'paulnull'
    #             else:
    #                 return ret_brand
    else:
        return row_ser['brand_name']


# TODO: base_name_get_brand可以尝试加入cat来判断，因为这个可能不是耗时的重点(不过一旦加入就要轮询band_list到底了)
def base_name_get_brand(rm_regex_brand_known_ordered_list:list, str_name):
    """
    通过前面的数据分析可以看到，name是不为空的，所以首先查看name中是否包含brand信息，找到匹配的brand集合
    再次使用cat信息来看对应哪个brand最可能在这个cat上 (因为想使用map而不是apply，所以可以考虑接下来再做一次map)
    """
    for rm_regex_brand in rm_regex_brand_known_ordered_list:
        # 在数据中，有的品牌名只会出现首个单词，不过看起来不像是大多数，所以索性还是从简单开始处理吧
        # 有的品牌名称和普通单词接近，比如Select，Complete之类，所以尽管有nike这样的小写存在，但是还是先不考虑小写了。
        brand_finder = re.compile(r'\b' + rm_regex_brand + r'\b')  # re.I
        if brand_finder.search(str_name):
            return recover_regex_char(rm_regex_brand)
    else:
        return 'paulnull'


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
        record_log(local_flag, 'drop_duplicates()')
        train_df_no_id = train_df.drop("train_id", axis=1)
        train_df_no_id = train_df_no_id.drop_duplicates()
        train_df = train_df.loc[train_df_no_id.index]

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
        elif item_desc_fill_type == 'fill_paulnull':
            train_df.loc[:, 'item_description'] = train_df['item_description'].map(lambda x: fill_item_description_null(x, 'paulnull'))
            test_df.loc[:, 'item_description'] = test_df['item_description'].map(lambda x: fill_item_description_null(x, 'paulnull'))
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

        train_df.loc[:, 'name'] = train_df['name'].map(normal_name)
        test_df.loc[:, 'name'] = test_df['name'].map(normal_name)


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
        if brand_fill_type == 'fill_paulnull':
            train_df['brand_name'].fillna(value="paulnull", inplace=True)
            test_df['brand_name'].fillna(value="paulnull", inplace=True)
        elif brand_fill_type == 'base_other_cols':
            def do_col2brand_dict(data_df: pd.DataFrame, key_col: str):
                group_by_key_to_brandset_ser = data_df['brand_name'].groupby(data_df[key_col]).apply(
                    lambda x: set(x.values))
                only_one_brand_ser = group_by_key_to_brandset_ser[group_by_key_to_brandset_ser.map(len) == 1]
                return only_one_brand_ser.map(lambda x: x.pop()).to_dict()

            def get_brand_by_key(key, map):
                if key in map:
                    return map[key]
                else:
                    return 'paulnull'

            col_key = 'name'
            brand_start_time = time.time()
            all_df = pd.concat([train_df, test_df]).reset_index(drop=True).loc[:, train_df.columns[1:]]
            have_brand_df = all_df[~all_df['brand_name'].isnull()].copy()
            train_brand_null_index = train_df[train_df['brand_name'].isnull()].index
            test_brand_null_index = test_df[test_df['brand_name'].isnull()].index
            key2brand_map = do_col2brand_dict(data_df=have_brand_df, key_col=col_key)
            train_df.loc[train_brand_null_index, 'brand_name'] = train_df.loc[train_brand_null_index, col_key].map(lambda x: get_brand_by_key(x, key2brand_map))
            test_df.loc[test_brand_null_index, 'brand_name'] = test_df.loc[test_brand_null_index, col_key].map(lambda x: get_brand_by_key(x, key2brand_map))
            n_before = train_brand_null_index.size + test_brand_null_index.size
            n_after = (train_df['brand_name']=='paulnull').sum() + (test_df['brand_name']=='paulnull').sum()
            record_log(local_flag, '直接name -> brand词, 耗时 {:.3f}s'.format(time.time() - brand_start_time))
            record_log(local_flag, '填充前有{}个空数据，填充后有{}个空数据，填充了{}个数据的brand'.format(n_before, n_after, n_before - n_after))

            # handling brand_name
            all_brands = set(have_brand_df['brand_name'].values)
            premissing = len(train_df.loc[train_df['brand_name'] == 'paulnull'])

            def brandfinder(line):
                """
                如果name含有brand信息，那么就用name代替brand
                :param line:
                :return:
                """
                brand = line[0]
                name = line[1]
                namesplit = name.split(' ')
                # TODO: 考虑下不管brand是否存在，都用name替换
                if brand == 'paulnull':
                    for x in namesplit:
                        if x in all_brands:
                            return name
                if name in all_brands:
                    return name
                return brand
            train_df['brand_name'] = train_df[['brand_name', 'name']].apply(brandfinder, axis=1)
            test_df['brand_name'] = test_df[['brand_name', 'name']].apply(brandfinder, axis=1)
            found = premissing - len(train_df.loc[train_df['brand_name'] == 'paulnull'])
            record_log(local_flag, 'On train dataset brandfinder() fill: {}'.format(found))

            # col_key = 'item_description'
            # brand_start_time = time.time()
            # all_df = pd.concat([train_df, test_df]).reset_index(drop=True).loc[:, train_df.columns[1:]]
            # have_brand_df = all_df[all_df['brand_name'] != 'paulnull'].copy()
            # train_brand_null_index = train_df[train_df['brand_name']=='paulnull'].index
            # test_brand_null_index = test_df[test_df['brand_name']=='paulnull'].index
            # key2brand_map = do_col2brand_dict(data_df=have_brand_df, key_col=col_key)
            # train_df.loc[train_brand_null_index, 'brand_name'] = train_df.loc[train_brand_null_index, col_key].map(lambda x: get_brand_by_key(x, key2brand_map))
            # test_df.loc[test_brand_null_index, 'brand_name'] = test_df.loc[test_brand_null_index, col_key].map(lambda x: get_brand_by_key(x, key2brand_map))
            # n_before = train_brand_null_index.size + test_brand_null_index.size
            # n_after = (train_df['brand_name'] == 'paulnull').sum() + (test_df['brand_name'] == 'paulnull').sum()
            # record_log(local_flag, '直接desc -> brand词, 耗时 {:.3f}s'.format(time.time() - brand_start_time))
            # record_log(local_flag, '填充前有{}个空数据，填充后有{}个空数据，填充了{}个数据的brand'.format(n_before, n_after, n_before - n_after))
            #
            # col_key = 'name+cat'
            # brand_start_time = time.time()
            # train_df['category_name'].fillna(value="paulnull/paulnull/paulnull", inplace=True)
            # test_df['category_name'].fillna(value="paulnull/paulnull/paulnull", inplace=True)
            # train_df[col_key] = train_df.apply(lambda row: row['name'] + row['category_name'], axis=1)
            # test_df[col_key] = test_df.apply(lambda row: row['name'] + row['category_name'], axis=1)
            # all_df = pd.concat([train_df, test_df]).reset_index(drop=True).loc[:, train_df.columns[1:]]
            # have_brand_df = all_df[all_df['brand_name'] != 'paulnull'].copy()
            # train_brand_null_index = train_df[train_df['brand_name']=='paulnull'].index
            # test_brand_null_index = test_df[test_df['brand_name']=='paulnull'].index
            # key2brand_map = do_col2brand_dict(data_df=have_brand_df, key_col=col_key)
            # train_df.loc[train_brand_null_index, 'brand_name'] = train_df.loc[train_brand_null_index, col_key].map(lambda x: get_brand_by_key(x, key2brand_map))
            # test_df.loc[test_brand_null_index, 'brand_name'] = test_df.loc[test_brand_null_index, col_key].map(lambda x: get_brand_by_key(x, key2brand_map))
            # n_before = train_brand_null_index.size + test_brand_null_index.size
            # n_after = (train_df['brand_name'] == 'paulnull').sum() + (test_df['brand_name'] == 'paulnull').sum()
            # record_log(local_flag, 'name+cat -> brand词, 耗时 {:.3f}s'.format(time.time() - brand_start_time))
            # record_log(local_flag, '填充前有{}个空数据，填充后有{}个空数据，填充了{}个数据的brand'.format(n_before, n_after, n_before - n_after))
        else:
            print('【错误】：brand_fill_type should be: "fill_paulnull" or "base_other_cols" or "base_NB" or "base_GRU" ')




        if cat_fill_type == 'fill_paulnull':
            train_df['category_name'].fillna(value="paulnull/paulnull/paulnull", inplace=True)
            test_df['category_name'].fillna(value="paulnull/paulnull/paulnull", inplace=True)
        else:
            print('【错误】：cat_fill_type should be: "fill_paulnull" others are too cost time: "base_name" or "base_brand"')

        # splitting category_name into subcategories
        def split_cat(text):
            try:
                return text.split("/")
            except:
                return ("No Label", "No Label", "No Label")
        train_df['cat_name_main'], train_df['cat_name_sub'], train_df['cat_name_sub2'] = zip(*train_df['category_name'].apply(lambda x: split_cat(x)))
        test_df['cat_name_main'], test_df['cat_name_sub'], test_df['cat_name_sub2'] = zip(*test_df['category_name'].apply(lambda x: split_cat(x)))
        record_log(local_flag, "\n初始化之后train_df的列有{}".format(train_df.columns))
        record_log(local_flag, "\n初始化之后test_df的列有{}".format(test_df.columns))

        self.train_df = train_df
        self.test_df = test_df

        self.name_seq_len = 0
        self.item_desc_seq_len = 0
        self.cat_name_seq_len = 0
        self.n_name_dict_words = 0
        self.n_desc_dict_words = 0
        self.n_cat_main = 0
        self.n_cat_sub = 0
        self.n_cat_sub2 = 0
        self.n_brand = 0
        self.n_condition_id = 0
        self.n_name_max_len = 0
        self.n_desc_max_len = 0
        self.n_npc_max_cnt = 0

    def le_encode(self):
        le = LabelEncoder()  # 给字符串或者其他对象编码, 从0开始编码

        # LabelEncoder cat_main & cat_sub & cat_sub2
        le.fit(np.hstack([self.train_df['cat_name_main'], self.test_df['cat_name_main']]))
        self.train_df['cat_main_le'] = le.transform(self.train_df['cat_name_main'])
        self.test_df['cat_main_le'] = le.transform(self.test_df['cat_name_main'])
        le.fit(np.hstack([self.train_df['cat_name_sub'], self.test_df['cat_name_sub']]))
        self.train_df['cat_sub_le'] = le.transform(self.train_df['cat_name_sub'])
        self.test_df['cat_sub_le'] = le.transform(self.test_df['cat_name_sub'])
        le.fit(np.hstack([self.train_df['cat_name_sub2'], self.test_df['cat_name_sub2']]))
        self.train_df['cat_sub2_le'] = le.transform(self.train_df['cat_name_sub2'])
        self.test_df['cat_sub2_le'] = le.transform(self.test_df['cat_name_sub2'])

        # LabelEncoder brand_name
        le.fit(np.hstack([self.train_df['brand_name'], self.test_df['brand_name']]))
        self.train_df['brand_le'] = le.transform(self.train_df['brand_name'])
        self.test_df['brand_le'] = le.transform(self.test_df['brand_name'])
        del le#, self.train_df['brand_name'], self.test_df['brand_name']

        record_log(self.local_flag, "\nLabelEncoder之后train_df的列有{}".format(self.train_df.columns))
        record_log(self.local_flag, "\nLabelEncoder之后test_df的列有{}".format(self.test_df.columns))

    def tokenizer_text_col(self):
        """
        将文本列分词并转编码，构成编码list
        """
        # 分割文本成词，然后将词转成编码(先分词，后编码, 编码从1开始)
        name_tok_raw = Tokenizer(num_words=150000, filters='\t\n')
        desc_tok_raw = Tokenizer(num_words=300000, filters='\t\n')
        # 这里构成raw文本的时候没有加入test数据是因为就算test中有新出现的词也不会在后续训练中改变词向量
        name_raw_text = np.hstack([self.train_df['name'].str.lower()])
        desc_raw_text = np.hstack([self.train_df['item_description'].str.lower()])
        name_tok_raw.fit_on_texts(name_raw_text)
        desc_tok_raw.fit_on_texts(desc_raw_text)
        self.n_name_dict_words = min(max(name_tok_raw.word_index.values()), name_tok_raw.num_words) + 2
        self.n_desc_dict_words = min(max(desc_tok_raw.word_index.values()), desc_tok_raw.num_words) + 2

        # self.train_df["cat_int_seq"] = tok_raw.texts_to_sequences(self.train_df.category_name.str.lower())
        # self.test_df["cat_int_seq"] = tok_raw.texts_to_sequences(self.test_df.category_name.str.lower())
        self.train_df["name_int_seq"] = name_tok_raw.texts_to_sequences(self.train_df.name.str.lower())
        self.test_df["name_int_seq"] = name_tok_raw.texts_to_sequences(self.test_df.name.str.lower())
        self.train_df["desc_int_seq"] = desc_tok_raw.texts_to_sequences(self.train_df.item_description.str.lower())
        self.test_df["desc_int_seq"] = desc_tok_raw.texts_to_sequences(self.test_df.item_description.str.lower())
        self.train_df['name_len'] = self.train_df['name_int_seq'].apply(len)
        self.test_df['name_len'] = self.test_df['name_int_seq'].apply(len)
        self.train_df['desc_len'] = self.train_df['desc_int_seq'].apply(len)
        self.test_df['desc_len'] = self.test_df['desc_int_seq'].apply(len)

        del name_tok_raw, desc_tok_raw

        record_log(self.local_flag, "\ntexts_to_sequences之后train_df的列有{}".format(self.train_df.columns))
        record_log(self.local_flag, "\ntexts_to_sequences之后test_df的列有{}".format(self.test_df.columns))

    def ensure_fixed_value(self):
        # TODO: 序列长度参数可调
        self.name_seq_len = 10  # 最长17个词
        self.item_desc_seq_len = 75  # 最长269个词，90%在62个词以内
        self.cat_name_seq_len = 8  # 最长8个词
        self.n_cat_main = np.max([self.train_df.cat_main_le.max(), self.test_df.cat_main_le.max()]) + 1  # LE编码后最大值+1
        self.n_cat_sub = np.max([self.train_df.cat_sub_le.max(), self.test_df.cat_sub_le.max()]) + 1
        self.n_cat_sub2 = np.max([self.train_df.cat_sub2_le.max(), self.test_df.cat_sub2_le.max()]) + 1
        self.n_brand = np.max([self.train_df.brand_le.max(), self.test_df.brand_le.max()])+1
        self.n_condition_id = np.max([self.train_df.item_condition_id.max(), self.test_df.item_condition_id.max()])+1
        self.n_desc_max_len = np.max([self.train_df.desc_len.max(), self.test_df.desc_len.max()]) + 1
        self.n_name_max_len = np.max([self.train_df.name_len.max(), self.test_df.name_len.max()]) + 1
        print("self.train_df.desc_npc_cnt.max() =", self.train_df.desc_npc_cnt.max())
        print("self.test_df.desc_npc_cnt.max() =", self.test_df.desc_npc_cnt.max())
        self.n_npc_max_cnt = np.max([self.train_df.desc_npc_cnt.max(), self.test_df.desc_npc_cnt.max()]) + 1

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
            'category_main': np.array(dataset.cat_main_le),
            'category_sub': np.array(dataset.cat_sub_le),
            'category_sub2': np.array(dataset.cat_sub2_le),
            'item_condition': np.array(dataset.item_condition_id),
            'num_vars': np.array(dataset[['shipping']]),
            'desc_len': np.array(dataset[["desc_len"]]),
            'name_len': np.array(dataset[["name_len"]]),
            'desc_npc_cnt': np.array(dataset[["desc_npc_cnt"]]),
        }
        return X

    def del_redundant_cols(self):
        useful_cols = ['train_id', 'test_id', 'name', 'item_condition_id', 'category_name', 'brand_name', 'price', 'shipping',
                       'item_description', 'cat_name_main', 'cat_name_sub', 'cat_name_sub2', 'cat_main_le', 'cat_sub_le', 'cat_sub2_le',
                       'brand_le', 'name_int_seq', 'desc_int_seq', 'desc_len', 'name_len', 'desc_npc_cnt']
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
            dataset['name_len'] = dataset['name_len'].astype(str)
            dataset['desc_npc_cnt'] = dataset['desc_npc_cnt'].astype(str)
            dataset['desc_len_diff'] = dataset['desc_len_diff'].astype(str)
        cols_astype_to_str(self.train_df)
        cols_astype_to_str(self.test_df)

        default_preprocessor = CountVectorizer().build_preprocessor()
        def build_preprocessor(field):
            field_idx = list(self.test_df.columns).index(field)
            return lambda x: default_preprocessor(x[field_idx])

        feat_union = FeatureUnion([
            ('name', CountVectorizer(
                token_pattern=r"(?u)\S+",
                ngram_range=(1, 2),
                max_features=50000,
                preprocessor=build_preprocessor('name'))),
            ('cat_name_main', CountVectorizer(
                token_pattern='.+',
                preprocessor=build_preprocessor('cat_name_main'))),
            ('cat_name_sub', CountVectorizer(
                token_pattern='.+',
                preprocessor=build_preprocessor('cat_name_sub'))),
            ('cat_name_sub2', CountVectorizer(
                token_pattern='.+',
                preprocessor=build_preprocessor('cat_name_sub2'))),
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
            ('name_len', CountVectorizer(
                token_pattern='\d+',
                preprocessor=build_preprocessor('name_len'))),
            ('desc_npc_cnt', CountVectorizer(
                token_pattern='\d+',
                preprocessor=build_preprocessor('desc_npc_cnt'))),
            ('desc_len_diff', CountVectorizer(
                token_pattern='\d+',
                preprocessor=build_preprocessor('desc_len_diff'))),
            ('item_description', TfidfVectorizer(
                token_pattern=r"(?u)\S+",
                ngram_range=(1, 2),
                max_features=100000,
                preprocessor=build_preprocessor('item_description'))),
        ])
        feat_union_start = time.time()
        feat_union.fit(self.train_df.drop('price', axis=1).values)
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




