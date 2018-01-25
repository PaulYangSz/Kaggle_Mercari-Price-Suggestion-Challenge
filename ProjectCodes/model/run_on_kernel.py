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
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer


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
    name_cv = None
    cat_main_cv = None
    cat_sub_cv = None
    cat_sub2_cv = None
    desc_tv = None
    brand_lb = None

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


        def fill_item_description_null(str_desc, replace):
            if pd.isnull(str_desc):
                return replace
            else:
                no_mean = re.compile(r"(No description yet|No description)", re.I)  # |\[rm\]
                left = re.sub(pattern=no_mean, repl='', string=str_desc)
                if len(left) > 2:
                    return left
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

        # 统计下description中特殊字符的个数
        def len_of_not_w(str_from):
            if isinstance(str_from, str):
                W_finder = re.compile('\W')
                return len(W_finder.findall(str_from))
            else:
                return 0

        # handling categorical variables
        def wordCount(text):
            try:
                if text in ['No description yet', '', 'paulnull']:
                    return 0
                else:
                    text = text.lower()
                    words = [w for w in text.split(" ")]
                    return len(words)
            except:
                return 0

        train_df['desc_len'] = train_df['item_description'].apply(lambda x: wordCount(x))
        test_df['desc_len'] = test_df['item_description'].apply(lambda x: wordCount(x))
        train_df['name_len'] = train_df['name'].apply(lambda x: wordCount(x))
        test_df['name_len'] = test_df['name'].apply(lambda x: wordCount(x))


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
            #
            # col_key = 'desc+cat'
            # brand_start_time = time.time()
            # train_df[col_key] = train_df.apply(lambda row: row['item_description'] + row['category_name'], axis=1)
            # test_df[col_key] = test_df.apply(lambda row: row['item_description'] + row['category_name'], axis=1)
            # all_df = pd.concat([train_df, test_df]).reset_index(drop=True).loc[:, train_df.columns[1:]]
            # have_brand_df = all_df[all_df['brand_name'] != 'paulnull'].copy()
            # train_brand_null_index = train_df[train_df['brand_name']=='paulnull'].index
            # test_brand_null_index = test_df[test_df['brand_name']=='paulnull'].index
            # key2brand_map = do_col2brand_dict(data_df=have_brand_df, key_col=col_key)
            # train_df.loc[train_brand_null_index, 'brand_name'] = train_df.loc[train_brand_null_index, col_key].map(lambda x: get_brand_by_key(x, key2brand_map))
            # test_df.loc[test_brand_null_index, 'brand_name'] = test_df.loc[test_brand_null_index, col_key].map(lambda x: get_brand_by_key(x, key2brand_map))
            # n_before = train_brand_null_index.size + test_brand_null_index.size
            # n_after = (train_df['brand_name'] == 'paulnull').sum() + (test_df['brand_name'] == 'paulnull').sum()
            # record_log(local_flag, 'desc+cat -> brand词, 耗时 {:.3f}s'.format(time.time() - brand_start_time))
            # record_log(local_flag, '填充前有{}个空数据，填充后有{}个空数据，填充了{}个数据的brand'.format(n_before, n_after, n_before - n_after))
        else:
            print('【错误】：brand_fill_type should be: "fill_paulnull" or "base_other_cols" or "base_NB" or "base_GRU" ')




        if cat_fill_type == 'fill_paulnull':
            train_df['category_name'].fillna(value="paulnull/paulnull/paulnull", inplace=True)
            test_df['category_name'].fillna(value="paulnull/paulnull/paulnull", inplace=True)
        elif cat_fill_type == 'base_name':
            all_df = pd.concat([train_df, test_df]).reset_index(drop=True).loc[:, train_df.columns[1:]]  # Update all_df

            def do_col2cat_dict(data_df: pd.DataFrame, key_col: str):
                group_by_key_to_catset_ser = data_df['category_name'].groupby(data_df[key_col]).apply(
                    lambda x: set(x.values))
                only_one_cat_ser = group_by_key_to_catset_ser[group_by_key_to_catset_ser.map(len) == 1]
                return only_one_cat_ser.map(lambda x: x.pop()).to_dict()

            def get_cat_by_key(key, map):
                if key in map:
                    return map[key]
                else:
                    return 'paulnull/paulnull/paulnull'

            col_key = 'name'
            cat_start_time = time.time()
            have_cat_df = all_df[all_df['category_name']!='paulnull/paulnull/paulnull'].copy()
            train_cat_null_index = train_df[train_df['category_name']=='paulnull/paulnull/paulnull'].index
            test_cat_null_index = test_df[test_df['category_name']=='paulnull/paulnull/paulnull'].index
            key2cat_map = do_col2cat_dict(data_df=have_cat_df, key_col=col_key)
            train_df.loc[train_cat_null_index, 'category_name'] = train_df.loc[train_cat_null_index, col_key].map(lambda x: get_cat_by_key(x, key2cat_map))
            test_df.loc[test_cat_null_index, 'category_name'] = test_df.loc[test_cat_null_index, col_key].map(lambda x: get_cat_by_key(x, key2cat_map))
            n_before = train_cat_null_index.size + test_cat_null_index.size
            n_after = (train_df['category_name'] == 'paulnull/paulnull/paulnull').sum() + (test_df['category_name'] == 'paulnull/paulnull/paulnull').sum()
            record_log(local_flag, '直接name -> cat词, 耗时 {:.3f}s'.format(time.time() - cat_start_time))
            record_log(local_flag, '填充前有{}个空数据，填充后有{}个空数据，填充了{}个数据的cat'.format(n_before, n_after, n_before - n_after))
        elif cat_fill_type == 'base_brand':
            all_df = pd.concat([train_df, test_df]).reset_index(drop=True).loc[:, train_df.columns[1:]]  # Update all_df
            brand_cat_main_info_df, brand_cat_dict = get_brand_top_cat0_info_df(all_df)

            def get_cat_main_by_brand(brand_most_cat_dict:dict, row_ser:pd.Series):
                if pd.isnull(row_ser['category_name']):
                    str_brand = row_ser['brand_name']
                    if str_brand == 'paulnull' or str_brand not in brand_most_cat_dict:
                        str_cat_main = 'paulnull'
                    else:
                        str_cat_main = brand_most_cat_dict[str_brand]
                    return str_cat_main + '/paulnull/paulnull'
                else:
                    cat_name = row_ser['category_name']
                    cat_classes = cat_name.split('/')
                    if len(cat_classes) < 3:
                        cat_name += "/paulnull" * (3 - len(cat_classes))
                    return cat_name
            log = '\ncategory_name填充前, train中为空的有{}个, test为空的有{}个'.format(train_df['category_name'].isnull().sum(),
                                                                     test_df['category_name'].isnull().sum())
            record_log(local_flag, log)
            train_df.loc[:, 'category_name'] = train_df.apply(lambda row: get_cat_main_by_brand(brand_cat_dict, row), axis=1)
            test_df.loc[:, 'category_name'] = test_df.apply(lambda row: get_cat_main_by_brand(brand_cat_dict, row), axis=1)
            log = '\ncategory_name填充后, train中为空的有{}个, test为空的有{}个'.format((train_df['category_name']=='paulnull/paulnull/paulnull').sum(),
                                                                     (test_df['category_name']=='paulnull/paulnull/paulnull').sum())
            record_log(local_flag, log)
        else:
            print('【错误】：cat_fill_type should be: "fill_paulnull" or "base_name" or "base_brand"')

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
        self.n_text_dict_words = 0
        self.n_category = 0
        self.n_cat_main = 0
        self.n_cat_sub = 0
        self.n_cat_sub2 = 0
        self.n_brand = 0
        self.n_condition_id = 0
        self.n_name_max_len = 0
        self.n_desc_max_len = 0

    def le_encode(self):
        le = LabelEncoder()  # 给字符串或者其他对象编码, 从0开始编码

        # LabelEncoder cat-name
        le.fit(np.hstack([self.train_df['category_name'], self.test_df['category_name']]))
        self.train_df['category_le'] = le.transform(self.train_df['category_name'])
        self.test_df['category_le'] = le.transform(self.test_df['category_name'])

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
        tok_raw = Tokenizer()  # 分割文本成词，然后将词转成编码(先分词，后编码, 编码从1开始)
        # 这里构成raw文本的时候没有加入test数据是因为就算test中有新出现的词也不会在后续训练中改变词向量
        raw_text = np.hstack([self.train_df['item_description'].str.lower(),
                              self.test_df['item_description'].str.lower(),
                              self.train_df['category_name'].str.lower(),
                              self.test_df['category_name'].str.lower(),
                              self.train_df['name'].str.lower(),
                              self.test_df['name'].str.lower()])
        tok_raw.fit_on_texts(raw_text)
        self.n_text_dict_words = max(tok_raw.word_index.values()) + 2

        # self.train_df["cat_int_seq"] = tok_raw.texts_to_sequences(self.train_df.category_name.str.lower())
        # self.test_df["cat_int_seq"] = tok_raw.texts_to_sequences(self.test_df.category_name.str.lower())
        self.train_df["name_int_seq"] = tok_raw.texts_to_sequences(self.train_df.name.str.lower())
        self.test_df["name_int_seq"] = tok_raw.texts_to_sequences(self.test_df.name.str.lower())
        self.train_df["desc_int_seq"] = tok_raw.texts_to_sequences(self.train_df.item_description.str.lower())
        self.test_df["desc_int_seq"] = tok_raw.texts_to_sequences(self.test_df.item_description.str.lower())

        del tok_raw

        record_log(self.local_flag, "\ntexts_to_sequences之后train_df的列有{}".format(self.train_df.columns))
        record_log(self.local_flag, "\ntexts_to_sequences之后test_df的列有{}".format(self.test_df.columns))

    def ensure_fixed_value(self):
        self.name_seq_len = 10  # 最长17个词
        self.item_desc_seq_len = 75  # 最长269个词，90%在62个词以内
        self.cat_name_seq_len = 8 # 最长8个词
        if self.n_text_dict_words == 0:
            self.n_text_dict_words = np.max([self.train_df.name_int_seq.map(max).max(),
                                             self.test_df.name_int_seq.map(max).max(),
                                             # self.train_df.cat_int_seq.map(max).max(),
                                             # self.test_df.cat_int_seq.map(max).max(),
                                             self.train_df.desc_int_seq.map(max).max(),
                                             self.test_df.desc_int_seq.map(max).max()]) + 2
        self.n_category = np.max([self.train_df.category_le.max(), self.test_df.category_le.max()]) + 1
        self.n_cat_main = np.max([self.train_df.cat_main_le.max(), self.test_df.cat_main_le.max()]) + 1  # LE编码后最大值+1
        self.n_cat_sub = np.max([self.train_df.cat_sub_le.max(), self.test_df.cat_sub_le.max()]) + 1
        self.n_cat_sub2 = np.max([self.train_df.cat_sub2_le.max(), self.test_df.cat_sub2_le.max()]) + 1
        self.n_brand = np.max([self.train_df.brand_le.max(), self.test_df.brand_le.max()])+1
        self.n_condition_id = np.max([self.train_df.item_condition_id.max(), self.test_df.item_condition_id.max()])+1
        self.n_desc_max_len = np.max([self.train_df.desc_len.max(), self.test_df.desc_len.max()]) + 1
        self.n_name_max_len = np.max([self.train_df.name_len.max(), self.test_df.name_len.max()]) + 1

    def split_get_train_validation(self):
        """
        Split the train_df -> sample and last_validation
        :return: sample, validation, test
        """
        self.train_df['target'] = np.log1p(self.train_df['price'])
        dsample, dvalid = train_test_split(self.train_df, random_state=666, test_size=0.01)
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
            'category_main': np.array(dataset.cat_main_le),
            'category_sub': np.array(dataset.cat_sub_le),
            'category_sub2': np.array(dataset.cat_sub2_le),
            'item_condition': np.array(dataset.item_condition_id),
            'num_vars': np.array(dataset[['shipping']]),
            'desc_len': np.array(dataset[["desc_len"]]),
            'name_len': np.array(dataset[["name_len"]]),
        }
        return X

    def del_redundant_cols(self):
        useful_cols = ['train_id', 'test_id', 'name', 'item_condition_id', 'category_name', 'brand_name', 'price', 'shipping', 'item_description',
                       'category_le', 'cat_name_main', 'cat_name_sub', 'cat_name_sub2', 'cat_main_le', 'cat_sub_le', 'cat_sub2_le',
                       'brand_le', 'name_int_seq', 'desc_int_seq', 'desc_len', 'name_len']
        for col in self.train_df.columns:
            if col not in useful_cols:
                del self.train_df[col]
        for col in self.test_df.columns:
            if col not in useful_cols:
                del self.test_df[col]
        gc.collect()

    def train_ridge_numpy_data_condition(self):
        """
        无法和Keras的数据在一个模型里共存，因为这里需要用稀疏矩阵存储，而且大家对原始特征数据的处理方式也有不同
        :return:
        """
        NUM_BRANDS = 4500
        NUM_CATEGORIES = 1000
        NAME_MIN_DF = 10
        MAX_FEATURES_ITEM_DESCRIPTION = 90000

        merge_df = pd.concat([self.train_df, self.test_df]).reset_index(drop=True).loc[:, self.train_df.columns[1:]]

        def cutting(merge_set, train_set, test_set):
            pop_brand = merge_set['brand_name'].value_counts().loc[lambda x: x.index != 'paulnull'].index[:NUM_BRANDS]
            train_set.loc[~train_set['brand_name'].isin(pop_brand), 'brand_name'] = 'paulnull'
            test_set.loc[~test_set['brand_name'].isin(pop_brand), 'brand_name'] = 'paulnull'
            pop_category1 = merge_set['cat_name_main'].value_counts().loc[lambda x: x.index != 'paulnull'].index[:NUM_CATEGORIES]
            pop_category2 = merge_set['cat_name_sub'].value_counts().loc[lambda x: x.index != 'paulnull'].index[:NUM_CATEGORIES]
            pop_category3 = merge_set['cat_name_sub2'].value_counts().loc[lambda x: x.index != 'paulnull'].index[:NUM_CATEGORIES]
            train_set.loc[~train_set['cat_name_main'].isin(pop_category1), 'cat_name_main'] = 'paulnull'
            test_set.loc[~test_set['cat_name_main'].isin(pop_category1), 'cat_name_main'] = 'paulnull'
            train_set.loc[~train_set['cat_name_sub'].isin(pop_category2), 'cat_name_sub'] = 'paulnull'
            test_set.loc[~test_set['cat_name_sub'].isin(pop_category2), 'cat_name_sub'] = 'paulnull'
            train_set.loc[~train_set['cat_name_sub2'].isin(pop_category3), 'cat_name_sub2'] = 'paulnull'
            test_set.loc[~test_set['cat_name_sub2'].isin(pop_category3), 'cat_name_sub2'] = 'paulnull'
        cutting(merge_df, self.train_df, self.test_df)

        def to_categorical(dataset):
            dataset['cat_name_main'] = dataset['cat_name_main'].astype('category')
            dataset['cat_name_sub'] = dataset['cat_name_sub'].astype('category')
            dataset['cat_name_sub2'] = dataset['cat_name_sub2'].astype('category')
            dataset['item_condition_id'] = dataset['item_condition_id'].astype('category')
        to_categorical(self.train_df)
        to_categorical(self.test_df)
        merge_df = pd.concat([self.train_df, self.test_df]).reset_index(drop=True).loc[:, self.train_df.columns[1:]]

        self.name_cv = CountVectorizer(min_df=NAME_MIN_DF, ngram_range=(1, 2), stop_words='english')
        self.name_cv.fit(merge_df['name'])

        self.cat_main_cv = CountVectorizer()
        self.cat_main_cv.fit(merge_df['cat_name_main'])
        self.cat_sub_cv = CountVectorizer()
        self.cat_sub_cv.fit(merge_df['cat_name_sub'])
        self.cat_sub2_cv = CountVectorizer()
        self.cat_sub2_cv.fit(merge_df['cat_name_sub2'])

        self.desc_tv = TfidfVectorizer(max_features=MAX_FEATURES_ITEM_DESCRIPTION,
                                       ngram_range=(1, 2),
                                       stop_words='english')
        self.desc_tv.fit(merge_df['item_description'])

        self.brand_lb = LabelBinarizer(sparse_output=True)
        self.brand_lb.fit(merge_df['brand_name'])

    def get_ridge_sparse_data(self, dataset):
        X_name = self.name_cv.transform(dataset['name'])
        X_category1 = self.cat_main_cv.transform(dataset['cat_name_main'])
        X_category2 = self.cat_sub_cv.transform(dataset['cat_name_sub'])
        X_category3 = self.cat_sub2_cv.transform(dataset['cat_name_sub2'])
        X_description = self.desc_tv.transform(dataset['item_description'])
        X_brand = self.brand_lb.transform(dataset['brand_name'])
        X_dummies = csr_matrix(pd.get_dummies(dataset[['item_condition_id', 'shipping']], sparse=True).values)
        print(X_dummies.shape, X_description.shape, X_brand.shape, X_category1.shape, X_category2.shape, X_category3.shape, X_name.shape)
        return hstack((X_dummies, X_description, X_brand, X_category1, X_category2, X_category3, X_name)).tocsr()




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
        model = Model(inputs=[name, item_desc, brand, category_main, category_sub, category_sub2, item_condition, num_vars, desc_len, name_len],  # category_name
                      outputs=output)
        # optimizer = optimizers.RMSprop()
        optimizer = optimizers.Adam(lr=0.001, decay=0.0)
        model.compile(loss="mse", optimizer=optimizer)
        return model

    def get_ridge_lgm_models(self):
        ridge1 = Ridge(alpha=.6, copy_X=True, fit_intercept=True, max_iter=100, normalize=False, random_state=101, solver='auto', tol=0.01)
        ridge2 = Ridge(solver='sag', fit_intercept=True)
        lgb1 = None
        lgb2 = None
        return [ridge1, ridge2, lgb1, lgb2]


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
        submission = test_df[["test_id"]]
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
        submission.to_csv("./csv_output/self_regressor_r2score_{:.5f}.csv".format(validation_scores.loc["last_valida_df", "r2score"]),
                          index=False)
        RECORD_LOG('[{:.4f}s] Finished submission...'.format(time.time() - start_time))



