#!/usr/bin/env python
# encoding: utf-8

"""
Read data and do some pre-process.
"""

import pandas as pd
import numpy as np
import re
import logging
import logging.config
import platform

import time
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer


if platform.system() == 'Windows':
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
            train_df = pd.read_csv("../" + TRAIN_FILE, sep='\t', engine='python')
            test_df = pd.read_csv("../" + TEST_FILE, sep='\t', engine='python')
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
                no_mean = re.compile(r"(No description yet|No description|\[rm\])", re.I)
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




        # 以防brand的名字中含有正则表达式中的特殊字符，所以将其统统转换为"_". 目前不在这里做，在函数里面做，免得修改brand后匹配不上真正的
        def rm_W_in_brand(str_brand):
            if pd.isnull(str_brand):
                return str_brand
            else:
                return re.sub(pattern="[\||^|$|?|+|*|#|!]", repl="_", string=str_brand)
        # train_df.loc[:, 'brand_name'] = train_df['brand_name'].map(rm_W_in_brand)
        # test_df.loc[:, 'brand_name'] = test_df['brand_name'].map(rm_W_in_brand)
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
            n_after = (train_df['brand_name'] == 'paulnull').sum() + (test_df['brand_name'] == 'paulnull').sum()
            record_log(local_flag, '直接name -> brand词, 耗时 {:.3f}s'.format(time.time() - brand_start_time))
            record_log(local_flag, '填充前有{}个空数据，填充后有{}个空数据，填充了{}个数据的brand'.format(n_before, n_after, n_before - n_after))

            col_key = 'item_description'
            brand_start_time = time.time()
            all_df = pd.concat([train_df, test_df]).reset_index(drop=True).loc[:, train_df.columns[1:]]
            have_brand_df = all_df[all_df['brand_name'] != 'paulnull'].copy()
            train_brand_null_index = train_df[train_df['brand_name']=='paulnull'].index
            test_brand_null_index = test_df[test_df['brand_name']=='paulnull'].index
            key2brand_map = do_col2brand_dict(data_df=have_brand_df, key_col=col_key)
            train_df.loc[train_brand_null_index, 'brand_name'] = train_df.loc[train_brand_null_index, col_key].map(lambda x: get_brand_by_key(x, key2brand_map))
            test_df.loc[test_brand_null_index, 'brand_name'] = test_df.loc[test_brand_null_index, col_key].map(lambda x: get_brand_by_key(x, key2brand_map))
            n_before = train_brand_null_index.size + test_brand_null_index.size
            n_after = (train_df['brand_name'] == 'paulnull').sum() + (test_df['brand_name'] == 'paulnull').sum()
            record_log(local_flag, '直接desc -> brand词, 耗时 {:.3f}s'.format(time.time() - brand_start_time))
            record_log(local_flag, '填充前有{}个空数据，填充后有{}个空数据，填充了{}个数据的brand'.format(n_before, n_after, n_before - n_after))

            col_key = 'name+cat'
            brand_start_time = time.time()
            train_df['category_name'].fillna(value="paulnull/paulnull/paulnull", inplace=True)
            test_df['category_name'].fillna(value="paulnull/paulnull/paulnull", inplace=True)
            train_df[col_key] = train_df.apply(lambda row: row['name'] + row['category_name'], axis=1)
            test_df[col_key] = test_df.apply(lambda row: row['name'] + row['category_name'], axis=1)
            all_df = pd.concat([train_df, test_df]).reset_index(drop=True).loc[:, train_df.columns[1:]]
            have_brand_df = all_df[all_df['brand_name'] != 'paulnull'].copy()
            train_brand_null_index = train_df[train_df['brand_name']=='paulnull'].index
            test_brand_null_index = test_df[test_df['brand_name']=='paulnull'].index
            key2brand_map = do_col2brand_dict(data_df=have_brand_df, key_col=col_key)
            train_df.loc[train_brand_null_index, 'brand_name'] = train_df.loc[train_brand_null_index, col_key].map(lambda x: get_brand_by_key(x, key2brand_map))
            test_df.loc[test_brand_null_index, 'brand_name'] = test_df.loc[test_brand_null_index, col_key].map(lambda x: get_brand_by_key(x, key2brand_map))
            n_before = train_brand_null_index.size + test_brand_null_index.size
            n_after = (train_df['brand_name'] == 'paulnull').sum() + (test_df['brand_name'] == 'paulnull').sum()
            record_log(local_flag, 'name+cat -> brand词, 耗时 {:.3f}s'.format(time.time() - brand_start_time))
            record_log(local_flag, '填充前有{}个空数据，填充后有{}个空数据，填充了{}个数据的brand'.format(n_before, n_after, n_before - n_after))

            col_key = 'desc+cat'
            brand_start_time = time.time()
            train_df[col_key] = train_df.apply(lambda row: row['item_description'] + row['category_name'], axis=1)
            test_df[col_key] = test_df.apply(lambda row: row['item_description'] + row['category_name'], axis=1)
            all_df = pd.concat([train_df, test_df]).reset_index(drop=True).loc[:, train_df.columns[1:]]
            have_brand_df = all_df[all_df['brand_name'] != 'paulnull'].copy()
            train_brand_null_index = train_df[train_df['brand_name']=='paulnull'].index
            test_brand_null_index = test_df[test_df['brand_name']=='paulnull'].index
            key2brand_map = do_col2brand_dict(data_df=have_brand_df, key_col=col_key)
            train_df.loc[train_brand_null_index, 'brand_name'] = train_df.loc[train_brand_null_index, col_key].map(lambda x: get_brand_by_key(x, key2brand_map))
            test_df.loc[test_brand_null_index, 'brand_name'] = test_df.loc[test_brand_null_index, col_key].map(lambda x: get_brand_by_key(x, key2brand_map))
            n_before = train_brand_null_index.size + test_brand_null_index.size
            n_after = (train_df['brand_name'] == 'paulnull').sum() + (test_df['brand_name'] == 'paulnull').sum()
            record_log(local_flag, 'desc+cat -> brand词, 耗时 {:.3f}s'.format(time.time() - brand_start_time))
            record_log(local_flag, '填充前有{}个空数据，填充后有{}个空数据，填充了{}个数据的brand'.format(n_before, n_after, n_before - n_after))
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


        # Split category_name -> 3 sub-classes
        def change_df_split_cat(change_df:pd.DataFrame):
            change_df.loc[:, 'cat_name_main'] = change_df.category_name.map(lambda x: split_cat_name(x, 'main'))
            change_df.loc[:, 'cat_name_sub'] = change_df.category_name.map(lambda x: split_cat_name(x, 'sub'))
            change_df.loc[:, 'cat_name_sub2'] = change_df.category_name.map(lambda x: split_cat_name(x, 'sub2'))
        change_df_split_cat(train_df)
        change_df_split_cat(test_df)
        record_log(local_flag, "\n初始化之后train_df的列有{}".format(train_df.columns))
        record_log(local_flag, "\n初始化之后test_df的列有{}".format(test_df.columns))

        self.train_df = train_df
        self.test_df = test_df

        self.name_seq_len = 0
        self.item_desc_seq_len = 0
        self.cat_name_seq_len = 0
        self.n_text_dict_words = 0
        self.n_cat_main = 0
        self.n_cat_sub = 0
        self.n_cat_sub2 = 0
        self.n_brand = 0
        self.n_condition_id = 0

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
        del le, self.train_df['brand_name'], self.test_df['brand_name']

        record_log(self.local_flag, "\nLabelEncoder之后train_df的列有{}".format(self.train_df.columns))
        record_log(self.local_flag, "\nLabelEncoder之后test_df的列有{}".format(self.test_df.columns))

    def tokenizer_text_col(self):
        """
        将文本列分词并转编码，构成编码list
        """
        tok_raw = Tokenizer()  # 分割文本成词，然后将词转成编码(先分词，后编码, 编码从1开始)
        # 这里构成raw文本的时候没有加入test数据是因为就算test中有新出现的词也不会在后续训练中改变词向量
        raw_text = np.hstack([self.train_df['category_name'].str.lower(),
                              self.train_df['item_description'].str.lower(),
                              self.train_df['name'].str.lower()])
        tok_raw.fit_on_texts(raw_text)
        self.n_text_dict_words = max(tok_raw.word_index.values()) + 2

        self.train_df["cat_int_seq"] = tok_raw.texts_to_sequences(self.train_df.category_name.str.lower())
        self.test_df["cat_int_seq"] = tok_raw.texts_to_sequences(self.test_df.category_name.str.lower())
        self.train_df["name_int_seq"] = tok_raw.texts_to_sequences(self.train_df.name.str.lower())
        self.test_df["name_int_seq"] = tok_raw.texts_to_sequences(self.test_df.name.str.lower())
        self.train_df["desc_int_seq"] = tok_raw.texts_to_sequences(self.train_df.item_description.str.lower())
        self.test_df["desc_int_seq"] = tok_raw.texts_to_sequences(self.test_df.item_description.str.lower())

        record_log(self.local_flag, "\ntexts_to_sequences之后train_df的列有{}".format(self.train_df.columns))
        record_log(self.local_flag, "\ntexts_to_sequences之后test_df的列有{}".format(self.test_df.columns))

    def ensure_fixed_value(self):
        self.name_seq_len = 20  # 最长17个词
        self.item_desc_seq_len = 60  # 最长269个词，90%在62个词以内
        self.cat_name_seq_len = 20 # 最长8个词
        if self.n_text_dict_words == 0:
            self.n_text_dict_words = np.max([self.train_df.name_int_seq.map(max).max(),
                                             self.test_df.name_int_seq.map(max).max(),
                                             self.train_df.cat_int_seq.map(max).max(),
                                             self.test_df.cat_int_seq.map(max).max(),
                                             self.train_df.desc_int_seq.map(max).max(),
                                             self.test_df.desc_int_seq.map(max).max()]) + 2
        self.n_cat_main = np.max([self.train_df.cat_main_le.max(), self.test_df.cat_main_le.max()]) + 1  # LE编码后最大值+1
        self.n_cat_sub = np.max([self.train_df.cat_sub_le.max(), self.test_df.cat_sub_le.max()]) + 1
        self.n_cat_sub2 = np.max([self.train_df.cat_sub2_le.max(), self.test_df.cat_sub2_le.max()]) + 1
        self.n_brand = np.max([self.train_df.brand_le.max(), self.test_df.brand_le.max()])+1
        self.n_condition_id = np.max([self.train_df.item_condition_id.max(), self.test_df.item_condition_id.max()])+1

    def split_get_train_validation(self):
        """
        Split the train_df -> sample and last_validation
        :return: sample, validation, test
        """
        self.train_df['target'] = np.log1p(self.train_df['price'])
        dsample, dvalid = train_test_split(self.train_df, random_state=666, train_size=0.99)
        record_log(self.local_flag, "train_test_split: sample={}, validation={}".format(dsample.shape, dvalid.shape))
        return dsample, dvalid, self.test_df

    def get_keras_data(self, dataset):
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
            # 'category_name': pad_sequences(dataset.cat_int_seq, maxlen=self.cat_name_seq_len),
            'item_condition': np.array(dataset.item_condition_id),
            'num_vars': np.array(dataset[['shipping']])
        }
        return X






