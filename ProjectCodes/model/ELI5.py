import platform
import os
import sys
import pandas as pd
import numpy as np
import time

from functools import reduce
from sklearn.linear_model import Ridge, RidgeCV
import logging
import logging.config
import lightgbm as lgb
import eli5

from ProjectCodes.model.OnlyRidge import selfregressor_predict_and_score


class CvGridParams(object):
    scoring = 'neg_mean_squared_error'  # 'r2'
    rand_state = 20180117

    def __init__(self, param_type:str='default'):
        if param_type == 'default':
            self.name = param_type
            self.all_params = {
                'solver': ['auto'],
                'fit_intercept': [True],
                'alpha': [4.75],  # np.linspace(0.01, 10, 100),
                'max_iter': [100],
                'normalize': [False],
                'tol': [0.05],
                'random_state': [self.rand_state],
            }
        else:
            print("Construct CvGridParams with error param_type: " + param_type)

    def rm_list_dict_params(self):
        for key in self.all_params.keys():
            self.all_params[key] = self.all_params.get(key)[0]


import re
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import train_test_split


def record_log(local_flag, str_log):
    print(str_log)


class DataReader():

    def __init__(self, local_flag: bool, cat_fill_type: str, brand_fill_type: str, item_desc_fill_type: str):
        record_log(local_flag, '\n构建数据DF时使用的参数：\n'
                               'local_flag={}, cat_fill_type={}, brand_fill_type={}, item_desc_fill_type={}'
                   .format(local_flag, cat_fill_type, brand_fill_type, item_desc_fill_type))
        TRAIN_FILE = "../input/train.tsv"
        TEST_FILE = "../input/test.tsv"
        self.local_flag = local_flag
        self.item_desc_fill_type = item_desc_fill_type

        if local_flag:
            train_df = pd.read_csv("../" + TRAIN_FILE, sep='\t', engine='python')  # , nrows=10000)
            test_df = pd.read_csv("../" + TEST_FILE, sep='\t', engine='python')  # , nrows=3000)
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
            train_df.loc[:, 'item_description'] = train_df['item_description'].map(
                lambda x: fill_item_description_null(x, ''))
            test_df.loc[:, 'item_description'] = test_df['item_description'].map(
                lambda x: fill_item_description_null(x, ''))
        elif item_desc_fill_type == 'fill_paulnull':
            train_df.loc[:, 'item_description'] = train_df['item_description'].map(
                lambda x: fill_item_description_null(x, 'paulnull'))
            test_df.loc[:, 'item_description'] = test_df['item_description'].map(
                lambda x: fill_item_description_null(x, 'paulnull'))
        elif item_desc_fill_type == 'base_name':
            train_df.loc[:, 'item_description'] = train_df[['item_description', 'name']].apply(
                lambda x: fill_item_description_null(x.iloc[0], x.iloc[1]), axis=1)
            test_df.loc[:, 'item_description'] = test_df[['item_description', 'name']].apply(
                lambda x: fill_item_description_null(x.iloc[0], x.iloc[1]), axis=1)
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
        # train_df['desc_W_len'] = train_df['item_description'].map(len_of_not_w)
        # test_df['desc_W_len'] = test_df['item_description'].map(len_of_not_w)

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
            train_df.loc[train_brand_null_index, 'brand_name'] = train_df.loc[train_brand_null_index, col_key].map(
                lambda x: get_brand_by_key(x, key2brand_map))
            test_df.loc[test_brand_null_index, 'brand_name'] = test_df.loc[test_brand_null_index, col_key].map(
                lambda x: get_brand_by_key(x, key2brand_map))
            n_before = train_brand_null_index.size + test_brand_null_index.size
            n_after = (train_df['brand_name'] == 'paulnull').sum() + (test_df['brand_name'] == 'paulnull').sum()
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
            have_cat_df = all_df[~all_df['category_name'].isnull()].copy()
            train_cat_null_index = train_df[train_df['category_name'] == 'paulnull/paulnull/paulnull'].index
            test_cat_null_index = test_df[test_df['category_name'] == 'paulnull/paulnull/paulnull'].index
            key2cat_map = do_col2cat_dict(data_df=have_cat_df, key_col=col_key)
            train_df.loc[train_cat_null_index, 'category_name'] = train_df.loc[train_cat_null_index, col_key].map(
                lambda x: get_cat_by_key(x, key2cat_map))
            test_df.loc[test_cat_null_index, 'category_name'] = test_df.loc[test_cat_null_index, col_key].map(
                lambda x: get_cat_by_key(x, key2cat_map))
            n_before = train_cat_null_index.size + test_cat_null_index.size
            n_after = (train_df['category_name'] == 'paulnull/paulnull/paulnull').sum() + (
                        test_df['category_name'] == 'paulnull/paulnull/paulnull').sum()
            record_log(local_flag, '直接name -> cat词, 耗时 {:.3f}s'.format(time.time() - cat_start_time))
            record_log(local_flag, '填充前有{}个空数据，填充后有{}个空数据，填充了{}个数据的cat'.format(n_before, n_after, n_before - n_after))
        elif cat_fill_type == 'base_brand':
            all_df = pd.concat([train_df, test_df]).reset_index(drop=True).loc[:, train_df.columns[1:]]  # Update all_df
            brand_cat_main_info_df, brand_cat_dict = get_brand_top_cat0_info_df(all_df)

            def get_cat_main_by_brand(brand_most_cat_dict: dict, row_ser: pd.Series):
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
            train_df.loc[:, 'category_name'] = train_df.apply(lambda row: get_cat_main_by_brand(brand_cat_dict, row),
                                                              axis=1)
            test_df.loc[:, 'category_name'] = test_df.apply(lambda row: get_cat_main_by_brand(brand_cat_dict, row),
                                                            axis=1)
            log = '\ncategory_name填充后, train中为空的有{}个, test为空的有{}个'.format(
                (train_df['category_name'] == 'paulnull/paulnull/paulnull').sum(),
                (test_df['category_name'] == 'paulnull/paulnull/paulnull').sum())
            record_log(local_flag, log)
        else:
            print('【错误】：cat_fill_type should be: "fill_paulnull" or "base_name" or "base_brand"')

        # splitting category_name into subcategories
        def split_cat(text):
            try:
                return text.split("/")
            except:
                return ("No Label", "No Label", "No Label")

        train_df['cat_name_main'], train_df['cat_name_sub'], train_df['cat_name_sub2'] = zip(
            *train_df['category_name'].apply(lambda x: split_cat(x)))
        test_df['cat_name_main'], test_df['cat_name_sub'], test_df['cat_name_sub2'] = zip(
            *test_df['category_name'].apply(lambda x: split_cat(x)))
        record_log(local_flag, "\n初始化之后train_df的列有{}".format(train_df.columns))
        record_log(local_flag, "\n初始化之后test_df的列有{}".format(test_df.columns))

        self.train_df = train_df
        self.test_df = test_df



data_reader = DataReader(local_flag=True, cat_fill_type='fill_paulnull', brand_fill_type='base_other_cols', item_desc_fill_type='fill_')



from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def cols_astype_to_str(dataset):
    dataset['shipping'] = dataset['shipping'].astype(str)
    dataset['item_condition_id'] = dataset['item_condition_id'].astype(str)
    dataset['desc_len'] = dataset['desc_len'].astype(str)
    dataset['name_len'] = dataset['name_len'].astype(str)
cols_astype_to_str(data_reader.train_df)
cols_astype_to_str(data_reader.test_df)

merge_df = pd.concat([data_reader.train_df, data_reader.test_df]).reset_index(drop=True)[data_reader.test_df.columns]
# print('~~Check~~ merge_df.axes = {}'.format(merge_df.axes))

default_preprocessor = CountVectorizer().build_preprocessor()
def build_preprocessor(field):
    field_idx = list(data_reader.test_df.columns).index(field)
    return lambda x: default_preprocessor(x[field_idx])

feat_union = FeatureUnion([
    ('name', CountVectorizer(
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
    ('item_description', TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=100000,
        preprocessor=build_preprocessor('item_description'))),
])
feat_union_start = time.time()
feat_union.fit(merge_df.values)
record_log(data_reader.local_flag, 'FeatureUnion fit() cost {}s'.format(time.time() - feat_union_start))
sparse_train_X = feat_union.transform(data_reader.train_df.drop('price', axis=1).values)
if 'target' in data_reader.train_df.columns:
    train_y = data_reader.train_df['target']
else:
    train_y = np.log1p(data_reader.train_df['price'])
sparse_test_X = feat_union.transform(data_reader.test_df.values)
record_log(data_reader.local_flag, 'FeatureUnion fit&transform() cost {}s'.format(time.time() - feat_union_start))

X_train, X_test, y_train, y_test = train_test_split(sparse_train_X, train_y, random_state=123, test_size=0.01)
record_log(data_reader.local_flag, "train_test_split: X_train={}, X_test={}".format(X_train.shape, X_test.shape))



cv_grid_params = CvGridParams()
cv_grid_params.rm_list_dict_params()
regress_model = Ridge(**cv_grid_params.all_params)
regress_model.fit(X_train, y_train)



validation_scores = pd.DataFrame(columns=["explained_var_score", "mean_abs_error", "mean_sqr_error", "median_abs_error", "r2score"])
predict_y, score_list = selfregressor_predict_and_score(regress_model, X_test, y_test)
validation_scores.loc["last_valida_df"] = score_list
with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None, 'display.height', None):
    print("对于样本集中留出的验证集整体打分有：\n{}".format(validation_scores))



eli5.show_weights(regress_model, vec=feat_union)


eli5.show_weights(regress_model, vec=feat_union, top=100, feature_filter=lambda x: x != '<BIAS>')


train_df_without_price = data_reader.train_df.drop('price', axis=1)
train_df_without_price.info()


eli5.show_prediction(regress_model, doc=train_df_without_price.values[0], vec=feat_union)


eli5.show_prediction(regress_model, doc=data_reader.test_df.values[0], vec=feat_union)

