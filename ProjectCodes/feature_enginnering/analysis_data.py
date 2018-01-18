#!/usr/bin/env python
# encoding: utf-8

"""
Analysis the train and test dataset, find out all characteristics of features.
"""

import pandas as pd
import numpy as np
import logging
import logging.config
import matplotlib.pyplot as plt
from functools import reduce

import time

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


def price_describe_by_col_value(df_train: pd.DataFrame, ser_value_cnt: pd.Series):
    price_describe_df = pd.DataFrame(index=df_train.describe().index)
    for value_i in ser_value_cnt.index:
        value_i_ser_ = train_df[df_train[ser_value_cnt.name] == value_i].price.describe()
        value_i_ser_.name = value_i
        price_describe_df = pd.concat([price_describe_df, value_i_ser_], axis=1)
    return price_describe_df


def have_substr(parent, child):
    if pd.isnull(parent):
        return False
    else:
        return parent.lower().find(child.lower()) != -1


if __name__ == "__main__":
    train_df = pd.read_csv('../../input/train.tsv', sep='\t', engine='python')
    test_df = pd.read_csv('../../input/test.tsv', sep='\t', engine='python')
    # train里面有train_id和price，test里面有test_id
    all_df = pd.concat([train_df, test_df]).reset_index(drop=True).loc[:, train_df.columns[1:]]  # price
    Logger.info("合并train和test数据后，输出看下：")
    Logger.info('Shape={}'.format(all_df.shape))
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None, 'display.height', None):
        Logger.info('\n{}'.format(all_df.head()))
        Logger.info('\n====看下DataFrame的info信息：\n{}'.format(all_df.info()))
        Logger.info('\n====看下DataFrame的数值列的描述信息：（其中有价值的只有price）\n{}'.format(all_df.describe()))
        Logger.info('\n====看下DataFrame的列值为空的情况：（其中test组进来的部分是没有price列的）\n{}'.format(all_df.isnull().any()))

    Logger.info("》》》》开始分析各个列的内容特点")

    Logger.info("【Price列】：（只存在train的DataFrame）")
    Logger.info('这个列的原始取值分布直方图：')
    price_ser = train_df['price']
    log_price_ser = price_ser.map(np.log1p)
    fig, axes = plt.subplots(nrows=1, ncols=2)  # 获得subplot集合
    price_ser.plot.hist(ax=axes[0], bins=50, figsize=(10, 5), edgecolor='white', range=[0, 250])
    axes[0].set_xlabel('price', fontsize=17)
    axes[0].set_ylabel('frequency', fontsize=17)
    axes[0].tick_params(labelsize=15)
    axes[0].set_title('Price Distribution - Training Set', fontsize=12)
    log_price_ser.plot.hist(ax=axes[1], bins=50, figsize=(10, 5), edgecolor='white')
    axes[1].set_xlabel('log(price+1)', fontsize=17)
    axes[1].set_ylabel('frequency', fontsize=17)
    axes[1].tick_params(labelsize=15)
    axes[1].set_title('Log(Price+1) Distribution - Training Set', fontsize=12)
    plt.show()
    Logger.info('可见price取对数后更像高斯分布，而且比赛规定的评价函数也是需要这个转换')

    Logger.info('【item_condition_id列】：分析取值类别和计数\n{}'.format(all_df['item_condition_id'].value_counts()))
    item_condition_id_describe_df = price_describe_by_col_value(train_df, train_df['item_condition_id'].value_counts())
    Logger.info('按照取值类别分析对价格的影响(注意price只存在于train中)\n{}'.format(item_condition_id_describe_df))
    Logger.info('简单分析来看item_condition_id列的取值和价格关系不是很明显')

    Logger.info('【category_name列】：分析子类以及fillna')
    Logger.info('先看不为null的类别名字的特点：')
    cat_name_have_df = all_df[all_df['category_name'].isnull() == False].copy()
    cat_name_have_df.loc[:, 'cat_num'] = cat_name_have_df.category_name.map(lambda name: len(name.split('/')))
    Logger.info('\n不为空的类别名字按照"/"来分割子类取得的子类个数\n{}'.format(cat_name_have_df['cat_num'].value_counts()))
    Logger.info('通过观察cat_num > 3的类别名字，我们暂时可以统一认为类别就是：主类/子类/子子类')
    # TODO: 这里用Null来填充，后续考虑根据其他描述信息使用别的来填充。
    # 因为看价格的分布，所以只会取train的数据
    cat_name_train_df = train_df[['category_name', 'price']].copy()
    cat_name_train_df.fillna("null_paul/null_paul/null_paul", inplace=True)
    def split_cat_name(name, str_class):
        sub_array = name.split('/')
        if str_class == 'main':
            return sub_array[0]
        elif str_class == 'sub':
            return sub_array[1]
        else:
            return '/'.join(sub_array[2:])
    cat_name_train_df.loc[:, 'cat_name_main'] = cat_name_train_df.category_name.map(lambda x: split_cat_name(x, 'main'))
    cat_name_train_df.loc[:, 'cat_name_sub'] = cat_name_train_df.category_name.map(lambda x: split_cat_name(x, 'sub'))
    cat_name_train_df.loc[:, 'cat_name_sub2'] = cat_name_train_df.category_name.map(lambda x: split_cat_name(x, 'sub2'))
    cat_main_name_describe_df = price_describe_by_col_value(cat_name_train_df, cat_name_train_df['cat_name_main'].value_counts())
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None, 'display.height', None):
        Logger.info('\n观察类别主类各个价格情况\n{}'.format(cat_main_name_describe_df))

    Logger.info('【brand_name】：分析品牌的价格区间')
    brand_n = len(set(all_df.brand_name.values))
    Logger.info('不同的品牌一共有多少种：{}'.format(brand_n))
    top10_brand_describe_df = price_describe_by_col_value(train_df, train_df['brand_name'].value_counts()[:10])
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None, 'display.height', None):
        Logger.info('\n观察Top10品牌的价格情况\n{}'.format(top10_brand_describe_df))
    Logger.info('看下品牌对应的商品类别主类的情况')
    all_cat_fillna_ser = all_df.category_name.fillna("null_paul/null_paul/null_paul")
    all_df.loc[:, 'cat_main'] = all_cat_fillna_ser.map(lambda x: split_cat_name(x, 'main'))
    brand_cat_class_n_ = all_df['cat_main'].groupby(all_df['brand_name']).apply(lambda x: x.value_counts().size)
    Logger.info('\n品牌含有商品主类个数 | 品牌个数\n{}'.format(brand_cat_class_n_.value_counts()))
    Logger.info('可见绝大多数品牌只在一两个主类中存在')
    Logger.info('将cat为空的去掉，再看下brand用有cat数目的关系')
    all_df_have_cat = all_df[~all_df.category_name.isnull()]
    remain_brand_n = len(set(all_df_have_cat.brand_name.values))
    Logger.info('去掉cat为空的数据后，还剩{}个brand，也就是说有{}个brand对应的cat都为空'.format(remain_brand_n, brand_n - remain_brand_n))
    all_df_have_cat.loc[:, 'cat_main'] = all_df_have_cat['category_name'].map(lambda x: split_cat_name(x, 'main'))
    brand_cat_class_n_ = all_df_have_cat['cat_main'].groupby(all_df_have_cat['brand_name']).apply(lambda x: len(set(x.values)))
    Logger.info('\n品牌含有商品主类个数 | 品牌个数\n{}'.format(brand_cat_class_n_.value_counts()))
    Logger.info('另外还需要看一点，就是brand中有多少\W字符存在，这个会影响后面用brand做正则匹配查找')
    def collect_W_char(str_from):
        if isinstance(str_from, str):
            W_finder = re.compile('\W')
            return set(W_finder.findall(str_from))
        else:
            return set()
    def set_merge(set1, set2):
        return set1.union(set2)
    Logger.info('brand_name中存在的\W字符有：{}'.format(reduce(set_merge, list(map(collect_W_char, set(all_df.brand_name.values))))))

    Logger.info('【shipping】：分析包邮与否的价格分布')
    price_shipBySeller = train_df.loc[train_df['shipping'] == 1, 'price']
    price_shipByBuyer = train_df.loc[train_df['shipping'] == 0, 'price']
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(np.log(price_shipBySeller + 1), color='#8CB4E1', alpha=1.0, bins=50,
            label='Price when Seller pays Shipping')
    ax.hist(np.log(price_shipByBuyer + 1), color='#007D00', alpha=0.7, bins=50,
            label='Price when Buyer pays Shipping')
    ax.set(title='Histogram Comparison', ylabel='% of Dataset in Bin')
    ax.legend(loc='best', fontsize=12)  # 把上面设置的图例（legend）创建生效
    plt.xlabel('log(price+1)', fontsize=14)
    plt.ylabel('frequency', fontsize=14)
    plt.title('价格分布 by Shipping Type', fontsize=14)
    plt.tick_params(labelsize=12)
    plt.show()

    Logger.info('【item_description】：分析商品描述信息（这是整个数据最关键的部分）')
    Logger.info('\n先看下数据为空的有多少\n{}'.format(all_df['item_description'].isnull().sum()))
    Logger.info('看起来只有4个数据为空，挺不错的吧。Too young。。。')
    Logger.info('商品描述里面充斥着大量的"No description yet"')
    Logger.info('\n全数据中==“No description yet”的有{}'.format((all_df['item_description']=="No description yet").sum()))
    Logger.info('数据中还有很多类似"No description yet"或者"No description"的数据')
    Logger.info('比如含“No description”的有: {}'.format((all_df.item_description.map(lambda x: have_substr(x, 'No description'))).sum()))
    Logger.info('不过其中也不是包含这种字符串的商品描述都是无效的，需要自定义函数来详细处理下')
    import re

    def filt_item_description_isnull(str_desc):
        if pd.isnull(str_desc):
            return True  # 'null_paul'
        else:
            no_mean = re.compile("(No description yet|No description)", re.I)
            left = re.sub(pattern=no_mean, repl='', string=str_desc)
            if len(left) > 4:
                return False
            else:
                return True
    Logger.info('真实空描述的数据有：{}'.format((all_df.item_description.map(filt_item_description_isnull)).sum()))

    # Null value is very important
    # TODO: 如何处理各个列中的NaN
    Logger.info('上面已经把数据集中的各个列分析了下，现在重点看下有NaN值的列并简单预估下处理方案')
    Logger.info('\ncategory_name列有NaN的可能，处理可以根据是否把类别展开做null填充。'
                '\n如果品牌名不为空，那么可以根据 品牌名|商品名|商品描述 进行估计'
                '\n还有一种填充方式是根据其他列的情况来用RF做预估')
    Logger.info('\nbrand_name列有NaN的可能，处理可以直接null填充。'
                '\n如果商品名不为空，那么从商品名进行推测'
                '\n还有一种填充方式是根据其他列的情况来用RF做预估')
    Logger.info('\nitem_description列有NaN的可能，处理可以null填充。'
                '\n或者使用商品名填充')
    Logger.info('查看这三列数据为空的数据个数和占比')
    all_df.loc[:, 'cat_is_null'] = all_df.category_name.map(pd.isnull)
    all_df.loc[:, 'brand_is_null'] = all_df.brand_name.map(pd.isnull)
    all_df.loc[:, 'desc_is_null'] = all_df.item_description.map(filt_item_description_isnull)
    Logger.info('cat_is_null个数{}, 占比{:.4%}'.format(all_df.cat_is_null.sum(), all_df.cat_is_null.sum() / all_df.shape[0]))
    Logger.info('brand_is_null个数{}, 占比{:.4%}'.format(all_df.brand_is_null.sum(), all_df.brand_is_null.sum() / all_df.shape[0]))
    Logger.info('desc_is_null个数{}, 占比{:.4%}'.format(all_df.desc_is_null.sum(), all_df.desc_is_null.sum() / all_df.shape[0]))
    Logger.info(
        'cat_is_null & brand_is_null个数{}, 占比{:.4%}'.
            format((all_df.cat_is_null & all_df.brand_is_null).sum(), (all_df.cat_is_null & all_df.brand_is_null).sum() / all_df.shape[0]))
    Logger.info(
        'cat_is_null & desc_is_null个数{}, 占比{:.4%}'.
            format((all_df.cat_is_null & all_df.desc_is_null).sum(), (all_df.cat_is_null & all_df.desc_is_null).sum() / all_df.shape[0]))
    Logger.info(
        'desc_is_null & brand_is_null个数{}, 占比{:.4%}'.
            format((all_df.desc_is_null & all_df.brand_is_null).sum(), (all_df.desc_is_null & all_df.brand_is_null).sum() / all_df.shape[0]))
    Logger.info(
        'cat_is_null & brand_is_null & desc_is_null 个数{}, 占比{:.4%}'.
            format((all_df.cat_is_null & all_df.brand_is_null & all_df.desc_is_null).sum(), (all_df.cat_is_null & all_df.brand_is_null & all_df.desc_is_null).sum() / all_df.shape[0]))

    # Test fillna methods accuracy
    Logger.info('因为填充fillna的方式有很多所以可以进行比较，看看填充之后与原有的差异度怎么样')
    brand_start_time = time.time()
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
    def base_name_fill_brand(rm_regex_brand_known_ordered_list:list, str_name):
        for brand_rm_regex in rm_regex_brand_known_ordered_list:
            brand_finder = re.compile(r'\b' + brand_rm_regex + r'\b')  # re.I
            if brand_finder.search(str_name):
                return recover_regex_char(brand_rm_regex)
        else:
            return 'paulnull'
    have_band_df = all_df[~all_df['brand_name'].isnull()].copy()
    brand_known_list = have_band_df['brand_name'].value_counts().index
    rm_regex_brand_known_list = list(map(rm_regex_char, brand_known_list))
    have_band_df['new_brand'] = have_band_df['name'].map(lambda x: base_name_fill_brand(rm_regex_brand_known_ordered_list=rm_regex_brand_known_list, str_name=x))
    real_new_brand_df = have_band_df[have_band_df['new_brand'] != 'paulnull']
    correct_brand_df = real_new_brand_df[real_new_brand_df['new_brand'] == real_new_brand_df['brand_name']]
    Logger.info('直接从name中提取brand词, 耗时 {:.3f}s'.format(time.time() - brand_start_time))
    Logger.info('直接从name中提取brand词: 在{}条有brand的数据中，从name中找到非空的new_brand有{}条，在找到的基础上正确率有{:.3%}'
                .format(have_band_df.shape[0], real_new_brand_df.shape[0], correct_brand_df.shape[0] / real_new_brand_df.shape[0]))
    Logger.info('直接从name中提取brand词: 这个方法的特点：1、有很大部分数据brand不在name中；2、找到的情况下有一定的错误概率；3、耗时太长')


    # 为了更好的定义time steps的长度，看下统计量
    Logger.info('查看name列，item_description列分词后的词长度统计')
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(token_pattern=r"\b\w+\b")
    tokenizer = vectorizer.build_tokenizer()

    def get_text_words_len(text):
        if pd.isnull(text):
            return 0
        else:
            return len(tokenizer(text))
    all_df.loc[:, 'name_words_len'] = all_df['name'].map(get_text_words_len)
    name_words_len_quantile = all_df.name_words_len.quantile([q / 10.0 for q in range(1, 11)])
    all_df.loc[:, 'desc_words_len'] = all_df['item_description'].map(get_text_words_len)
    desc_words_len_quantile = all_df.desc_words_len.quantile([q / 10.0 for q in range(1, 11)])

    def format_quantile_info(ser_quantil):
        ret_quantil = ser_quantil.map(lambda x: '{}个词'.format(int(x)))
        ret_quantil.name = 'name词长度的分位数值'
        ret_quantil.index.name = '分位数'
        ret_quantil.index = ret_quantil.index.map(lambda x: '{:.2%}'.format(x))
        return ret_quantil
    Logger.info('\nname列的词长度统计为：\n{}'.format(format_quantile_info(name_words_len_quantile)))
    Logger.info('\nitem_description列的词长度统计为：\n{}'.format(format_quantile_info(desc_words_len_quantile)))





