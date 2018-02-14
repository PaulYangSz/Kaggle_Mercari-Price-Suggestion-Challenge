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


def dameraulevenshtein(seq1, seq2):
    """Calculate the Damerau-Levenshtein distance between sequences.

    This method has not been modified from the original.
    Source: http://mwh.geek.nz/2009/04/26/python-damerau-levenshtein-distance/

    This distance is the number of additions, deletions, substitutions,
    and transpositions needed to transform the first sequence into the
    second. Although generally used with strings, any sequences of
    comparable objects will work.

    Transpositions are exchanges of *consecutive* characters; all other
    operations are self-explanatory.

    This implementation is O(N*M) time and O(M) space, for N and M the
    lengths of the two sequences.

    >>> dameraulevenshtein('ba', 'abc')
    2
    >>> dameraulevenshtein('fee', 'deed')
    2

    It works with arbitrary sequences too:
    >>> dameraulevenshtein('abcd', ['b', 'a', 'c', 'd', 'e'])
    2
    """
    # codesnippet:D0DE4716-B6E6-4161-9219-2903BF8F547F
    # Conceptually, this is based on a len(seq1) + 1 * len(seq2) + 1 matrix.
    # However, only the current and two previous rows are needed at once,
    # so we only store those.
    oneago = None
    thisrow = list(range(1, len(seq2) + 1)) + [0]
    for x in range(len(seq1)):
        # Python lists wrap around for negative indices, so put the
        # leftmost column at the *end* of the list. This matches with
        # the zero-indexed strings and saves extra calculation.
        twoago, oneago, thisrow = (oneago, thisrow, [0] * len(seq2) + [x + 1])
        for y in range(len(seq2)):
            delcost = oneago[y] + 1
            addcost = thisrow[y - 1] + 1
            subcost = oneago[y - 1] + (seq1[x] != seq2[y])
            thisrow[y] = min(delcost, addcost, subcost)
            # This block deals with transpositions
            if (x > 0 and y > 0 and seq1[x] == seq2[y - 1]
                    and seq1[x - 1] == seq2[y] and seq1[x] != seq2[y]):
                thisrow[y] = min(thisrow[y], twoago[y - 2] + 1)
    return thisrow[len(seq2) - 1]


class SymSpell:
    def __init__(self, max_edit_distance=3, verbose=0):
        self.max_edit_distance = max_edit_distance
        self.verbose = verbose
        # 0: top suggestion
        # 1: all suggestions of smallest edit distance
        # 2: all suggestions <= max_edit_distance (slower, no early termination)

        self.dictionary = {}
        self.longest_word_length = 0

    def get_deletes_list(self, w):
        """given a word, derive strings with up to max_edit_distance characters
           deleted"""

        deletes = []
        queue = [w]
        for d in range(self.max_edit_distance):
            temp_queue = []
            for word in queue:
                if len(word) > 1:
                    for c in range(len(word)):  # character index
                        word_minus_c = word[:c] + word[c + 1:]
                        if word_minus_c not in deletes:
                            deletes.append(word_minus_c)
                        if word_minus_c not in temp_queue:
                            temp_queue.append(word_minus_c)
            queue = temp_queue

        return deletes

    def create_dictionary_entry(self, w):
        '''add word and its derived deletions to dictionary'''
        # check if word is already in dictionary
        # dictionary entries are in the form: (list of suggested corrections,
        # frequency of word in corpus)
        new_real_word_added = False
        if w in self.dictionary:
            # increment count of word in corpus
            self.dictionary[w] = (self.dictionary[w][0], self.dictionary[w][1] + 1)
        else:
            self.dictionary[w] = ([], 1)
            self.longest_word_length = max(self.longest_word_length, len(w))

        if self.dictionary[w][1] == 1:
            # first appearance of word in corpus
            # n.b. word may already be in dictionary as a derived word
            # (deleting character from a real word)
            # but counter of frequency of word in corpus is not incremented
            # in those cases)
            new_real_word_added = True
            deletes = self.get_deletes_list(w)
            for item in deletes:
                if item in self.dictionary:
                    # add (correct) word to delete's suggested correction list
                    self.dictionary[item][0].append(w)
                else:
                    # note frequency of word in corpus is not incremented
                    self.dictionary[item] = ([w], 0)

        return new_real_word_added

    def create_dictionary_from_arr(self, arr, token_pattern=r'[a-z]+'):
        total_word_count = 0
        unique_word_count = 0

        for line in arr:
            # separate by words by non-alphabetical characters
            words = re.findall(token_pattern, line.lower())
            for word in words:
                total_word_count += 1
                if self.create_dictionary_entry(word):
                    unique_word_count += 1

        print("total words processed: %i" % total_word_count)
        print("total unique words in corpus: %i" % unique_word_count)
        print("total items in dictionary (corpus words and deletions): %i" % len(self.dictionary))
        print("  edit distance for deletions: %i" % self.max_edit_distance)
        print("  length of longest word in corpus: %i" % self.longest_word_length)
        return self.dictionary

    def create_dictionary(self, fname):
        total_word_count = 0
        unique_word_count = 0

        with open(fname) as file:
            for line in file:
                # separate by words by non-alphabetical characters
                words = re.findall('[a-z]+', line.lower())
                for word in words:
                    total_word_count += 1
                    if self.create_dictionary_entry(word):
                        unique_word_count += 1

        print("total words processed: %i" % total_word_count)
        print("total unique words in corpus: %i" % unique_word_count)
        print("total items in dictionary (corpus words and deletions): %i" % len(self.dictionary))
        print("  edit distance for deletions: %i" % self.max_edit_distance)
        print("  length of longest word in corpus: %i" % self.longest_word_length)
        return self.dictionary

    def get_suggestions(self, string, silent=False):
        """return list of suggested corrections for potentially incorrectly
           spelled word"""
        if (len(string) - self.longest_word_length) > self.max_edit_distance:
            if not silent:
                print("no items in dictionary within maximum edit distance")
            return []

        suggest_dict = {}
        min_suggest_len = float('inf')

        queue = [string]
        q_dictionary = {}  # items other than string that we've checked

        while len(queue) > 0:
            q_item = queue[0]  # pop
            queue = queue[1:]

            # early exit
            if ((self.verbose < 2) and (len(suggest_dict) > 0) and
                    ((len(string) - len(q_item)) > min_suggest_len)):
                break

            # process queue item
            if (q_item in self.dictionary) and (q_item not in suggest_dict):
                if self.dictionary[q_item][1] > 0:
                    # word is in dictionary, and is a word from the corpus, and
                    # not already in suggestion list so add to suggestion
                    # dictionary, indexed by the word with value (frequency in
                    # corpus, edit distance)
                    # note q_items that are not the input string are shorter
                    # than input string since only deletes are added (unless
                    # manual dictionary corrections are added)
                    assert len(string) >= len(q_item)
                    suggest_dict[q_item] = (self.dictionary[q_item][1],
                                            len(string) - len(q_item))
                    # early exit
                    if (self.verbose < 2) and (len(string) == len(q_item)):
                        break
                    elif (len(string) - len(q_item)) < min_suggest_len:
                        min_suggest_len = len(string) - len(q_item)

                # the suggested corrections for q_item as stored in
                # dictionary (whether or not q_item itself is a valid word
                # or merely a delete) can be valid corrections
                for sc_item in self.dictionary[q_item][0]:
                    if sc_item not in suggest_dict:

                        # compute edit distance
                        # suggested items should always be longer
                        # (unless manual corrections are added)
                        assert len(sc_item) > len(q_item)

                        # q_items that are not input should be shorter
                        # than original string
                        # (unless manual corrections added)
                        assert len(q_item) <= len(string)

                        if len(q_item) == len(string):
                            assert q_item == string
                            item_dist = len(sc_item) - len(q_item)

                        # item in suggestions list should not be the same as
                        # the string itself
                        assert sc_item != string

                        # calculate edit distance using, for example,
                        # Damerau-Levenshtein distance
                        item_dist = dameraulevenshtein(sc_item, string)

                        # do not add words with greater edit distance if
                        # verbose setting not on
                        if (self.verbose < 2) and (item_dist > min_suggest_len):
                            pass
                        elif item_dist <= self.max_edit_distance:
                            assert sc_item in self.dictionary  # should already be in dictionary if in suggestion list
                            suggest_dict[sc_item] = (self.dictionary[sc_item][1], item_dist)
                            if item_dist < min_suggest_len:
                                min_suggest_len = item_dist

                        # depending on order words are processed, some words
                        # with different edit distances may be entered into
                        # suggestions; trim suggestion dictionary if verbose
                        # setting not on
                        if self.verbose < 2:
                            suggest_dict = {k: v for k, v in suggest_dict.items() if v[1] <= min_suggest_len}

            # now generate deletes (e.g. a substring of string or of a delete)
            # from the queue item
            # as additional items to check -- add to end of queue
            assert len(string) >= len(q_item)

            # do not add words with greater edit distance if verbose setting
            # is not on
            if (self.verbose < 2) and ((len(string) - len(q_item)) > min_suggest_len):
                pass
            elif (len(string) - len(q_item)) < self.max_edit_distance and len(q_item) > 1:
                for c in range(len(q_item)):  # character index
                    word_minus_c = q_item[:c] + q_item[c + 1:]
                    if word_minus_c not in q_dictionary:
                        queue.append(word_minus_c)
                        q_dictionary[word_minus_c] = None  # arbitrary value, just to identify we checked this

        # queue is now empty: convert suggestions in dictionary to
        # list for output
        if not silent and self.verbose != 0:
            print("number of possible corrections: %i" % len(suggest_dict))
            print("  edit distance for deletions: %i" % self.max_edit_distance)

        # output option 1
        # sort results by ascending order of edit distance and descending
        # order of frequency
        #     and return list of suggested word corrections only:
        # return sorted(suggest_dict, key = lambda x:
        #               (suggest_dict[x][1], -suggest_dict[x][0]))

        # output option 2
        # return list of suggestions with (correction,
        #                                  (frequency in corpus, edit distance)):
        as_list = suggest_dict.items()
        # outlist = sorted(as_list, key=lambda (term, (freq, dist)): (dist, -freq))
        outlist = sorted(as_list, key=lambda x: (x[1][1], -x[1][0]))

        if self.verbose == 0:
            return outlist[0]
        else:
            return outlist

        '''
        Option 1:
        ['file', 'five', 'fire', 'fine', ...]

        Option 2:
        [('file', (5, 0)),
         ('five', (67, 1)),
         ('fire', (54, 1)),
         ('fine', (17, 1))...]  
        '''

    def best_word(self, s, silent=False):
        try:
            return self.get_suggestions(s, silent)[0]
        except:
            return None


def brands_filling(dataset):
    vc = dataset['brand_name'].value_counts()
    brands = vc[vc > 0].index
    brand_word = r"[a-z0-9*/+\-'’?!.,|&%®™ôèéü]+"

    many_w_brands = brands[brands.str.contains(' ')]
    one_w_brands = brands[~brands.str.contains(' ')]

    ss2 = SymSpell(max_edit_distance=0)
    ss2.create_dictionary_from_arr(many_w_brands, token_pattern=r'.+')

    ss1 = SymSpell(max_edit_distance=0)
    ss1.create_dictionary_from_arr(one_w_brands, token_pattern=r'.+')

    two_words_re = re.compile(r"(?=(\s[a-z0-9*/+\-'’?!.,|&%®™ôèéü]+\s[a-z0-9*/+\-'’?!.,|&%®™ôèéü]+))")

    def find_in_str_ss2(row):
        for doc_word in two_words_re.finditer(row):
            print(doc_word)
            suggestion = ss2.best_word(doc_word.group(1), silent=True)
            if suggestion is not None:
                return doc_word.group(1)
        return ''

    def find_in_list_ss1(list):
        for doc_word in list:
            suggestion = ss1.best_word(doc_word, silent=True)
            if suggestion is not None:
                return doc_word
        return ''

    def find_in_list_ss2(list):
        for doc_word in list:
            suggestion = ss2.best_word(doc_word, silent=True)
            if suggestion is not None:
                return doc_word
        return ''

    print("Before empty brand_name: {}".format(len(dataset[dataset['brand_name'] == ''].index)))

    n_name = dataset[dataset['brand_name'] == '']['name'].str.findall(
        pat=r"^[a-z0-9*/+\-'’?!.,|&%®™ôèéü]+\s[a-z0-9*/+\-'’?!.,|&%®™ôèéü]+")
    dataset.loc[dataset['brand_name'] == '', 'brand_name'] = [find_in_list_ss2(row) for row in n_name]

    n_desc = dataset[dataset['brand_name'] == '']['item_description'].str.findall(
        pat=r"^[a-z0-9*/+\-'’?!.,|&%®™ôèéü]+\s[a-z0-9*/+\-'’?!.,|&%®™ôèéü]+")
    dataset.loc[dataset['brand_name'] == '', 'brand_name'] = [find_in_list_ss2(row) for row in n_desc]

    n_name = dataset[dataset['brand_name'] == '']['name'].str.findall(pat=brand_word)
    dataset.loc[dataset['brand_name'] == '', 'brand_name'] = [find_in_list_ss1(row) for row in n_name]

    desc_lower = dataset[dataset['brand_name'] == '']['item_description'].str.findall(pat=brand_word)
    dataset.loc[dataset['brand_name'] == '', 'brand_name'] = [find_in_list_ss1(row) for row in desc_lower]

    print("After empty brand_name: {}".format(len(dataset[dataset['brand_name'] == ''].index)))

    del ss1, ss2
    gc.collect()


# 对文本数据进行正则化处理，把某些字符串连接在一起
def preprocess_regex(dataset):
    karats_regex = r'(\d)([\s-]?)(karat|karats|carat|carats|kt)([^\w])'
    karats_repl = r'\1k\4'

    unit_regex = r'(\d+)[\s-]([a-z]{2})(\s)'
    unit_repl = r'\1\2\3'

    dataset['name'] = dataset['name'].str.replace(karats_regex, karats_repl)
    dataset['item_description'] = dataset['item_description'].str.replace(karats_regex, karats_repl)
    print('[{time() - start_time}] Karats normalized.')

    dataset['name'] = dataset['name'].str.replace(unit_regex, unit_repl)
    dataset['item_description'] = dataset['item_description'].str.replace(unit_regex, unit_repl)
    print('[{time() - start_time}] Units glued.')


class DataReader():

    def __init__(self, local_flag:bool):
        TRAIN_FILE = "../input/train.tsv"
        TEST_FILE = "../input/test.tsv"
        self.local_flag = local_flag

        if local_flag:
            train_df = pd.read_csv("../" + TRAIN_FILE, sep='\t', engine='python')#, nrows=10000)
            test_df = pd.read_csv("../" + TEST_FILE, sep='\t', engine='python')#, nrows=3000)
        else:
            train_df = pd.read_csv(TRAIN_FILE, sep='\t')
            test_df = pd.read_csv(TEST_FILE, sep='\t')

        record_log(local_flag, 'Remain price!=0 items')
        train_df = train_df[train_df['price'] != 0]

        # brand填充
        train_df['brand_name'].fillna(value="", inplace=True)
        test_df['brand_name'].fillna(value="", inplace=True)

        # category fillna
        train_df['category_name'].fillna(value="Other", inplace=True)
        test_df['category_name'].fillna(value="Other", inplace=True)

        # description fillna
        train_df['item_description'].fillna(value="None", inplace=True)
        test_df['item_description'].fillna(value="None", inplace=True)
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

        # 以防万一的填充
        train_df['name'].fillna(value='', inplace=True)
        test_df['name'].fillna(value='', inplace=True)

        # Add new strategy of feature engineering
        # 1 "cat+cond"
        train_df['cat_cond'] = train_df['category_name'] + '_' + train_df['item_condition_id'].astype(str)
        test_df['cat_cond'] = test_df['category_name'] + '_' + test_df['item_condition_id'].astype(str)
        # 2 Regexlize 'name' and 'desc'
        preprocess_regex(train_df)  # 连接特定的字符串
        preprocess_regex(test_df)  # 连接特定的字符串
        # 3 fill brand
        merge = pd.concat([train_df, test_df])  # 合并训练集和测试集
        brands_filling(merge)
        train_df['brand_name'] = merge.iloc[:train_df.shape[0]]['brand_name']
        test_df['brand_name'] = merge.iloc[train_df.shape[0]:]['brand_name']
        del merge
        gc.collect()



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

        # LabelEncoder cat_cond
        le.fit(np.hstack([self.train_df['cat_cond'], self.test_df['cat_cond']]))
        self.train_df['cat_cond_le'] = le.transform(self.train_df['cat_cond'])
        self.test_df['cat_cond_le'] = le.transform(self.test_df['cat_cond'])

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
        self.n_cat_cond = np.max([self.train_df.cat_cond_le.max(), self.test_df.cat_cond_le.max()]) + 1  # LE编码后最大值+1
        self.n_brand = np.max([self.train_df.brand_le.max(), self.test_df.brand_le.max()])+1
        self.n_condition_id = np.max([self.train_df.item_condition_id.max(), self.test_df.item_condition_id.max()])+1
        self.n_desc_max_len = np.max([self.train_df.desc_len.max(), self.test_df.desc_len.max()]) + 1
        print("#self.train_df.desc_npc_cnt.max() =", self.train_df.desc_npc_cnt.max())
        print("#self.test_df.desc_npc_cnt.max() =", self.test_df.desc_npc_cnt.max())
        self.n_npc_max_cnt = np.max([self.train_df.desc_npc_cnt.max(), self.test_df.desc_npc_cnt.max()]) + 1
        print("#n_category:{},n_brand:{},n_condition_id:{},n_desc_max_len:{}".format(self.n_category, self.n_brand, self.n_condition_id, self.n_desc_max_len))
        print("#n_cat_cond:{}".format(self.n_cat_cond))

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
            'cat_cond': np.array(dataset.cat_cond_le),
            'item_condition': np.array(dataset.item_condition_id),
            'num_vars': np.array(dataset[['shipping']]),
            'desc_len': np.array(dataset[["desc_len"]]),
            'desc_npc_cnt': np.array(dataset[["desc_npc_cnt"]]),
        }
        return X

    def del_redundant_cols(self):
        useful_cols = ['train_id', 'test_id', 'name', 'item_condition_id', 'category_name', 'cat_cond', 'brand_name', 'price', 'shipping',
                       'item_description', 'category_le', 'cat_cond_le',
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
                ngram_range=(1, 3),
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




