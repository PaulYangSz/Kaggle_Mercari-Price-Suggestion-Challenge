#changes:
#optimizing RNN and Ridge again
#based on https://www.kaggle.com/valkling/mercari-rnn-2ridge-models-with-notes-0-42755
#required libraries
import gc
import numpy as np
import pandas as pd

from datetime import datetime
start_real = datetime.now()

from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dropout, Dense, concatenate, GRU, Embedding, Flatten, Activation, BatchNormalization
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K
import time
import re
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix, hstack
import math
import sys
sys.path.insert(0, '../input/wordbatch/wordbatch/')
import wordbatch

from wordbatch.extractors import WordBag, WordHash
from wordbatch.models import FTRL, FM_FTRL

# set seed
np.random.seed(123)
BN_FLAG = True
USE_NAME_BRAND_MAP = True
RNN_VERBOSE = 10
SPEED_UP = True
if SPEED_UP:
    import pyximport
    pyximport.install()
    import os
    import random
    import tensorflow as tf
    # os.environ['PYTHONHASHSEED'] = '10000'
    # np.random.seed(10001)
    # random.seed(10002)
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=5, inter_op_parallelism_threads=1)
    from keras import backend
    # tf.set_random_seed(10003)
    backend.set_session(tf.Session(graph=tf.get_default_graph(), config=session_conf))


def time_measure(section, start, elapsed):
    lap = time.time() - start - elapsed
    elapsed = time.time() - start
    print("{:60}: {:15.2f}[sec]{:15.2f}[sec]".format(section, lap, elapsed))
    return elapsed

start = time.time()
elapsed = 0
#Load the train and test data
train_df = pd.read_table('../input/mercari-price-suggestion-challenge/train.tsv')
test_df = pd.read_table('../input/mercari-price-suggestion-challenge/test.tsv')
#check the shape of the dataframes
print('train:',train_df.shape, ',test:',test_df.shape)
elapsed = time_measure("load data", start, elapsed)

# removing prices less than 3
train_df = train_df.drop(train_df[(train_df.price < 3.0)].index)
print('After drop pricee < 3.0{}'.format(train_df.shape))

# 品牌名中有stopwords，所以不能对name做操作，否则会影响后面的->map效果
# stop_patten = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')  # 会把Burt's Bees匹配到
# del stop_patten
stopwords_list = stopwords.words('english')
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
train_df.loc[:, 'item_description'] = train_df['item_description'].map(lambda x: fill_item_description_null(x, ''))
test_df.loc[:, 'item_description'] = test_df['item_description'].map(lambda x: fill_item_description_null(x, ''))
elapsed = time_measure("item_description fill_(include normalize)", start, elapsed)


# 尝试下对name只做normal但是不去停止词
def normal_name(name):
    try:
        normal_text = " ".join(list(map(lambda x: x[0], word_patten.findall(name))))
        return normal_text
    except:
        return ''
train_df.loc[:, 'name'] = train_df['name'].map(normal_name)
test_df.loc[:, 'name'] = test_df['name'].map(normal_name)
elapsed = time_measure("normal_name without stopwords ", start, elapsed)


npc_patten = re.compile(r'!')
# handling categorical variables
def patten_count(text, patten_):
    try:
        # text = text.lower()
        return len(patten_.findall(text))
    except:
        return 0
train_df['desc_npc_cnt'] = train_df['item_description'].apply(lambda x: patten_count(x, npc_patten))
test_df['desc_npc_cnt'] = test_df['item_description'].apply(lambda x: patten_count(x, npc_patten))
elapsed = time_measure("Statistic NPC count", start, elapsed)

#splitting category_name into subcategories
train_df.category_name.fillna(value="missing/missing/missing", inplace=True)
test_df.category_name.fillna(value="missing/missing/missing", inplace=True)
def split_cat(text):
    try: return text.split("/")
    except: return ("missing", "missing", "missing")
train_df['subcat_0'], train_df['subcat_1'], train_df['subcat_2'] = zip(*train_df['category_name'].apply(lambda x: split_cat(x)))
test_df['subcat_0'], test_df['subcat_1'], test_df['subcat_2'] = zip(*test_df['category_name'].apply(lambda x: split_cat(x)))
elapsed = time_measure("wordCount() and split_cat()", start, elapsed)

#combine the train and test dataframes
full_set = pd.concat([train_df,test_df])
if USE_NAME_BRAND_MAP:
    def do_col2brand_dict(data_df: pd.DataFrame, key_col: str):
        group_by_key_to_brandset_ser = data_df['brand_name'].groupby(data_df[key_col]).apply(lambda x: set(x.values))
        only_one_brand_ser = group_by_key_to_brandset_ser[group_by_key_to_brandset_ser.map(len) == 1]
        return only_one_brand_ser.map(lambda x: x.pop()).to_dict()


    def get_brand_by_key(key, map):
        if key in map:
            return map[key]
        else:
            return 'paulnull'


    col_key = 'name'
    have_brand_df = full_set[~full_set['brand_name'].isnull()].copy()
    train_brand_null_index = train_df[train_df['brand_name'].isnull()].index
    test_brand_null_index = test_df[test_df['brand_name'].isnull()].index
    key2brand_map = do_col2brand_dict(data_df=have_brand_df, key_col=col_key)
    train_df.loc[train_brand_null_index, 'brand_name'] = train_df.loc[train_brand_null_index, col_key].map(
        lambda x: get_brand_by_key(x, key2brand_map))
    test_df.loc[test_brand_null_index, 'brand_name'] = test_df.loc[test_brand_null_index, col_key].map(
        lambda x: get_brand_by_key(x, key2brand_map))
    n_before = train_brand_null_index.size + test_brand_null_index.size
    n_after = (train_df['brand_name'] == 'paulnull').sum() + (test_df['brand_name'] == 'paulnull').sum()
    elapsed = time_measure("Use name -> brand Map", start, elapsed)
    print('填充前有{}个空数据，填充后有{}个空数据，填充了{}个数据的brand'.format(n_before, n_after, n_before - n_after))

    # handling brand_name
    all_brands = set(have_brand_df['brand_name'].values)
    del have_brand_df
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
    elapsed = time_measure("brandfinder()", start, elapsed)
else:
    #handling brand_name
    all_brands = set(full_set['brand_name'].values)
    #fill NA values
    train_df.brand_name.fillna(value="missing", inplace=True)
    test_df.brand_name.fillna(value="missing", inplace=True)
    premissing = len(train_df.loc[train_df['brand_name'] == 'missing'])
    def brandfinder(line):
        brand = line[0]
        name = line[1]
        namesplit = name.split(' ')
        if brand == 'missing':
            for x in namesplit:
                if x in all_brands:
                    return name
        if name in all_brands:
            return name
        return brand
    train_df['brand_name'] = train_df[['brand_name','name']].apply(brandfinder, axis = 1)
    test_df['brand_name'] = test_df[['brand_name','name']].apply(brandfinder, axis = 1)
    found = premissing-len(train_df.loc[train_df['brand_name'] == 'missing'])
    elapsed = time_measure("brandfinder()", start, elapsed)
print(found)
del full_set
gc.collect()


# Scale target variable-price to log
train_df["target"] = np.log1p(train_df.price)
# Split training examples into train/dev
train_df, dev_df = train_test_split(train_df, random_state=123, test_size=0.01)
# Calculate number of train/dev/test examples.
n_trains = train_df.shape[0]
n_devs = dev_df.shape[0]
n_tests = test_df.shape[0]
print("Training on:", n_trains, "examples")
print("Validating on:", n_devs, "examples")
print("Testing on:", n_tests, "examples")
elapsed = time_measure("target & train_test_split", start, elapsed)


# Concatenate train - dev - test data for easy to handle
full_df = pd.concat([train_df, dev_df, test_df])


print("Processing categorical data...")
le = LabelEncoder()
le.fit(full_df.brand_name)
full_df.brand_name = le.transform(full_df.brand_name)

le.fit(full_df.subcat_0)
full_df.subcat_0 = le.transform(full_df.subcat_0)
le.fit(full_df.subcat_1)
full_df.subcat_1 = le.transform(full_df.subcat_1)
le.fit(full_df.subcat_2)
full_df.subcat_2 = le.transform(full_df.subcat_2)
del le
elapsed = time_measure("LabelEncoder(brand_name & subcat0/1/2)", start, elapsed)


print("Transforming text data to sequences...")
name_raw_text = np.hstack([full_df.name.str.lower()])
desc_raw_text = np.hstack([full_df.item_description.str.lower()])

print("Fitting tokenizer...")
name_tok_raw = Tokenizer(num_words=150000, filters='\t\n')
desc_tok_raw = Tokenizer(num_words=300000, filters='\t\n')  # 使用filter然后split。会导致T-Shirt，hi-tech这种词被误操作
name_tok_raw.fit_on_texts(name_raw_text)
desc_tok_raw.fit_on_texts(desc_raw_text)

print("Transforming text to sequences...")
full_df['seq_item_description'] = desc_tok_raw.texts_to_sequences(full_df.item_description.str.lower())
full_df['seq_name'] = name_tok_raw.texts_to_sequences(full_df.name.str.lower())
full_df['desc_len'] = full_df['seq_item_description'].apply(len)
train_df['desc_len'] = full_df[:n_trains]['desc_len']
dev_df['desc_len'] = full_df[n_trains:n_trains+n_devs]['desc_len']
test_df['desc_len'] = full_df[n_trains+n_devs:]['desc_len']
full_df['name_len'] = full_df['seq_name'].apply(len)
train_df['name_len'] = full_df[:n_trains]['name_len']
dev_df['name_len'] = full_df[n_trains:n_trains+n_devs]['name_len']
test_df['name_len'] = full_df[n_trains+n_devs:]['name_len']
elapsed = time_measure("tok_raw.texts_to_sequences(name & desc)", start, elapsed)


#constants to use in RNN model
MAX_NAME_SEQ = 10
MAX_ITEM_DESC_SEQ = 75
MAX_CATEGORY_SEQ = 8
MAX_NAME_DICT_WORDS = min(max(name_tok_raw.word_index.values()), name_tok_raw.num_words) + 2
MAX_DESC_DICT_WORDS = min(max(desc_tok_raw.word_index.values()), desc_tok_raw.num_words) + 2
del name_tok_raw, desc_tok_raw
MAX_BRAND = np.max(full_df.brand_name.max()) + 1
MAX_CONDITION = np.max(full_df.item_condition_id.max()) + 1
MAX_DESC_LEN = np.max(full_df.desc_len.max()) + 1
MAX_NAME_LEN = np.max(full_df.name_len.max()) + 1
MAX_NPC_LEN = np.max(full_df.desc_npc_cnt.max()) + 1
MAX_SUBCAT_0 = np.max(full_df.subcat_0.max()) + 1
MAX_SUBCAT_1 = np.max(full_df.subcat_1.max()) + 1
MAX_SUBCAT_2 = np.max(full_df.subcat_2.max()) + 1


#transform the data for RNN model
def get_rnn_data(dataset):
    X = {
        'name': pad_sequences(dataset.seq_name, maxlen=MAX_NAME_SEQ),
        'item_desc': pad_sequences(dataset.seq_item_description, maxlen=MAX_ITEM_DESC_SEQ),
        'brand_name': np.array(dataset.brand_name),
        'item_condition': np.array(dataset.item_condition_id),
        'num_vars': np.array(dataset[["shipping"]]),
        'desc_len': np.array(dataset[["desc_len"]]),
        'name_len': np.array(dataset[["name_len"]]),
        'desc_npc_cnt': np.array(dataset[["desc_npc_cnt"]]),
        'subcat_0': np.array(dataset.subcat_0),
        'subcat_1': np.array(dataset.subcat_1),
        'subcat_2': np.array(dataset.subcat_2),
    }
    return X

train = full_df[:n_trains]
dev = full_df[n_trains:n_trains+n_devs]
test = full_df[n_trains+n_devs:]

X_train = get_rnn_data(train)
Y_train = train.target.values.reshape(-1, 1)

X_dev = get_rnn_data(dev)
Y_dev = dev.target.values.reshape(-1, 1)

X_test = get_rnn_data(test)


#our own loss function
# def root_mean_squared_logarithmic_error(y_true, y_pred):
#     first_log = K.log(K.clip(y_pred, K.epsilon(), None) + 1.)
#     second_log = K.log(K.clip(y_true, K.epsilon(), None) + 1.)
#     return K.sqrt(K.mean(K.square(first_log - second_log), axis=-1)+0.0000001)
# def root_mean_squared_error(y_true, y_pred):
#     return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)+0.0000001)


# build the model
np.random.seed(123)


def new_rnn_model(lr=0.001, decay=0.0):
    name = Input(shape=[X_train["name"].shape[1]], name="name")
    item_desc = Input(shape=[X_train["item_desc"].shape[1]], name="item_desc")
    brand_name = Input(shape=[1], name="brand_name")
    item_condition = Input(shape=[1], name="item_condition")
    num_vars = Input(shape=[X_train["num_vars"].shape[1]], name="num_vars")
    desc_len = Input(shape=[1], name="desc_len")
    name_len = Input(shape=[1], name="name_len")
    desc_npc_cnt = Input(shape=[1], name="desc_npc_cnt")
    subcat_0 = Input(shape=[1], name="subcat_0")
    subcat_1 = Input(shape=[1], name="subcat_1")
    subcat_2 = Input(shape=[1], name="subcat_2")

    # Embeddings layers (adjust outputs to help model)
    emb_name = Embedding(MAX_NAME_DICT_WORDS, 20)(name)
    emb_item_desc = Embedding(MAX_DESC_DICT_WORDS, 60)(item_desc)
    emb_brand_name = Embedding(MAX_BRAND, 10)(brand_name)
    emb_item_condition = Embedding(MAX_CONDITION, 5)(item_condition)
    emb_desc_len = Embedding(MAX_DESC_LEN, 5)(desc_len)
    emb_name_len = Embedding(MAX_NAME_LEN, 5)(name_len)
    emb_desc_npc_cnt = Embedding(MAX_NPC_LEN, 5)(desc_npc_cnt)
    emb_subcat_0 = Embedding(MAX_SUBCAT_0, 10)(subcat_0)
    emb_subcat_1 = Embedding(MAX_SUBCAT_1, 10)(subcat_1)
    emb_subcat_2 = Embedding(MAX_SUBCAT_2, 10)(subcat_2)

    # rnn layers (GRUs are faster than LSTMs and speed is important here)
    rnn_layer1 = GRU(16)(emb_item_desc)
    rnn_layer2 = GRU(8)(emb_name)

    # main layers
    main_layer = concatenate([
        Flatten()(emb_brand_name),
        Flatten()(emb_item_condition),
        Flatten()(emb_desc_len),
        Flatten()(emb_name_len),
        Flatten()(emb_desc_npc_cnt),
        Flatten()(emb_subcat_0),
        Flatten()(emb_subcat_1),
        Flatten()(emb_subcat_2),
        rnn_layer1,
        rnn_layer2,
        num_vars,
    ])

    # Concat[all] -> Dense1 -> ... -> DenseN
    dense_layers_unit = [512, 256, 128, 64]
    drop_out_layers = [0.1, 0.1, 0.1, 0.1]
    for i in range(len(dense_layers_unit)):
        main_layer = Dense(dense_layers_unit[i])(main_layer)
        if BN_FLAG:
            main_layer = BatchNormalization()(main_layer)
        main_layer = Activation(activation='relu')(main_layer)
        main_layer = Dropout(drop_out_layers[i])(main_layer)
    # (increasing the nodes or adding layers does not effect the time quite as much as the rnn layers)

    # the output layer.
    output = Dense(1, activation="linear")(main_layer)

    model = Model([name, item_desc, brand_name, item_condition,
                   num_vars, desc_len, name_len, desc_npc_cnt, subcat_0, subcat_1, subcat_2], output)

    optimizer = Adam(lr=lr, decay=decay)

    # (mean squared error loss function works as well as custom functions)
    model.compile(loss='mse', optimizer=optimizer)

    return model


model = new_rnn_model()
model.summary()
del model


#Fit RNN model to train data

# Set hyper parameters for the model
BATCH_SIZE = 512 * 3
epochs = 2

# Calculate learning rate decay
exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1
steps = int(len(X_train['name']) / BATCH_SIZE) * epochs
lr_init, lr_fin = 0.01485, 0.00056
lr_decay = exp_decay(lr_init, lr_fin, steps)

# Create model and fit it with training dataset.
rnn_model = new_rnn_model(lr=lr_init, decay=lr_decay)
rnn_model.fit(X_train, Y_train, epochs=epochs, batch_size=BATCH_SIZE,validation_data=(X_dev, Y_dev), verbose=RNN_VERBOSE)
elapsed = time_measure("rnn_model.fit()", start, elapsed)


#Define RMSL Error Function for checking prediction
def rmsle(Y, Y_pred):
    assert Y.shape == Y_pred.shape
    return np.sqrt(np.mean(np.square(Y_pred - Y )))


print("Evaluating the model on validation data...")
Y_dev_preds_rnn = rnn_model.predict(X_dev, batch_size=BATCH_SIZE)
print("RNN RMSLE error:", rmsle(Y_dev, Y_dev_preds_rnn))


#prediction for test data
rnn_preds = rnn_model.predict(X_test, batch_size=BATCH_SIZE, verbose=RNN_VERBOSE)
rnn_preds = np.expm1(rnn_preds)
elapsed = time_measure("rnn_model.predict()", start, elapsed)



#=======================================
print('+'*60)
#WordBatch modelling
def normalize_text(text):
    return text
wb = wordbatch.WordBatch(normalize_text, extractor=(WordBag, {"hash_ngrams": 2, "hash_ngrams_weights": [1.5, 1.0],
                                                              "hash_size": 2 ** 22, "norm": None, "tf": 'binary',
                                                              "idf": None,}), procs=8, n_words=500000)
wb.dictionary_freeze= True
wb.fit(train_df['name'])
X_name = wb.transform(full_df['name'])
del(wb)
X_name = X_name[:, np.array(np.clip(X_name.getnnz(axis=0) - 1, 0, 1), dtype=bool)]
elapsed = time_measure("X_name: Vectorize 'name' completed. shape={}".format(X_name.shape), start, elapsed)


wb = CountVectorizer()
X_category1 = wb.fit_transform(full_df['subcat_0'].astype(str))
X_category2 = wb.fit_transform(full_df['subcat_1'].astype(str))
X_category3 = wb.fit_transform(full_df['subcat_2'].astype(str))
del(wb)
elapsed = time_measure("X_category: Vectorize 'categories 0/1/2' completed.", start, elapsed)


wb = wordbatch.WordBatch(normalize_text, extractor=(WordBag, {"hash_ngrams": 2, "hash_ngrams_weights": [1.0, 1.0],
                                                              "hash_size": 2 ** 22, "norm": "l2", "tf": 1.0,
                                                              "idf": None}), procs=8, n_words=2000000)
wb.dictionary_freeze= True
wb.fit(train_df['item_description'])
X_description = wb.transform(full_df['item_description'])
del(wb)
X_description = X_description[:, np.array(np.clip(X_description.getnnz(axis=0) - 1, 0, 1), dtype=bool)]
elapsed = time_measure("X_description: Vectorize 'item_description' completed. shape={}".format(X_description.shape), start, elapsed)


lb = LabelBinarizer(sparse_output=True)
X_brand = lb.fit_transform(full_df['brand_name'])
del(lb)
elapsed = time_measure("X_brand: Vectorize 'brand_name' completed. shape={}".format(X_brand.shape), start, elapsed)


X_dummies = csr_matrix(pd.get_dummies(full_df[['item_condition_id', 'shipping']].astype(str), sparse=True).values)
elapsed = time_measure("X_dummies: Vectorize 'condition_id & shipping' completed. {}".format(X_dummies.shape), start, elapsed)


sparse_merge = hstack((X_dummies, X_description, X_brand, X_category1, X_category2, X_category3, X_name)).tocsr()
elapsed = time_measure("Create sparse merge completed, shape={}".format(sparse_merge.shape), start, elapsed)
mask = np.array(np.clip(sparse_merge.getnnz(axis=0) - 1, 0, 1), dtype=bool)
sparse_merge = sparse_merge[:, mask]
elapsed = time_measure("Remove features with document frequency <=1, shape={}".format(sparse_merge.shape), start, elapsed)


X_train = sparse_merge[:n_trains]
Y_train = train_df.target.values
X_dev = sparse_merge[n_trains:n_trains+n_devs]
Y_dev = dev_df.target.values.reshape(-1, 1)
X_test = sparse_merge[n_trains+n_devs:]


print("Fitting FTRL model on training examples...")
ftrl_model = FTRL(alpha=0.01, beta=0.1, L1=0.00001, L2=1.0, D=sparse_merge.shape[1], iters=50, inv_link="identity", threads=1)
ftrl_model.fit(X_train, Y_train)
elapsed = time_measure("FTRL().fit()", start, elapsed)
# dev data
Y_dev_preds_ftrl = ftrl_model.predict(X=X_dev)
Y_dev_preds_ftrl = Y_dev_preds_ftrl.reshape(-1, 1)
print("FTRL RMSL error on dev set:", rmsle(Y_dev, Y_dev_preds_ftrl))
# prediction for test data
ftrl_preds = ftrl_model.predict(X_test)
ftrl_preds = ftrl_preds.reshape(-1, 1)
ftrl_preds = np.expm1(ftrl_preds)


print("Fitting Ridge model on training examples...")
ridge_model = Ridge(solver='auto', fit_intercept=True, alpha=5.0,max_iter=None, normalize=False, tol=0.05, random_state = 1)
# ridge_modelCV = Ridge(solver='auto', fit_intercept=True, alpha=5.0,max_iter=None, normalize=False, tol=0.05, random_state = 1)
# ridge_modelCV = RidgeCV(fit_intercept=True, alphas=[5.0], normalize=False, cv = 2, scoring='neg_mean_squared_error')
ridge_model.fit(X_train, Y_train)
elapsed = time_measure("Ridge.fit()--RidgeCV.fit()", start, elapsed)
#Evaluating Ridge model on dev data
Y_dev_preds_ridge = ridge_model.predict(X_dev)
Y_dev_preds_ridge = Y_dev_preds_ridge.reshape(-1, 1)
print("Ridge RMSL error on dev set:", rmsle(Y_dev, Y_dev_preds_ridge))
#prediction for test data
ridge_preds = ridge_model.predict(X_test)
ridge_preds = ridge_preds.reshape(-1, 1)
ridge_preds = np.expm1(ridge_preds)


#combine all predictions
def aggregate_predicts3(Y1, Y2, Y3, ratio1, ratio2):
    # print("{},{},{}".format(Y1.shape, Y2.shape, Y3.shape))
    assert Y1.shape == Y2.shape
    return Y1 * ratio1 + Y2 * ratio2 + Y3 * (1.0 - ratio1-ratio2)


#ratio optimum finder for 3 models
best1 = 0
best2 = 0
lowest = 0.99
for i in range(100):
    for j in range(100):
        r = i*0.01
        r2 = j*0.01
        if r+r2 < 1.0:
            Y_dev_preds = aggregate_predicts3(Y_dev_preds_rnn, Y_dev_preds_ftrl, Y_dev_preds_ridge, r, r2)
            fpred = rmsle(Y_dev, Y_dev_preds)
            if fpred < lowest:
                best1 = r
                best2 = r2
                lowest = fpred
Y_dev_preds = aggregate_predicts3(Y_dev_preds_rnn, Y_dev_preds_ftrl, Y_dev_preds_ridge, best1, best2)
elapsed = time_measure("aggregate_predicts3() get best coefficients", start, elapsed)

dev_best_rmsle = rmsle(Y_dev, Y_dev_preds)
print("(Best) RMSL error for RNN + Ridge + RidgeCV on dev set:", dev_best_rmsle)


# best predicted submission
preds = aggregate_predicts3(rnn_preds, ftrl_preds, ridge_preds, best1, best2)
submission = pd.DataFrame({"test_id": test_df.test_id, "price": preds.reshape(-1)}, columns=['test_id', 'price'])
# submission.to_csv("./rnn_ridge_submission.csv", index=False)
submission.to_csv("./best1_{}_best2_{}_DevBestRmsle_{:.5f}_.csv".format(best1,best2,dev_best_rmsle), index=False)
print("completed time:")
stop_real = datetime.now()
execution_time_real = stop_real-start_real
print(execution_time_real)






