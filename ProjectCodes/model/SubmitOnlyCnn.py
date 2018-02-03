#changes:
#optimizing RNN and Ridge again
#based on https://www.kaggle.com/valkling/mercari-rnn-2ridge-models-with-notes-0-42755
#required libraries
import gc
import numpy as np
import pandas as pd

from datetime import datetime
start_real = datetime.now()

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dropout, Dense, concatenate, GRU, Embedding, Flatten, Activation, Conv1D, \
    BatchNormalization, MaxPooling1D, Concatenate
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K
import time
import re
from nltk.corpus import stopwords
import math

# set seed
np.random.seed(123)

BN_FLAG = False

USE_NAME_BRAND_MAP = True
CNN_VERBOSE = 10
EXPM1_before_MUL_Best12_Flag = True
SPEED_UP = False
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
train_df = pd.read_table('../input/train.tsv')
test_df = pd.read_table('../input/test.tsv')
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


#constants to use in CNN model
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


#transform the data for CNN model
def get_cnn_data(dataset):
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

X_train = get_cnn_data(train)
Y_train = train.target.values.reshape(-1, 1)

X_dev = get_cnn_data(dev)
Y_dev = dev.target.values.reshape(-1, 1)

X_test = get_cnn_data(test)


#our own loss function
def root_mean_squared_logarithmic_error(y_true, y_pred):
    first_log = K.log(K.clip(y_pred, K.epsilon(), None) + 1.)
    second_log = K.log(K.clip(y_true, K.epsilon(), None) + 1.)
    return K.sqrt(K.mean(K.square(first_log - second_log), axis=-1)+0.0000001)
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)+0.0000001)


# build the model
np.random.seed(123)


def new_cnn_model(lr=0.001, decay=0.0):
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

    # CNN: Use Conv1D and MaxPooling1D(or GlobalMaxPooling1D)
    def cnn_layer_output(filter_size_list, num_filters_list, strides_list, pool_size_list, emb_words):
        conv_blocks = []
        assert len(filter_size_list) == len(num_filters_list) and len(num_filters_list) == len(strides_list) and len(
            strides_list) == len(pool_size_list)
        for i in range(len(filter_size_list)):
            # [Input] (samples_n, time_n, dim) -> [Output] (samples_n, new_time_n, filter_n)
            conv = Conv1D(filters=num_filters_list[i],
                          kernel_size=filter_size_list[i],
                          padding="valid",
                          strides=strides_list[i])(emb_words)
            if BN_FLAG:
                conv = BatchNormalization()(conv)
            conv = Activation(activation="relu")(conv)
            # [Input] (samples_n, time_n, feat_n) -> [Output] (samples_n, down_time_n, feat_n)
            conv = MaxPooling1D(pool_size=pool_size_list[i])(conv)
            conv = Flatten()(conv)
            conv_blocks.append(conv)
        return Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
    cnn_layer_name = cnn_layer_output(filter_size_list=[2, 4], num_filters_list=[7, 7], strides_list=[1, 1],
                                      pool_size_list=[2, 2], emb_words=emb_name)
    cnn_layer_item_desc = cnn_layer_output(filter_size_list=[2, 4], num_filters_list=[10, 10], strides_list=[1, 1],
                                           pool_size_list=[2, 2], emb_words=emb_item_desc)

    # main layers
    main_l = concatenate([
        Flatten()(emb_brand_name),
        Flatten()(emb_item_condition),
        Flatten()(emb_desc_len),
        Flatten()(emb_name_len),
        Flatten()(emb_desc_npc_cnt),
        Flatten()(emb_subcat_0),
        Flatten()(emb_subcat_1),
        Flatten()(emb_subcat_2),
        cnn_layer_name,
        cnn_layer_item_desc,
        num_vars,
    ])

    # (incressing the nodes or adding layers does not effect the time quite as much as the cnn layers)
    main_l = Dropout(0.1)(Dense(512, kernel_initializer='normal', activation='relu')(main_l))
    main_l = Dropout(0.1)(Dense(256, kernel_initializer='normal', activation='relu')(main_l))
    main_l = Dropout(0.1)(Dense(128, kernel_initializer='normal', activation='relu')(main_l))
    main_l = Dropout(0.1)(Dense(64, kernel_initializer='normal', activation='relu')(main_l))

    # the output layer.
    output = Dense(1, activation="linear")(main_l)

    model = Model([name, item_desc, brand_name, item_condition,
                   num_vars, desc_len, name_len, desc_npc_cnt, subcat_0, subcat_1, subcat_2], output)

    optimizer = Adam(lr=lr, decay=decay)

    # (mean squared error loss function works as well as custom functions)
    model.compile(loss='mse', optimizer=optimizer)

    return model


model = new_cnn_model()
model.summary()
del model


#Fit CNN model to train data

# Set hyper parameters for the model
BATCH_SIZE = 512 * 3
epochs = 2

# Calculate learning rate decay
exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1
steps = int(len(X_train['name']) / BATCH_SIZE) * epochs
lr_init, lr_fin = 0.005, 0.001
lr_decay = exp_decay(lr_init, lr_fin, steps)

# Create model and fit it with training dataset.
cnn_model = new_cnn_model(lr=lr_init, decay=lr_decay)
cnn_model.fit(X_train, Y_train, epochs=epochs, batch_size=BATCH_SIZE,validation_data=(X_dev, Y_dev), verbose=CNN_VERBOSE)
elapsed = time_measure("cnn_model.fit()", start, elapsed)


#Define RMSL Error Function for checking prediction
def rmsle(Y, Y_pred):
    assert Y.shape == Y_pred.shape
    return np.sqrt(np.mean(np.square(Y_pred - Y )))


print("Evaluating the model on validation data...")
Y_dev_preds_cnn = cnn_model.predict(X_dev, batch_size=BATCH_SIZE)
dev_best_rmsle = rmsle(Y_dev, Y_dev_preds_cnn)
print(" RMSLE error:", dev_best_rmsle)


#prediction for test data
cnn_preds = cnn_model.predict(X_test, batch_size=BATCH_SIZE, verbose=CNN_VERBOSE)
if EXPM1_before_MUL_Best12_Flag:
    cnn_preds = np.expm1(cnn_preds)
elapsed = time_measure("cnn_model.predict()", start, elapsed)


# best predicted submission
preds = cnn_preds
submission = pd.DataFrame({"test_id": test_df.test_id, "price": preds.reshape(-1)}, columns=['test_id', 'price'])
# submission.to_csv("./cnn_ridge_submission.csv", index=False)
submission.to_csv("./onlyCNN_DevBestRmsle_{:.5f}_.csv".format(dev_best_rmsle), index=False)
print("completed time:")
stop_real = datetime.now()
execution_time_real = stop_real-start_real
print(execution_time_real)






