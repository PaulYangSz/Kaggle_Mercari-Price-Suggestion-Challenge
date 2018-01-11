# coding: utf-8

# Forked from www.kaggle.com/isaienkov/rnn-with-keras-ridge-sgdr-0-43553/code
# Borrowing some embedding and GRU process idea


import gc
import time
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences  # 默认在前面补零，或者抹掉前面

start_time = time.time()


# TODO: Need modify when run on Kaggle kernel.
train = pd.read_csv('../../input/train.tsv', sep='\t', engine='python')
test = pd.read_csv('../../input/test.tsv', sep='\t', engine='python')


train['target'] = np.log1p(train['price'])


print(train.shape)
print('5 folds scaling the test_df')
print(test.shape)
test_len = test.shape[0]


def simulate_test(test_data):
    if test_data.shape[0] < 800000:
        indices = np.random.choice(test_data.index.values, 2800000)
        test_ = pd.concat([test_data, test_data.iloc[indices]], axis=0)
        return test_.copy()
    else:
        return test_data
# TODO: Need or Not?
test = simulate_test(test)
print('new shape ', test.shape)
print('[{}] Finished scaling test set...'.format(time.time() - start_time))


# HANDLE MISSING VALUES
print("Handling missing values...")


def handle_missing(dataset):
    dataset.category_name.fillna(value="missing", inplace=True)
    dataset.brand_name.fillna(value="missing", inplace=True)
    dataset.item_description.fillna(value="missing", inplace=True)
    return dataset

train = handle_missing(train)
test = handle_missing(test)
print(train.shape)
print(test.shape)
print('[{}] Finished handling missing data...'.format(time.time() - start_time))


# PROCESS CATEGORICAL DATA
# TODO: 需要改变下分类规则然后重新编码尝试结果
print("Handling categorical variables...")
le = LabelEncoder()  # 给字符串或者其他对象编码, 从0开始编码

le.fit(np.hstack([train.category_name, test.category_name]))
train['category'] = le.transform(train.category_name)
test['category'] = le.transform(test.category_name)

le.fit(np.hstack([train.brand_name, test.brand_name]))
train['brand'] = le.transform(train.brand_name)
test['brand'] = le.transform(test.brand_name)
del le, train['brand_name'], test['brand_name']

print('[{}] Finished PROCESSING CATEGORICAL DATA...'.format(time.time() - start_time))
with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None, 'display.height', None):
    print(train.head(3))


# PROCESS TEXT: RAW
print("Text to seq process...")
print("   Fitting tokenizer...")
raw_text = np.hstack([train.category_name.str.lower(),
                      train.item_description.str.lower(),
                      train.name.str.lower()])

tok_raw = Tokenizer()  # 分割文本成词，然后将词转成编码(先分词，后编码)
tok_raw.fit_on_texts(raw_text)
print("   Transforming text to seq...")
train["seq_category_name"] = tok_raw.texts_to_sequences(train.category_name.str.lower())
test["seq_category_name"] = tok_raw.texts_to_sequences(test.category_name.str.lower())
train["seq_item_description"] = tok_raw.texts_to_sequences(train.item_description.str.lower())
test["seq_item_description"] = tok_raw.texts_to_sequences(test.item_description.str.lower())
train["seq_name"] = tok_raw.texts_to_sequences(train.name.str.lower())
test["seq_name"] = tok_raw.texts_to_sequences(test.name.str.lower())
with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None, 'display.height', None):
    print(train.head(3))
print('[{}] Finished PROCESSING TEXT DATA...'.format(time.time() - start_time))


# EXTRACT DEVELOPMENT TEST
dtrain, dvalid = train_test_split(train, random_state=666, train_size=0.99)
print(dtrain.shape)
print(dvalid.shape)


# EMBEDDINGS MAX VALUE
# Base on the histograms, we select the next lengths
# TODO: TimeSteps的长度是否需要改变
MAX_NAME_SEQ = 20  # 17 name列转texts_to_sequences后的list的最大长度，不足会补足，过长会截断
MAX_ITEM_DESC_SEQ = 60  # 269, 同上，是item_description
MAX_CATEGORY_NAME_SEQ = 20  # 8, 同上，是category_name列的
MAX_ALL_TEXT_DICT = np.max([np.max(train.seq_name.max()),
                            np.max(test.seq_name.max()),
                            np.max(train.seq_category_name.max()),
                            np.max(test.seq_category_name.max()),
                            np.max(train.seq_item_description.max()),
                            np.max(test.seq_item_description.max())]) + 2
MAX_CATEGORY = np.max([train.category.max(), test.category.max()]) + 1  # LE编码后最大值+1
MAX_BRAND = np.max([train.brand.max(), test.brand.max()])+1
MAX_CONDITION = np.max([train.item_condition_id.max(),
                        test.item_condition_id.max()])+1
print('[{}] Finished EMBEDDINGS MAX VALUE...'.format(time.time() - start_time))


# KERAS DATA DEFINITION
# name:名字词编号pad列表, item_desc:描述词编号pad列表,
# brand:品牌编号, category:类别编号, category_name:类别词编号pad列表,
# item_condition: item_condition_id, num_vars: shipping
def get_keras_data(dataset):
    X = {
        'name': pad_sequences(dataset.seq_name, maxlen=MAX_NAME_SEQ),
        'item_desc': pad_sequences(dataset.seq_item_description, maxlen=MAX_ITEM_DESC_SEQ),
        'brand': np.array(dataset.brand),
        'category': np.array(dataset.category),
        'category_name': pad_sequences(dataset.seq_category_name, maxlen=MAX_CATEGORY_NAME_SEQ),
        'item_condition': np.array(dataset.item_condition_id),
        'num_vars': np.array(dataset[["shipping"]])
    }
    return X
X_train = get_keras_data(dtrain)
X_valid = get_keras_data(dvalid)
X_test = get_keras_data(test)
print('[{}] Finished DATA PREPARATION...'.format(time.time() - start_time))


# KERAS MODEL DEFINITION
from keras.layers import Input, Dropout, Dense, BatchNormalization, \
    Activation, concatenate, GRU, Embedding, Flatten
from keras.models import Model
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping#, TensorBoard
from keras import backend as K
from keras import optimizers
from keras import initializers
def rmsle(y, y_pred):
    import math
    assert len(y) == len(y_pred)
    to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i, pred in enumerate(y_pred)]
    return (sum(to_sum) * (1.0/len(y))) ** 0.5


dr = 0.25


def get_model():
    # params
    dr_r = dr

    # Inputs
    name = Input(shape=[X_train["name"].shape[1]], name="name")
    item_desc = Input(shape=[X_train["item_desc"].shape[1]], name="item_desc")
    brand = Input(shape=[1], name="brand")
    category = Input(shape=[1], name="category")
    category_name = Input(shape=[X_train["category_name"].shape[1]],
                          name="category_name")
    item_condition = Input(shape=[1], name="item_condition")
    num_vars = Input(shape=[X_train["num_vars"].shape[1]], name="num_vars")

    # Embeddings layers
    emb_size = 60

    # Embedding的作用是配置字典size和词向量len后，根据call参数的indices，返回词向量.
    #  类似TF的embedding_lookup
    #  name.shape=[None, MAX_NAME_SEQ], emb_name.shape=[None, MAX_NAME_SEQ, output_dim]
    emb_name = Embedding(input_dim=MAX_ALL_TEXT_DICT, output_dim=emb_size // 3)(name)
    emb_item_desc = Embedding(MAX_ALL_TEXT_DICT, emb_size)(item_desc)  # [None, MAX_ITEM_DESC_SEQ, emb_size]
    emb_category_name = Embedding(MAX_ALL_TEXT_DICT, emb_size // 3)(category_name)
    emb_brand = Embedding(MAX_BRAND, 10)(brand)
    emb_category = Embedding(MAX_CATEGORY, 10)(category)
    emb_item_condition = Embedding(MAX_CONDITION, 5)(item_condition)

    # GRU是配置一个cell输出的units长度后，根据call词向量入参,输出最后一个GRU cell的输出(因为默认return_sequences=False)
    rnn_layer1 = GRU(units=16)(emb_item_desc)  # rnn_layer1.shape=[None, 16]
    rnn_layer2 = GRU(8)(emb_category_name)
    rnn_layer3 = GRU(8)(emb_name)

    # main layer
    # 连接列表中的Tensor，按照axis组成一个大的Tensor
    main_l = concatenate([Flatten()(emb_brand),   # [None, 1, 10] -> [None, 10]
                          Flatten()(emb_category),
                          Flatten()(emb_item_condition),
                          rnn_layer1,
                          rnn_layer2,
                          rnn_layer3,
                          num_vars])
    # TODO: 全连接隐单元个数和Dropout因子需要调整
    main_l = Dropout(0.25)(Dense(128,activation='relu')(main_l))
    main_l = Dropout(0.1)(Dense(64,activation='relu')(main_l))

    # output
    output = Dense(1, activation="linear")(main_l)

    # model
    model = Model(inputs=[name, item_desc, brand, category, category_name, item_condition, num_vars], outputs=output)
    # optimizer = optimizers.RMSprop()
    optimizer = optimizers.Adam()
    model.compile(loss="mse",
                  optimizer=optimizer)
    return model


def eval_model(model):
    val_preds = model.predict(X_valid)
    val_preds = np.expm1(val_preds)

    y_true = np.array(dvalid.price.values)
    y_pred = val_preds[:, 0]
    v_rmsle = rmsle(y_true, y_pred)
    print(" RMSLE error on dev test: "+str(v_rmsle))
    return v_rmsle

# fin_lr=init_lr * (1/(1+decay))**(steps-1)
exp_decay = lambda init, final, step_num: (init / final) ** (1 / (step_num - 1)) - 1
print('[{}] Finished DEFINEING MODEL...'.format(time.time() - start_time))


gc.collect()


# FITTING THE MODEL
# TODO: 数据训练的轮数等参数需要调整
epochs = 3
BATCH_SIZE = 512 * 3
steps = int(len(X_train['name'])/BATCH_SIZE) * epochs
lr_init, lr_final = 0.015, 0.007
lr_decay = exp_decay(lr_init, lr_final, steps)
log_subdir = '_'.join(['ep', str(epochs),
                       'bs', str(BATCH_SIZE),
                       'lrI', str(lr_init),
                       'lrF', str(lr_final),
                       'dr', str(dr)])
model = get_model()
K.set_value(model.optimizer.lr, lr_init)
K.set_value(model.optimizer.decay, lr_decay)


history = model.fit(X_train, dtrain.target,
                    epochs=epochs,
                    batch_size=BATCH_SIZE,
                    validation_split=0.01,
                    # callbacks=[TensorBoard('./logs/'+log_subdir)],
                    verbose=10
                    )
print('[{}] Finished FITTING MODEL...'.format(time.time() - start_time))


# EVALUATE THE MODEL ON DEV TEST
v_rmsle = eval_model(model)
print('[{}] Finished predicting valid set...'.format(time.time() - start_time))

# CREATE PREDICTIONS
preds = model.predict(X_test, batch_size=BATCH_SIZE)
preds = np.expm1(preds)
print('[{}] Finished predicting test set...'.format(time.time() - start_time))
submission = test[["test_id"]][:test_len]
submission["price"] = preds[:test_len]
submission.to_csv("./myNN"+log_subdir+"_{:.6}.csv".format(v_rmsle), index=False)
print('[{}] Finished submission...'.format(time.time() - start_time))

