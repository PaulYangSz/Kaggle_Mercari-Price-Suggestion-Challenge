# coding: utf-8

# Forked from www.kaggle.com/isaienkov/rnn-with-keras-ridge-sgdr-0-43553/code
# Borrowing some embedding and GRU process idea


import gc
import time
import pandas as pd
import numpy as np

from .DataReader import DataReader

start_time = time.time()


# TODO: Need modify when run on Kaggle kernel.
data_reader = DataReader(local_flag=True, cat_fill_type='fill_paulnull', brand_fill_type='fill_paulnull', item_desc_fill_type='fill_')
# Initial get fillna dataframe
print(data_reader.train_df.shape)
print(data_reader.test_df.shape)
print('[{}] Finished handling missing data...'.format(time.time() - start_time))


# PROCESS CATEGORICAL DATA
# TODO: 需要改变下分类规则然后重新编码尝试结果
print("Handling categorical variables...")
data_reader.le_encode()
print('[{}] Finished PROCESSING CATEGORICAL DATA...'.format(time.time() - start_time))
with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None, 'display.height', None):
    print(data_reader.train_df.head(3))


# PROCESS TEXT: RAW
print("Text to seq process...")
print("   Fitting tokenizer...")
data_reader.tokenizer_text_col()
with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None, 'display.height', None):
    print(data_reader.train_df.head(3))
print('[{}] Finished PROCESSING TEXT DATA...'.format(time.time() - start_time))


# EMBEDDINGS MAX VALUE
# Base on the histograms, we select the next lengths
# TODO: TimeSteps的长度是否需要改变
data_reader.ensure_fixed_value()
print('[{}] Finished EMBEDDINGS MAX VALUE...'.format(time.time() - start_time))


# EXTRACT DEVELOPMENT TEST
dtrain, dvalid, test = data_reader.split_get_train_validation()
print(dtrain.shape)
print(dvalid.shape)


# KERAS DATA DEFINITION
X_train = data_reader.get_keras_data(dtrain)
X_valid = data_reader.get_keras_data(dvalid)
X_test = data_reader.get_keras_data(test)
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
    category_main = Input(shape=[1], name="category_main")
    category_sub = Input(shape=[1], name="category_sub")
    category_sub2 = Input(shape=[1], name="category_sub2")
    category_name = Input(shape=[X_train["category_name"].shape[1]],
                          name="category_name")
    item_condition = Input(shape=[1], name="item_condition")
    num_vars = Input(shape=[X_train["num_vars"].shape[1]], name="num_vars")

    # Embeddings layers
    emb_size = 60

    # Embedding的作用是配置字典size和词向量len后，根据call参数的indices，返回词向量.
    #  类似TF的embedding_lookup
    #  name.shape=[None, MAX_NAME_SEQ], emb_name.shape=[None, MAX_NAME_SEQ, output_dim]
    emb_name = Embedding(input_dim=data_reader.n_text_dict_words, output_dim=emb_size // 3)(name)
    emb_item_desc = Embedding(data_reader.n_text_dict_words, emb_size)(item_desc)  # [None, MAX_ITEM_DESC_SEQ, emb_size]
    emb_category_name = Embedding(data_reader.n_text_dict_words, emb_size // 3)(category_name)
    emb_brand = Embedding(data_reader.n_brand, 10)(brand)
    emb_category_main = Embedding(data_reader.n_cat_main, 10)(category_main)
    emb_category_sub = Embedding(data_reader.n_cat_sub, 10)(category_sub)
    emb_category_sub2 = Embedding(data_reader.n_cat_sub2, 10)(category_sub2)
    emb_item_condition = Embedding(data_reader.n_condition_id, 5)(item_condition)

    # GRU是配置一个cell输出的units长度后，根据call词向量入参,输出最后一个GRU cell的输出(因为默认return_sequences=False)
    rnn_layer1 = GRU(units=16)(emb_item_desc)  # rnn_layer1.shape=[None, 16]
    rnn_layer2 = GRU(8)(emb_category_name)
    rnn_layer3 = GRU(8)(emb_name)

    # main layer
    # 连接列表中的Tensor，按照axis组成一个大的Tensor
    main_l = concatenate([Flatten()(emb_brand),   # [None, 1, 10] -> [None, 10]
                          Flatten()(emb_category_main),
                          Flatten()(emb_category_sub),
                          Flatten()(emb_category_sub2),
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
    model = Model(inputs=[name, item_desc, brand, category_main, category_sub, category_sub2, category_name, item_condition, num_vars], outputs=output)
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
submission = test[["test_id"]]
submission["price"] = preds
submission.to_csv("./myNN"+log_subdir+"_{:.6}.csv".format(v_rmsle), index=False)
print('[{}] Finished submission...'.format(time.time() - start_time))

