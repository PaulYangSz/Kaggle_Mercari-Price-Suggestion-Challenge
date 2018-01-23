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
from keras.layers import Input, Dropout, Dense, concatenate, GRU, Embedding, Flatten, Activation
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K
from nltk.corpus import stopwords
import math


# set seed
np.random.seed(123)

#Load the train and test data
train_df = pd.read_table('../input/train.tsv')
test_df = pd.read_table('../input/test.tsv')
#check the shape of the dataframes
print('train:',train_df.shape, ',test:',test_df.shape)

# removing prices less than 3
train_df = train_df.drop(train_df[(train_df.price < 3.0)].index)
print('After drop price < 3', train_df.shape)


# handling categorical variables
def wordCount(text):
    try:
        if text == 'No description yet':
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


# splitting category_name into subcategories
def split_cat(text):
    try: return text.split("/")
    except: return ("No Label", "No Label", "No Label")
train_df['subcat_0'], train_df['subcat_1'], train_df['subcat_2'] = \
zip(*train_df['category_name'].apply(lambda x: split_cat(x)))
test_df['subcat_0'], test_df['subcat_1'], test_df['subcat_2'] = \
zip(*test_df['category_name'].apply(lambda x: split_cat(x)))



#combine the train and test dataframes
full_set = pd.concat([train_df,test_df])
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
print(found)


# Scale target variable-price to log
train_df["target"] = np.log1p(train_df.price)
# Split training examples into train/dev
train_df, dev_df = train_test_split(train_df, random_state=123, train_size=0.99)
# Calculate number of train/dev/test examples.
n_trains = train_df.shape[0]
n_devs = dev_df.shape[0]
n_tests = test_df.shape[0]
print("Training on:", n_trains, "examples")
print("Validating on:", n_devs, "examples")
print("Testing on:", n_tests, "examples")


# Concatenate train - dev - test data for easy to handle
full_df = pd.concat([train_df, dev_df, test_df])


# Filling missing values in combine dataset
def fill_missing_values(df):
    df.category_name.fillna(value="missing", inplace=True)
    df.brand_name.fillna(value="missing", inplace=True)
    df.item_description.fillna(value="missing", inplace=True)
    df.item_description.replace('No description yet',"missing", inplace=True)
    return df

print("Filling missing data...")
full_df = fill_missing_values(full_df)
print(full_df.category_name[1])



print("Processing categorical data...")
le = LabelEncoder()
le.fit(full_df.category_name)
full_df['category'] = le.transform(full_df.category_name)

le.fit(full_df.brand_name)
full_df.brand_name = le.transform(full_df.brand_name)

le.fit(full_df.subcat_0)
full_df.subcat_0 = le.transform(full_df.subcat_0)

le.fit(full_df.subcat_1)
full_df.subcat_1 = le.transform(full_df.subcat_1)

le.fit(full_df.subcat_2)
full_df.subcat_2 = le.transform(full_df.subcat_2)

del le


print("Transforming text data to sequences...")
raw_text = np.hstack([full_df.item_description.str.lower(), full_df.name.str.lower(), full_df.category_name.str.lower()])

print("Fitting tokenizer...")
tok_raw = Tokenizer()
tok_raw.fit_on_texts(raw_text)

print("Transforming text to sequences...")
full_df['seq_item_description'] = tok_raw.texts_to_sequences(full_df.item_description.str.lower())
full_df['seq_name'] = tok_raw.texts_to_sequences(full_df.name.str.lower())

del tok_raw


# constants to use in RNN model
MAX_NAME_SEQ = 10
MAX_ITEM_DESC_SEQ = 75
MAX_CATEGORY_SEQ = 8
MAX_TEXT = np.max([
    np.max(full_df.seq_name.max()),
    np.max(full_df.seq_item_description.max()),
]) + 100
MAX_CATEGORY = np.max(full_df.category.max()) + 1
MAX_BRAND = np.max(full_df.brand_name.max()) + 1
MAX_CONDITION = np.max(full_df.item_condition_id.max()) + 1
MAX_DESC_LEN = np.max(full_df.desc_len.max()) + 1
MAX_NAME_LEN = np.max(full_df.name_len.max()) + 1
MAX_SUBCAT_0 = np.max(full_df.subcat_0.max()) + 1
MAX_SUBCAT_1 = np.max(full_df.subcat_1.max()) + 1
MAX_SUBCAT_2 = np.max(full_df.subcat_2.max()) + 1


# transform the data for RNN model
def get_rnn_data(dataset):
    X = {
        'name': pad_sequences(dataset.seq_name, maxlen=MAX_NAME_SEQ),
        'item_desc': pad_sequences(dataset.seq_item_description, maxlen=MAX_ITEM_DESC_SEQ),
        'brand_name': np.array(dataset.brand_name),
        'category': np.array(dataset.category),
        'item_condition': np.array(dataset.item_condition_id),
        'num_vars': np.array(dataset[["shipping"]]),
        'desc_len': np.array(dataset[["desc_len"]]),
        'name_len': np.array(dataset[["name_len"]]),
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
def root_mean_squared_logarithmic_error(y_true, y_pred):
    first_log = K.log(K.clip(y_pred, K.epsilon(), None) + 1.)
    second_log = K.log(K.clip(y_true, K.epsilon(), None) + 1.)
    return K.sqrt(K.mean(K.square(first_log - second_log), axis=-1)+0.0000001)
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)+0.0000001)


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
    subcat_0 = Input(shape=[1], name="subcat_0")
    subcat_1 = Input(shape=[1], name="subcat_1")
    subcat_2 = Input(shape=[1], name="subcat_2")

    # Embeddings layers (adjust outputs to help model)
    emb_name = Embedding(MAX_TEXT, 20)(name)
    emb_item_desc = Embedding(MAX_TEXT, 60)(item_desc)
    emb_brand_name = Embedding(MAX_BRAND, 10)(brand_name)
    emb_item_condition = Embedding(MAX_CONDITION, 5)(item_condition)
    emb_desc_len = Embedding(MAX_DESC_LEN, 5)(desc_len)
    emb_name_len = Embedding(MAX_NAME_LEN, 5)(name_len)
    emb_subcat_0 = Embedding(MAX_SUBCAT_0, 10)(subcat_0)
    emb_subcat_1 = Embedding(MAX_SUBCAT_1, 10)(subcat_1)
    emb_subcat_2 = Embedding(MAX_SUBCAT_2, 10)(subcat_2)

    # rnn layers (GRUs are faster than LSTMs and speed is important here)
    rnn_layer1 = GRU(16)(emb_item_desc)
    rnn_layer2 = GRU(8)(emb_name)

    # main layers
    main_l = concatenate([
        Flatten()(emb_brand_name)
        , Flatten()(emb_item_condition)
        , Flatten()(emb_desc_len)
        , Flatten()(emb_name_len)
        , Flatten()(emb_subcat_0)
        , Flatten()(emb_subcat_1)
        , Flatten()(emb_subcat_2)
        , rnn_layer1
        , rnn_layer2
        , num_vars
    ])

    # (incressing the nodes or adding layers does not effect the time quite as much as the rnn layers)
    main_l = Dropout(0.1)(Dense(512, kernel_initializer='normal', activation='relu')(main_l))
    main_l = Dropout(0.1)(Dense(256, kernel_initializer='normal', activation='relu')(main_l))
    main_l = Dropout(0.1)(Dense(128, kernel_initializer='normal', activation='relu')(main_l))
    main_l = Dropout(0.1)(Dense(64, kernel_initializer='normal', activation='relu')(main_l))

    # the output layer.
    output = Dense(1, activation="linear")(main_l)

    model = Model([name, item_desc, brand_name, item_condition,
                   num_vars, desc_len, name_len, subcat_0, subcat_1, subcat_2], output)

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
lr_init, lr_fin = 0.005, 0.001
lr_decay = exp_decay(lr_init, lr_fin, steps)

# Create model and fit it with training dataset.
rnn_model = new_rnn_model(lr=lr_init, decay=lr_decay)
rnn_model.fit(
        X_train, Y_train, epochs=epochs, batch_size=BATCH_SIZE,
        validation_data=(X_dev, Y_dev), verbose=1,
)


#Define RMSL Error Function for checking prediction
def rmsle(Y, Y_pred):
    assert Y.shape == Y_pred.shape
    return np.sqrt(np.mean(np.square(Y_pred - Y )))


print("Evaluating the model on validation data...")
Y_dev_preds_rnn = rnn_model.predict(X_dev, batch_size=BATCH_SIZE)
print(" RMSLE error:", rmsle(Y_dev, Y_dev_preds_rnn))


#prediction for test data
rnn_preds = rnn_model.predict(X_test, batch_size=BATCH_SIZE, verbose=1)
rnn_preds = np.expm1(rnn_preds)


# best predicted submission
preds = rnn_preds
submission = pd.DataFrame({
        "test_id": test_df.test_id,
        "price": preds.reshape(-1),
})
submission.to_csv("./rnn_ridge_submission.csv", index=False)
print("completed time:")
stop_real = datetime.now()
execution_time_real = stop_real-start_real
print(execution_time_real)

