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

from nltk.corpus import stopwords
import math


# set seed
np.random.seed(123)

#Load the train and test data
train_df = pd.read_table('../../input/train.tsv', engine='python')
test_df = pd.read_table('../../input/test.tsv', engine='python')
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


# Define RMSL Error Function for checking prediction
def rmsle(Y, Y_pred):
    assert Y.shape == Y_pred.shape
    return np.sqrt(np.mean(np.square(Y_pred - Y )))


# Ridge modelling
# Concatenate train - dev - test data for furthur handling
full_df = pd.concat([train_df, dev_df, test_df])


print("Handling missing values...")
full_df['category_name'] = full_df['category_name'].fillna('missing').astype(str)
full_df['subcat_0'] = full_df['subcat_0'].astype(str)
full_df['subcat_1'] = full_df['subcat_1'].astype(str)
full_df['subcat_2'] = full_df['subcat_2'].astype(str)
full_df['brand_name'] = full_df['brand_name'].fillna('missing').astype(str)
full_df['shipping'] = full_df['shipping'].astype(str)
full_df['item_condition_id'] = full_df['item_condition_id'].astype(str)
full_df['desc_len'] = full_df['desc_len'].astype(str)
full_df['name_len'] = full_df['name_len'].astype(str)
full_df['item_description'] = full_df['item_description'].fillna('No description yet').astype(str)


print("Vectorizing data for Ridge Modeling...")
default_preprocessor = CountVectorizer().build_preprocessor()
def build_preprocessor(field):
    field_idx = list(full_df.columns).index(field)
    return lambda x: default_preprocessor(x[field_idx])

vectorizer = FeatureUnion([
    ('name', CountVectorizer(
        ngram_range=(1, 2),
        max_features=50000,
        preprocessor=build_preprocessor('name'))),
    ('subcat_0', CountVectorizer(
        token_pattern='.+',
        preprocessor=build_preprocessor('subcat_0'))),
    ('subcat_1', CountVectorizer(
        token_pattern='.+',
        preprocessor=build_preprocessor('subcat_1'))),
    ('subcat_2', CountVectorizer(
        token_pattern='.+',
        preprocessor=build_preprocessor('subcat_2'))),
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

X = vectorizer.fit_transform(full_df.values)

X_train = X[:n_trains]
Y_train = train_df.target.values.reshape(-1, 1)

X_dev = X[n_trains:n_trains+n_devs]
Y_dev = dev_df.target.values.reshape(-1, 1)

X_test = X[n_trains+n_devs:]
print(X.shape, X_train.shape, X_dev.shape, X_test.shape)


print("Fitting Ridge model on training examples...")
ridge_model = Ridge(
    solver='auto', fit_intercept=True, alpha=1.0,
    max_iter=100, normalize=False, tol=0.05, random_state = 1,
)
ridge_modelCV = RidgeCV(
    fit_intercept=True, alphas=[5.0],
    normalize=False, cv = 2, scoring='neg_mean_squared_error',
)
ridge_model.fit(X_train, Y_train)
ridge_modelCV.fit(X_train, Y_train)


#Evaluating Ridge model on dev data
Y_dev_preds_ridge = ridge_model.predict(X_dev)
Y_dev_preds_ridge = Y_dev_preds_ridge.reshape(-1, 1)
print("RMSL error on dev set:", rmsle(Y_dev, Y_dev_preds_ridge))



Y_dev_preds_ridgeCV = ridge_modelCV.predict(X_dev)
Y_dev_preds_ridgeCV = Y_dev_preds_ridgeCV.reshape(-1, 1)
print("CV RMSL error on dev set:", rmsle(Y_dev, Y_dev_preds_ridgeCV))


#prediction for test data
ridge_preds = ridge_model.predict(X_test)
ridge_preds = np.expm1(ridge_preds)
ridgeCV_preds = ridge_modelCV.predict(X_test)
ridgeCV_preds = np.expm1(ridgeCV_preds)


#combine all predictions
def aggregate_predicts3(Y1, Y2, Y3, ratio1, ratio2):
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
            Y_dev_preds = aggregate_predicts3(Y_dev_preds_rnn, Y_dev_preds_ridgeCV, Y_dev_preds_ridge, r, r2)
            fpred = rmsle(Y_dev, Y_dev_preds)
            if fpred < lowest:
                best1 = r
                best2 = r2
                lowest = fpred
Y_dev_preds = aggregate_predicts3(Y_dev_preds_rnn, Y_dev_preds_ridgeCV, Y_dev_preds_ridge, best1, best2)


print("(Best) RMSL error for RNN + Ridge + RidgeCV on dev set:", rmsle(Y_dev, Y_dev_preds))

# best predicted submission
preds = aggregate_predicts3(rnn_preds, ridgeCV_preds, ridge_preds, best1, best2)
submission = pd.DataFrame({
        "test_id": test_df.test_id,
        "price": preds.reshape(-1),
})
submission.to_csv("./rnn_ridge_submission.csv", index=False)
print("completed time:")
stop_real = datetime.now()
execution_time_real = stop_real-start_real
print(execution_time_real)

