# create a cox proportional hazards model

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import train_test_split

# Read in data
train = pd.read_csv('data/generated_cn_data.csv')

# drop first column
# train = train.drop(train.columns[0], axis=1)

# Change last_DX to boolean
train['last_DX'] = train['last_DX'].astype(bool)

# # Change last_visit to last_visit2
train['last_visit'] = train['last_visit2']

# drop last_visit2
train = train.drop('last_visit2', axis=1)
# check for missing values
print(train.isnull().sum())

val = pd.read_csv('data/mci_preprocessed_wo_csf_real.csv')

val['last_DX'] = val['last_DX'].astype(bool)

# reorder train columns to match val columns order
train = train[val.columns]

# check for missing values
print(val.isnull().sum())

# do the same for all ordinal columns
for col in val.columns:
    if val[col].nunique() < 10:
        real_prop = val[col].value_counts(normalize=True) * 100
        synth_prop = train[col].value_counts(normalize=True) * 100
        prop_dict = real_prop.to_dict()
        synth_dict = synth_prop.to_dict()
        mapping_dict = dict(zip(synth_dict.keys(), prop_dict.keys()))
        train[col] = train[col].map(mapping_dict)

# check for missing values
print(val.isnull().sum())
print(train.isnull().sum())


# split the validation data into val and test
val, test = train_test_split(val, test_size=0.2, random_state=0)

# split the train data into X and y
y_train = train[['last_DX', 'last_visit']]
y_train = y_train.to_records(index=False)

X_train = train.drop(['last_DX', 'last_visit'], axis=1)

# split the val data into X and y
y_val = val[['last_DX', 'last_visit']]
y_val = y_val.to_records(index=False)

X_val = val.drop(['last_DX', 'last_visit'], axis=1)

# split the test data into X and y
y_test = test[['last_DX', 'last_visit']]
y_test = y_test.to_records(index=False)

X_test = test.drop(['last_DX', 'last_visit'], axis=1)

# check for missing values
print(X_train.isnull().sum())
print(X_val.isnull().sum())
print(X_test.isnull().sum())

# drop columns with missing values
train = train.dropna(axis=1)
val = val.dropna(axis=1)
test = test.dropna(axis=1)



# create a cox proportional hazards model
cph = CoxPHFitter(penalizer=0.1)
cph.fit(train, duration_col='last_visit', event_col='last_DX')

# predict on validation data
val_preds = cph.predict_expectation(val)

# calculate concordance index
c_index = concordance_index(val['last_visit'], val_preds)

