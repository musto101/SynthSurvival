# train and tune a random forest model on synthetic data for MCI using real data for validation

# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from lifelines.utils import concordance_index
import numpy as np
from sksurv.ensemble import RandomSurvivalForest
import optuna

# Read in data
train = pd.read_csv('data/generated__mci_data.csv')

# drop first column
train = train.drop(train.columns[0], axis=1)

# Change last_DX to boolean
train['last_DX'] = train['last_DX'].astype(bool)

#
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


# parameters for testing

# n_estimators = 1000
# max_depth = 5
# min_samples_split = 2
# min_samples_leaf = 1
# max_features = 'sqrt'

# define objective function

def objective(trial):

    # define parameters
    n_estimators = trial.suggest_int('n_estimators', 1000, 5000)
    max_depth = trial.suggest_int('max_depth', 1, 9)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])

    # create survival random forest model
    rsf = RandomSurvivalForest(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split,
                               min_samples_leaf=min_samples_leaf, max_features=max_features, n_jobs=-1)

    # fit model
    rsf.fit(X_train, y_train)

    # predict on test set
    y_pred = rsf.predict(X_val)

    # # count number of unique values in y_pred
    # unique_values = np.unique(y_pred).shape[0]

    # round predictions to nearest integer
    y_pred = np.round(y_pred)

    # calculate concordance index for test set
    c_index = concordance_index(y_val['last_visit'], -y_pred)


    return -c_index

# define study
study = optuna.create_study(direction='minimize')

# optimize study
study.optimize(objective, n_trials=1000)

# get best parameters
best_params = study.best_params

# print best parameters
print(best_params)

# get best value
print(study.best_value)

# create survival random forest model with best parameters
rsf = RandomSurvivalForest(n_estimators=best_params['n_estimators'], max_depth=best_params['max_depth'],
                           min_samples_split=best_params['min_samples_split'], min_samples_leaf=best_params['min_samples_leaf'],
                           max_features=best_params['max_features'], n_jobs=-1)

# fit model
rsf.fit(X_train, y_train)

# predict on test set
y_pred = rsf.predict(X_test)

# round predictions to nearest integer
y_pred = np.round(y_pred)

# calculate concordance index for test set
c_index = concordance_index(y_test['last_visit'], -y_pred)

# print concordance index
print(c_index)




