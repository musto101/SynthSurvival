# train and tune a random forest model on synthetic data for MCI using real data for validation

# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from lifelines.utils import concordance_index
import numpy as np
from sksurv.ensemble import RandomSurvivalForest
import optuna

# Read in data
train = pd.read_csv('data/generated_cn_data.csv')

# drop first column
train = train.drop(train.columns[0], axis=1)

# Change last_DX to boolean
train['last_DX'] = train['last_DX'].astype(bool)

#
# # Change last_visit to absolute value
# # train['last_visit'] = train['last_visit'].abs()
#
# # show value counts of last_visit
# print(train['last_visit'].value_counts())
#
# # show summary statistics for last_visit
# print(train['last_visit'].describe())

# read in real data
val = pd.read_csv('data/cn_preprocessed_wo_csf_real.csv')

val['last_DX'] = val['last_DX'].astype(int)

# change last_DX to boolean
val['last_DX'] = val['last_DX'].astype(bool)

# change last_visit to int
val['last_visit'] = val['last_visit'].astype(int)

# find the correlation between val and train and order from highest to lowest
# correlation = val.corrwith(train, axis=0)
# correlation = correlation.abs().sort_values(ascending=False)

# # get summary statistics for last_visit
# print(val['last_visit'].describe())
#
# # get value counts for last_visit
# print(val['last_visit'].value_counts())
#
# # round last_visit to the two decimal places
# train['last_visit'] = train['last_visit'].round(2)
#
# print(train['last_visit'].value_counts(normalize=True) * 100)
#
# # calculate the proportion of each value in last_visit for real data
# real_prop = val['last_visit'].value_counts(normalize=True) * 100
#
# # order the synthetic data by last_visit and create a new column with int values ranging from 6 to 60, inclusive, with the same proportion as the real data
# train = train.sort_values(by='last_visit')
# # train['last_visit_int'] = np.arange(6, 61, )
# # train['last_visit_int'] = train['last_visit_int'].astype(int)
#
# # calculate the proportion of each value in last_visit_int for synthetic data
# synth_prop = train['last_visit'].value_counts(normalize=True) * 100
#
# # create a dictionary with the real data proportions
# prop_dict = real_prop.to_dict()
#
# # create a dictionary with the synthetic data proportions
# synth_dict = synth_prop.to_dict()
#
# # create a dictionary with the mapping of synthetic data values to real data values
# mapping_dict = dict(zip(synth_dict.keys(), prop_dict.keys()))
#
# # map the synthetic data values to real data values
# train['last_visit'] = train['last_visit'].map(mapping_dict)
#

# reorder train columns to match val columns order
train = train[val.columns]

# do the same for all ordinal columns
# for col in val.columns:
#     if val[col].nunique() < 10:
#         real_prop = val[col].value_counts(normalize=True) * 100
#         synth_prop = train[col].value_counts(normalize=True) * 100
#         prop_dict = real_prop.to_dict()
#         synth_dict = synth_prop.to_dict()
#         mapping_dict = dict(zip(synth_dict.keys(), prop_dict.keys()))
#         train[col] = train[col].map(mapping_dict)

# convert the value fou

#

# split the val data into test and val sets
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
# print(X_train.isnull().sum())
# print(X_val.isnull().sum())
# print(X_test.isnull().sum())


# parameters for testing

# n_estimators = 1000
# max_depth = 5
# min_samples_split = 2
# min_samples_leaf = 1
# max_features = 'sqrt'

# define objective function

def objective(trial):

    # define parameters
    n_estimators = trial.suggest_int('n_estimators', 100, 5000)
    max_depth = trial.suggest_int('max_depth', 1, 9)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])

    # check for missing values in parameters and return None if any is missing
    if None in (n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features):
        return None

    # create survival random forest model
    rsf = RandomSurvivalForest(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split,
                               min_samples_leaf=min_samples_leaf, max_features=max_features)

    # fit model
    rsf.fit(X_train, y_train)

    # predict on test set
    y_pred = rsf.predict(X_val)

    # # count number of unique values in y_pred
    # unique_values = np.unique(y_pred).shape[0]

    # round predictions to nearest integer
    y_pred = np.round(y_pred)

    # calculate the negative log likelihood for the validation set
    c_index = 1 - concordance_index(y_val['last_visit'], y_pred)


    return c_index

# define study
study = optuna.create_study(direction='maximize')

# optimize study
study.optimize(objective, n_trials=10)

# get best parameters
best_params = study.best_params

# print best parameters
print(best_params)

# get best value
print(study.best_value)

# create survival random forest model with best parameters
rsf = RandomSurvivalForest(n_estimators=best_params['n_estimators'], max_depth=best_params['max_depth'],
                           min_samples_split=best_params['min_samples_split'], min_samples_leaf=best_params['min_samples_leaf'],
                           max_features=best_params['max_features'])

# fit model
rsf.fit(X_train, y_train)



# # predict on test set
y_pred = rsf.predict(X_test)

# round predictions to nearest integer
y_pred = np.round(y_pred)

# calculate concordance index for test set
c_index = concordance_index(y_test['last_visit'], y_pred)

# print concordance index
print(c_index)




