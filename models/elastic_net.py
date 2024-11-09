# train and tune an xgboost model on synthetic data for MCI using real data for validation
# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from lifelines.utils import concordance_index
import numpy as np
# import elastic_net from sklearn
from sklearn.linear_model import ElasticNet
from lifelines import CoxPHFitter
import optuna
# import mse from sklearn
from sklearn.metrics import mean_squared_error

# Read in data
train = pd.read_csv('data/generated__mci_data.csv')

# drop first column
train = train.drop(train.columns[0], axis=1)

# Change last_DX to boolean
train['last_DX'] = train['last_DX'].astype(int)

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
val = pd.read_csv('data/mci_preprocessed_wo_csf_real.csv')

val['last_DX'] = val['last_DX'].astype(int)

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
y_train = train[['last_visit']]
# y_train = y_train.to_records(index=False)

X_train = train.drop(['last_visit'], axis=1)

# split the val data into X and y
y_val = val[['last_visit']]
# y_val = y_val.to_records(index=False)

X_val = val.drop(['last_visit'], axis=1)

# split the test data into X and y
y_test = test[['last_visit']]
# y_test = y_test.to_records(index=False)

X_test = test.drop(['last_visit'], axis=1)

# parameters for testing
# n_estimators = 100
# max_depth = 3
# learning_rate = 0.1
# subsample = 1.0
# colsample_bytree = 1.0
# gamma = 0
# reg_alpha = 1e-8
# reg_lambda = 1e-8
# min_child_weight = 1


# define objective function

#parameters for testing
penaliser = 1e-8
l1_ratio = 0.5

def objective(trial):

    # define parameters for cox model
    penaliser = trial.suggest_loguniform('penalizer', 1e-8, 1e-1)
    l1_ratio = trial.suggest_uniform('l1_ratio', 0, 1)


    # create the cox model
    coxph = CoxPHFitter(penalizer=penaliser, l1_ratio=l1_ratio)

    # fit model
    coxph.fit(train, duration_col='last_visit', event_col='last_DX', show_progress=True)

    # get concordance index for validation set
    cindex = 1 - coxph.score(val, scoring_method='concordance_index')




    # # convert to int
    # y_pred = y_pred.astype(int)

    # # count number of unique values in y_pred
    # unique_values = np.unique(y_pred).shape[0]

    # round predictions to nearest integer
    # y_pred = np.round(y_pred)

    # calculate roc_auc score for test set
    # rocauc = roc_auc_score(y_val, y_pred)


    return cindex

# define study
study = optuna.create_study(direction='maximize')

# optimize study
study.optimize(objective, n_trials=1000, show_progress_bar=True,  timeout=3600)


# get best parameters
best_params = study.best_params

# print best parameters
print(best_params)

# get best value
print(study.best_value)

# create random forest model with best parameters
coxph = CoxPHFitter(penalizer=best_params['penalizer'], l1_ratio=best_params['l1_ratio'])
# fit model
coxph.fit(train, duration_col='last_visit', event_col='last_DX', show_progress=True)

# get concordance index for test set
cindex = 1 - coxph.score(test, scoring_method='concordance_index')

# print concordance index
print(cindex)



