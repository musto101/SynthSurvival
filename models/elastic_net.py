import pandas as pd
from sklearn.model_selection import train_test_split
from lifelines import CoxPHFitter
import optuna

# Read in data
train = pd.read_csv('data/generated__mci_data.csv')

# drop first column
train = train.drop(train.columns[0], axis=1)

# Change last_DX to boolean
train['last_DX'] = train['last_DX'].astype(int)

# read in real data
val = pd.read_csv('data/mci_preprocessed_wo_csf_real.csv')

val['last_DX'] = val['last_DX'].astype(int)

# reorder train columns to match val columns order
train = train[val.columns]

# split the val data into test and val sets
val, test = train_test_split(val, test_size=0.2, random_state=0)

# split the train data into X and y
y_train = train[['last_visit']]

X_train = train.drop(['last_visit'], axis=1)

# split the val data into X and y
y_val = val[['last_visit']]

X_val = val.drop(['last_visit'], axis=1)

# split the test data into X and y
y_test = test[['last_visit']]

X_test = test.drop(['last_visit'], axis=1)

# define objective function

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



