# run survival xgboost model on synth data
import pandas as pd
import xgboost as xgb
from lifelines.utils import concordance_index
import optuna
import numpy as np
from sklearn.model_selection import train_test_split

indices = []

for i in range(100):
    # Read in data
    train = pd.read_csv('data/generated_cn_data.csv')

    # drop first column
    train = train.drop(train.columns[0], axis=1)

    # Change last_DX to boolean
    # train['last_DX'] = train['last_DX'].astype(bool)

    # read in real data
    val = pd.read_csv('data/cn_preprocessed_wo_csf_real.csv')

    val['last_DX'] = val['last_DX'].astype(int)

    # change last_DX to boolean
    # val['last_DX'] = val['last_DX'].astype(bool)

    # change last_visit to int
    val['last_visit'] = val['last_visit'].astype(int)

    # reorder train columns to match val columns order
    train = train[val.columns]

    # Handle NaNs by dropping them or imputing them
    train = train.dropna()
    val = val.dropna()

    # scale age
    train['AGE'] = (train['AGE'] - train['AGE'].mean()) / train['AGE'].std()

    # split the val data into test and val sets
    val, test = train_test_split(val, test_size=0.2, random_state=0)

    # bootstrap the test data so that it has 1000 rows
    test = test.sample(n=1000, replace=True)

    # split the train data into X and y
    y_train = train[['last_DX', 'last_visit']]
    # y_train = y_train.to_records(index=False)

    X_train = train.drop(['last_DX', 'last_visit'], axis=1)

    # split the val data into X and y
    y_val = val[['last_DX', 'last_visit']]
    # y_val = y_val.to_records(index=False)

    X_val = val.drop(['last_DX', 'last_visit'], axis=1)

    # split the test data into X and y
    y_test = test[['last_DX', 'last_visit']]
    # y_test = y_test.to_records(index=False)

    # Check for NaNs in target variable
    print(y_train.isna().sum())
    print(y_val.isna().sum())
    print(y_test.isna().sum())

    X_test = test.drop(['last_DX', 'last_visit'], axis=1)

    dtrain = xgb.DMatrix(X_train, label=y_train['last_DX'], weight=y_train['last_visit'])
    dval = xgb.DMatrix(X_val, label=y_val['last_DX'], weight=y_val['last_visit'])
    dtest = xgb.DMatrix(X_test,  label=y_test['last_DX'], weight=y_test['last_visit'])


    # parameters for testing
    # params = {
    #     'objective': 'survival:cox',
    #     'eval_metric': 'cox-nloglik',
    #     'verbosity': 0,
    #     'booster': 'gbtree',
    #     'nthread': 4,
    #     'max_depth': 6,
    #     'eta': 0.01,
    #     'subsample': 0.8,
    #     'colsample_bytree': 0.8,
    #     'lambda': 0.01,
    #     'alpha': 0.01,
    #     'min_child_weight': 0.001,
    #     'gamma': 0.001,
    #     'n_estimators': 1000
    # }


    def objective(trial):

        # define parameters for xgboost
        params = {
            'objective': 'survival:cox',
            'eval_metric': 'cox-nloglik',
            'verbosity': 0,
            'booster': 'gbtree',
            'nthread': 4,
            'max_depth': trial.suggest_int('max_depth', 2, 10),
            'eta': trial.suggest_loguniform('eta', 1e-3, 0.1),
            'subsample': trial.suggest_uniform('subsample', 0.5, 1),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1),
            'lambda': trial.suggest_loguniform('lambda', 1e-2, 1),
            'alpha': trial.suggest_loguniform('alpha', 1e-2, 1),
            'min_child_weight': trial.suggest_loguniform('min_child_weight', 1e-3, 10),
            'gamma': trial.suggest_loguniform('gamma', 1e-2, 1),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000)
        }

        print(params)

        # check for missing values in parameters and return None if any is missing
        if None in (params['max_depth'], params['eta'], params['subsample'],
                    params['colsample_bytree'], params['lambda'], params['alpha'],
                    params['min_child_weight'], params['gamma']):
            return None

        # create survival xgboost model
        model = xgb.train(params, dtrain)

        # predict on validation set
        y_pred = model.predict(dval)

        # check for NaNs in predictions
        print(np.isnan(y_pred).sum())

        # if there are nan values in y_pred, continue to the next iteration
        # if np.isnan(y_pred).sum() > 0:
        #     model = xgb.train(params, dtrain)
        #     y_pred = model.predict(dval)
        #
        # # check for nan values in y_pred
        # print(np.isnan(y_pred).sum())

        if np.isnan(y_pred).any():
            print(f"Trial failed due to NaNs in predictions. Parameters: {params}")
            return None

        # check for nan values in y_val
        print(np.isnan(y_val['last_visit']).sum())
        # calculate concordance index
        c_index = concordance_index(y_val['last_visit'], y_pred, y_val['last_DX'])

        return c_index

    # create study
    study = optuna.create_study(direction='maximize')

    # optimize study
    study.optimize(objective, n_trials=1000)

    # if there is an error, continue to the next iteration
    if study.best_params is None:
        continue

    # get best parameters
    best_params = study.best_params

    # print best parameters
    print(best_params)

    # get best value
    print('best val c_index', study.best_value)

    # create survival xgboost model with best parameters
    model = xgb.train(best_params, dtrain)

    # predict on test set
    y_pred = model.predict(dtest)

    # calculate concordance index
    c_index = concordance_index(y_test['last_visit'], -y_pred, y_test['last_DX'])

    indices.append(c_index)

# get the average of the indices and the standard deviation
indices = np.array(indices)
avg = np.mean(indices)
std = np.std(indices)

print('average c_index', avg)
print('std', std)







