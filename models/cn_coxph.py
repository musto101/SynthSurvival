import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import numpy as np

indices = []

for i in range(100):
    training = pd.read_csv('data/generated_cn_data.csv')

    # drop first column
    training = training.drop(training.columns[0], axis=1)

    # remove columns with low variance
    training = training.loc[:, training.var() > 0.001]

    # Change last_DX to boolean
    # train['last_DX'] = train['last_DX'].astype(bool)

    # read in real data
    val = pd.read_csv('data/cn_preprocessed_wo_csf_real.csv')

    # only keep columns that are in training
    val = val[training.columns]

    # bootstrap the data so that it has 1000 rows
    val = val.sample(n=1000, replace=True)

    val['last_DX'] = val['last_DX'].astype(int)

    # change last_DX to boolean
    # val['last_DX'] = val['last_DX'].astype(bool)

    # change last_visit to int
    val['last_visit'] = val['last_visit'].astype(int)

    # reorder train columns to match val columns order
    training = training[val.columns]

    # scale age
    training['AGE'] = (training['AGE'] - training['AGE'].mean()) / training['AGE'].std()

    # count na values by column
    print(training.isna().sum().sort_values(ascending=False))

    # drop na values
    training = training.dropna()
    val = val.dropna()
    print(val.isna().sum().sort_values(ascending=False))


    # create a cox proportional hazards model
    cph = CoxPHFitter(penalizer=0.001)

    # cph.fit_options = dict(step_size=0.0005)

    cph.fit(training, duration_col='last_visit', event_col='last_DX')

    c_index = concordance_index(val['last_visit'], cph.predict_partial_hazard(val), val['last_DX'])
    indices.append(c_index)


# get the average of the indices and the standard deviation
indices = np.array(indices)
avg = np.mean(indices)
std = np.std(indices)
