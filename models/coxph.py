import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import numpy as np

indices = []

for i in range(10):
    training = pd.read_csv('data/generated__mci_data.csv')

    # drop first column
    training = training.drop(training.columns[0], axis=1)

    # read in real data
    val = pd.read_csv('data/mci_preprocessed_wo_csf_real.csv')

    # bootstrap the data so that it has 1000 rows
    val = val.sample(n=1000, replace=True)

    val['last_DX'] = val['last_DX'].astype(int)

    # change last_visit to int
    val['last_visit'] = val['last_visit'].astype(int)

    # reorder train columns to match val columns order
    training = training[val.columns]

    # scale age
    training['AGE'] = (training['AGE'] - training['AGE'].mean()) / training['AGE'].std()

    # create a cox proportional hazards model
    cph = CoxPHFitter(penalizer=0.001)

    cph.fit(training, duration_col='last_visit', event_col='last_DX')

    c_index = concordance_index(val['last_visit'], cph.predict_partial_hazard(val), val['last_DX'])
    indices.append(c_index)


indices = np.array(indices)
avg = np.mean(indices)
std = np.std(indices)


