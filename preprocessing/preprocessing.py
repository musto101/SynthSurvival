import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.impute import KNNImputer

mci = pd.read_csv('data/mci_preprocessed_wo_csf.csv') # read in data
codes = {'CN_MCI': 0, 'Dementia': 1} # change last_DX to boolean
mci['last_DX'].replace(codes, inplace=True)
mci = mci.drop(['Unnamed: 0'], axis=1) # drop first column

mci.dtypes

# find number of nan values for each column and order by most to least
mci.isna().sum().sort_values(ascending=False)

# use knn imputer to fill in nan values

imputer = KNNImputer(n_neighbors=5)
mci.iloc[:,1:] = imputer.fit_transform(mci.iloc[:,1:])

# find count of unique values for each last_DX
mci['last_DX'].value_counts()

# use SMOTE to oversample minority class

X = mci.iloc[:,1:]
y = mci.iloc[:,0]
print('Original dataset shape %s' % Counter(y))
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)
print('Resampled dataset shape %s' % Counter(y_res))

# create dataframe from X_res and y_res with last_DX as first column
mci = pd.DataFrame(X_res)

mci.dtypes
mci.insert(0, 'last_DX', y_res)

mci.to_csv('data/mci_preprocessed_wo_csf3.csv', index=False)

# get names of ordinal columns with less than 3 unique values
ordinal_columns = mci.columns[mci.nunique() < 3]

# drop ordinal columns
mci_numeric = mci.drop(ordinal_columns, axis=1)
numeric_columns = mci_numeric.columns

mci_ordinal = mci[ordinal_columns]
# convert to int32
mci_ordinal = mci_ordinal.astype('int32')

# convert to float32
mci_numeric = mci_numeric.astype('float32')

# scale data
scaler = StandardScaler()
x = scaler.fit_transform(mci_numeric)

# convert x to dataframe, add the column names, and concatenate with mci_ordinal
x = pd.DataFrame(x)
x.columns = numeric_columns
x = pd.concat([mci_ordinal, x], axis=1)

x.dtypes

# restore column names and save to csv

x.to_csv('data/mci_preprocessed_wo_csf_vae2.csv', index=False)

# save column names to csv
ordinal_columns = pd.DataFrame(ordinal_columns)
ordinal_columns.to_csv('data/ordinal_columns.csv', index=False)

numeric_columns = pd.DataFrame(numeric_columns)
numeric_columns.to_csv('data/numeric_columns.csv', index=False)
# preprocess data for real data
mci = pd.read_csv('data/mci_preprocessed_wo_csf.csv')
codes = {'CN_MCI': 0, 'Dementia': 1}
mci['last_DX'].replace(codes, inplace=True)
mci = mci.drop(['Unnamed: 0'], axis=1)

mci.dtypes

# find number of nan values for each column and order by most to least
mci.isna().sum().sort_values(ascending=False)

# use knn imputer to fill in nan values
imputer = KNNImputer(n_neighbors=5)
mci.iloc[:,2:] = imputer.fit_transform(mci.iloc[:,2:])

# find count of unique values for each last_DX
mci['last_DX'].value_counts()

# use SMOTE to oversample minority class
X = mci.iloc[:,1:]
y = mci.iloc[:,0]
print('Original dataset shape %s' % Counter(y))
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)
print('Resampled dataset shape %s' % Counter(y_res))

# create dataframe from X_res and y_res with last_DX as first column
mci = pd.DataFrame(X_res)
mci.insert(0, 'last_DX', y_res)

# convert to float32
mci.iloc[:,2:] = mci.iloc[:,2:].astype('float32')

scaler = StandardScaler()
x = scaler.fit_transform(mci.iloc[:,2:])
combined = np.concatenate((mci.iloc[:,:2], x), axis=1)

# restore column names and save to csv
combined = pd.DataFrame(combined)
combined.columns = mci.columns
combined.to_csv('data/mci_preprocessed_wo_csf_real.csv', index=False)