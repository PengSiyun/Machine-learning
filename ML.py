# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd

# Path of the file to read
file_path = 'C:/Users/bluep/Dropbox/peng/Academia/Work with Brea/SNAD/SNAD data/Peng/SNAD-Analysis-T1234-20201001-Imaging from Shannon.dta'

# Fill in the line below to read the file into a variable home_data
home_data = pd.read_stata(file_path, index_col='SUBID')

# Print summary statistics in next line
home_data.describe()
home_data.columns

# Create the list of features below
feature_names = ['grade', 'female', 'black', 'age', 'MOCATOTS']

# Select data corresponding to features in feature_names
home_data = home_data[feature_names]
# dropna drops missing values (think of na as "not available")
home_data = home_data.dropna(axis=0)
feature_names = ['grade', 'female', 'black', 'age']
X = home_data[feature_names]

y = home_data.MOCATOTS
# Review data
# print description or statistics from X
X.describe()

# print the top few lines
X.head()

# load regression function
from sklearn.tree import DecisionTreeRegressor

# Define model. Specify a number for random_state to ensure same results each run
moca_model = DecisionTreeRegressor(random_state=1)

# Fit model
moca_model.fit(X, y)

# Predict top 6 cases
print(moca_model.predict(X.head(6)))
# real values in the data
home_data.head(6)

# Model fit
# in sample MAE
from sklearn.metrics import mean_absolute_error

predicted_moca = moca_model.predict(X)
mean_absolute_error(y, predicted_moca)

# split data into training and validation data, for both features and target
from sklearn.model_selection import train_test_split
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state = 0)

# Fit model
moca_model.fit(train_X, train_y)

# get predicted prices on validation data
val_predictions = moca_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))

# Compare MAE of different trees
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

# compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))

#########################################
# random forest 
#########################################

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
rf_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, rf_preds))

# compare & select best models
model_1 = RandomForestRegressor(n_estimators=50, random_state=0)
model_2 = RandomForestRegressor(n_estimators=100, random_state=0)
model_3 = RandomForestRegressor(n_estimators=100, criterion='mae', random_state=0)
model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)
model_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)

models = [model_1, model_2, model_3, model_4, model_5]
# Function for comparing different models
def score_model(model, X_t=train_X, X_v=val_X, y_t=train_y, y_v=val_y):
    model.fit(X_t, y_t)
    preds = model.predict(X_v)
    return mean_absolute_error(y_v, preds)

for i in range(0, len(models)):
    mae = score_model(models[i])
    print("Model %d MAE: %d" % (i+1, mae))
    
#########################################
# Missing data
#########################################

# Load the data
import pandas as pd
file_path = 'C:/Users/bluep/Dropbox/peng/Academia/Work with Brea/SNAD/SNAD data/Peng/SNAD-Analysis-T1234-20201001-Imaging from Shannon.dta'
home_data = pd.read_stata(file_path, index_col='SUBID')
# Remove rows with missing target
home_data.dropna(axis=0, subset=['MOCATOTS'], inplace=True)
# Select target
y = home_data.MOCATOTS
# Select features
feature_names = ['grade', 'female', 'black', 'age']
X = home_data[feature_names]
# Divide data into training and validation subsets
from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state = 0)

# Function for comparing different approaches
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def score_dataset(train_X, val_X, train_y, val_y):
    model = RandomForestRegressor(n_estimators=10, random_state=0)
    model.fit(train_X, train_y)
    preds = model.predict(val_X)
    return mean_absolute_error(val_y, preds)

# Drop columns in training and validation data (Approach 1)
# Number of missing values in each column of training data
missing_val_count_by_column = (train_X.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])
# Get names of columns with missing values
cols_with_missing = [col for col in train_X.columns
                     if train_X[col].isnull().any()]

reduced_train_X = train_X.drop(cols_with_missing, axis=1)
reduced_val_X = val_X.drop(cols_with_missing, axis=1)

print("MAE from Approach 1 (Drop columns with missing values):")
print(score_dataset(reduced_train_X, reduced_val_X, train_y, val_y))

from sklearn.impute import SimpleImputer

# Imputation (Approach 2)
my_imputer = SimpleImputer() #replace missing values with the mean value
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(train_X))
imputed_X_valid = pd.DataFrame(my_imputer.transform(val_X))

# Imputation removed column names; put them back
imputed_X_train.columns = train_X.columns
imputed_X_valid.columns = val_X.columns

print("MAE from Approach 2 (Imputation):")
print(score_dataset(imputed_X_train, imputed_X_valid, train_y, val_y))

# An Extension to Imputation (Approach 3): keeping track of which values were imputed
# Make copy to avoid changing original data (when imputing)
train_X_plus = train_X.copy()
val_X_plus = val_X.copy()

# Make new columns indicating what will be imputed
for col in cols_with_missing:
    train_X_plus[col + '_was_missing'] = train_X_plus[col].isnull()
    val_X_plus[col + '_was_missing'] = val_X_plus[col].isnull()

# Imputation
my_imputer = SimpleImputer()
imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(train_X_plus))
imputed_X_valid_plus = pd.DataFrame(my_imputer.transform(val_X_plus))

# Imputation removed column names; put them back
imputed_X_train_plus.columns = train_X_plus.columns
imputed_X_valid_plus.columns = val_X_plus.columns

print("MAE from Approach 3 (An Extension to Imputation):")
print(score_dataset(imputed_X_train_plus, imputed_X_valid_plus, train_y, val_y))

#####################################
#Categorical Variables
#####################################
# Get list of categorical variables
s = (train_X.dtypes == 'object')
object_cols = list(s[s].index)
print("Categorical variables:")
print(object_cols)

#####Drop Categorical Variables
drop_X_train = X_train.select_dtypes(exclude=['object'])
drop_X_valid = X_valid.select_dtypes(exclude=['object'])

#####Label Encoding (convert categorical to ordinal)
from sklearn.preprocessing import LabelEncoder

# Make copy to avoid changing original data 
label_X_train = X_train.copy()
label_X_valid = X_valid.copy()

# Apply label encoder to each column with categorical data
label_encoder = LabelEncoder()
for col in object_cols:
    label_X_train[col] = label_encoder.fit_transform(X_train[col])
    label_X_valid[col] = label_encoder.transform(X_valid[col])

#####One-Hot Encoding (convert categorical to indicators)
from sklearn.preprocessing import OneHotEncoder

# Apply one-hot encoder to each column with categorical data
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[object_cols]))

# One-hot encoding removed index; put it back
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

# Remove categorical columns (will replace with one-hot encoding)
num_X_train = X_train.drop(object_cols, axis=1)
num_X_valid = X_valid.drop(object_cols, axis=1)

# Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)