# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd

# Path of the file to read
file_path = 'C:/Users/bluep/Dropbox/peng/Academia/Work with Brea/SNAD/SNAD data/Peng/SNAD-Analysis-T1234-20201001-Imaging from Shannon.dta'

# Fill in the line below to read the file into a variable home_data
home_data = pd.read_stata(file_path)

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
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

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


# random forest 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
rf_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, rf_preds))
