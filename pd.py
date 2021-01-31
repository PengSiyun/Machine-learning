# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 16:25:35 2021

@author: Peng
"""

#########################################
#Creating data
#########################################
import pandas as pd
#The dictionary-list constructor assigns values to the column labels, 
#but just uses an ascending count from 0 (0, 1, 2, 3, ...) for the row labels. 
#Sometimes this is OK, but oftentimes we will want to assign these labels ourselves.
#The list of row labels used in a DataFrame is known as an Index. We can assign 
#values to it by using an index parameter in our constructor
pd.DataFrame({'Bob': ['I liked it.', 'It was awful.'], 
              'Sue': ['Pretty good.', 'Bland.']},
             index=['Product A', 'Product B'])

#########################################
#Reading data
#########################################
wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv")
wine_reviews.shape #check how large the resulting DataFrame is

pip install /Users/bluep/Desktop/opcov

#pip install git+https://github.com/yy/optum-covid/tree/impact_depression/libs/opcov

#########################################
#Selecting data
#########################################
####select a column/property of an object
data['country']
data.country
data['country'][0] #first row of country
#To select the first row of data in a DataFrame
data.iloc[0] 
#Both loc and iloc are row-first, column-second.
# This is the opposite of what we do in native Python, which is column-first, row-second.
#iloc: index-based selection
data.iloc[:, 0] #: refers to everything
data.iloc[:3, 0] #1, 2, 3 rows
data.iloc[1:3, 0] # 2, 3 rows
data.iloc[[0, 1, 2], 0] # 1, 2, 3 rows
data.iloc[-5:] # last five elements of the dataset.

#loc: label-based selection
data.loc[0, 'country']
data.loc[:, ['taster_name', 'taster_twitter_handle', 'points']]
"""iloc uses the Python stdlib indexing scheme, where the first element of the 
range is included and the last one excluded. So 0:10 will select entries 0,...,9. 
loc, meanwhile, indexes inclusively. So 0:10 will select entries 0,...,10.
"""
####Conditional selection
data.country == 'Italy'
data.loc[data.country == 'Italy']
data.loc[(data.country == 'Italy') & (data.points >= 90)]
# isin is lets you select data whose value "is in" a list of values.
data.loc[data.country.isin(['Italy', 'France'])]
#isnull (and its companion notnull) let you highlight values which are (or are not) empty (NaN)
data.loc[data.price.notnull()]

####Assigning data
data['critic'] = 'everyone'
data['index_backwards'] = range(len(data), 0, -1)

