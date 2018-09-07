# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 12:07:19 2018

@author: zahid
"""

# If you're working with a notebook, don't forget to use Matplotlib magic! 
# %matplotlib inline

import pandas as pd
import seaborn as sns
# Set the Seaborn theme if desired
sns.set_style('darkgrid')

times_df = pd.read_csv('data/timesData.csv', thousands = ',')
shanghai_df = pd.read_csv('data/shanghaiData.csv')

times_df.head()
times_df.describe()
shanghai_df.head()
shanghai_df.describe()

times_df.loc[2]
times_df.iloc[2]
times_df[2] # will give error
times_df[2:3]
# Retrieve the total score of the first row
times_df.loc[0, 'total_score']
# Retrieve rows 0 and 1
times_df.iloc[0:2]
times_df[0:2]
# Retrieve the values at columns and rows 1-3
times_df.iloc[1:4, 1:4]
times_df[1:4, 1:4] # will give error
# Retrieve the column `total_score`
times_df.loc['total_score'] # will give error, as loc and iloc are meant to be used for indices??
times_df['total_score']

# Are the last entries after 2006?
shanghai_df.loc[:-10, 'year']
shanghai_df[:, -1]