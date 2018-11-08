# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 12:07:19 2018

@author: zahid
"""

# If you're working with a notebook, don't forget to use Matplotlib magic! 
# %matplotlib inline

import matplotlib.pyplot as plt
# %matplotlib qt  ## for separate window for plot in spyder
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
times_df[2:3]
# Retrieve the total score of the first row
times_df.loc[0, 'total_score']
# multiple columns
times_df.loc[0, ['total_score','income']]
# Retrieve rows 0 and 1
times_df.iloc[0:2]
times_df[0:2]  # same as times_df.loc[0:2]
# Retrieve the values at columns and rows 1-3
times_df.iloc[1:4, 1:4]
times_df[1:4, 1:4] # will give error
# Retrieve the column `total_score`
times_df.loc['total_score'] # will give error, as loc and iloc are meant to be used for indices??
times_df.loc[:, 'total_score']  # no error
times_df['total_score']
times_df.total_score

times_df.dtypes
times_df['total_score'].dtype
times_df.total_score.dtype

# Are the last 10 entries after 2006?
shanghai_df.iloc[-10:]['year'] > 2006
# Was the alumni count higher than 90 for the first  ten universities?
shanghai_df.iloc[:10]['alumni'] > 90

# Query `shanghai_df` for universities with total score between 40 and 50
average_schools = shanghai_df.query('total_score > 40 and total_score < 50')
# universities with a first national rank and a first world rank
shanghai_df.query('national_rank == "1" and world_rank == "1"')
# universities with alumni numbers greater than 20
shanghai_df.query('alumni > 20')

## method chaining ######################################################
# Extract info
def extract_info(input_df, name):
    df = input_df.copy()
    info_df = pd.DataFrame({'nb_rows': df.shape[0], 
                            'nb_cols': df.shape[1], 
                            'name': name}, index=range(1))
    return info_df
# Gather all info
all_info = pd.concat([times_df.pipe(extract_info, 'times'), 
                      shanghai_df.pipe(extract_info, 'shanghai')])
# or
all_info = pd.concat([times_df.pipe((extract_info, "input_df"), 'times'), 
                      shanghai_df.pipe(extract_info, 'shanghai')])
## use of pipe() still not clear
## concat takes union of columns in both datasets
    
common_columns = set(times_df.columns) & set(shanghai_df.columns)
print(common_columns)

# Clean up the `world_rank` 
def clean_world_rank(input_df):
    df = input_df.copy()
    df.world_rank = df.world_rank.str.split("-").str[0].str.split("=").str[0]
    return df

# assign common years of shanghai_df and times_df to common_years
common_years = set(shanghai_df.year) & set(times_df.year)
print(common_years)

# filter years
def filter_year(input_df, years):
    df = input_df.copy()
    return df.query('year in {}'.format(list(years)))
# shanghai_df.query('year in [2011,2012]')  # works
# shanghai_df.query('year in (2011,2012)')  # also works

# clean times_df and shanghai_df
cleaned_times_df = times_df.loc[:, common_columns].pipe(filter_year, common_years).pipe(clean_world_rank).assign(name = "times")
# assign - For adding new columns to a DataFrame in a chain
cleaned_shanghai_df = shanghai_df.loc[:, common_columns].pipe(filter_year, common_years).pipe(clean_world_rank).assign(name = "shanghai")

ranking_df = pd.concat([cleaned_times_df, cleaned_shanghai_df])

# Calculate the percentage of missing data
# pd.isnull() returns a boolean pd.Series with True for null entries
# sum() on boolean pd.Series gives count of True entries
# len(pd.DataFrame) returns number of rows, ie, shape[0]
missing_data = 100 * pd.isnull(ranking_df.total_score).sum() / len(ranking_df)

# Drop the `total_score` column of `ranking_df`
# df.drop(label, axis)  # returns dropped label; axis = 0 for row, axis = 1 for column
ranking_df = ranking_df.drop("total_score", axis = 1)


## Memory Optimization ##################################################
# Print the memory usage of `ranking_df`
ranking_df.info()
# Print the deep memory usage of `ranking_df`
ranking_df.info(memory_usage = "deep")

def memory_change(input_df, column, dtype):
    df = input_df.copy()
    old = round(df[column].memory_usage(deep = True)/1024, 2)  # in KB
    new = round(df[column].astype(dtype).memory_usage(deep = True)/1024, 2)
    change = round(100 * (old - new) / old, 2)
    report = """The initial memory footprint for {column} is: {old} KB.
    The casted {column} now takes: {new} KB.
    A change of {change} %.""".format(**locals())
    return report

print(memory_change(ranking_df, "university_name", "category"))
print(memory_change(ranking_df, "world_rank", "int16"))
print(memory_change(ranking_df, "name", "category"))

# cast columns to suitable types for memory optimization
ranking_df.world_rank = ranking_df.world_rank.astype("int16")
ranking_df.university_name = ranking_df.university_name.astype("category")
ranking_df.name = ranking_df.name.astype("category")
ranking_df.info(memory_usage = "deep")
# can optimize even further by casting the year column down to int32


## GroupBy ##############################################################
# Query for the rows with university name 'Massachusetts Institute of Technology (MIT)'
ranking_df.query('university_name == "Massachusetts Institute of Technology (MIT)"')
ranking_df[ranking_df.university_name == "Massachusetts Institute of Technology (MIT)"]  # same result as above line
# Localize the rows with the MIT university name and replace value 
ranking_df.loc[ranking_df.university_name == "Massachusetts Institute of Technology (MIT)", "university_name"] = "Massachusetts Institute of Technology"
# or
ranking_df.loc[lambda df: df.university_name == 'Massachusetts Institute of Technology (MIT)', 'university_name'] = 'Massachusetts Institute of Technology'

# find the 5 top universities over the years, for each ranking system
# Load in `itertools`
import itertools
# example use of itertools.product
for i in itertools.product(common_years, ["times", "shanghai"]):
    print(i)
# initialize ranking
ranking = {}
for year, name in itertools.product(common_years, ["times", "shanghai"]):
    s = ranking_df.loc[(ranking_df.year == year) & (ranking_df.name == name) & (ranking_df.world_rank.isin(range(1,6))), :].sort_values("world_rank", ascending = False).university_name
    ## or
    # s = (ranking_df.loc[lambda df: ((df.year == year) & (df.name == name) & (df.world_rank.isin(range(1,6)))), :].sort_values('world_rank', ascending=False).university_name)
    # print(s)
    ranking[(year, name)] = list(s)
print(ranking)

## Import `defaultdict`
from collections import defaultdict
## Initialize `compare`
compare = defaultdict(list)
## Initialize `exact_similarity` and `set_similarity`
exact_similarity, set_similarity = {}, {}
c = {}
for (year, method), universities in ranking.items():
    compare[year].append(universities)
#    if c.get(year, None):
#        c[year].append(universities)
#    else:
#        c[year] = [universities]
for year, ranks in compare.items():
    set_similarity[year] = 100 * len(set(ranks[0]) & set(ranks[1])) / 5
print(set_similarity)

## easier method to do above work
## Construct a DataFrame with the top 5 universities 
top_5_df = ranking_df.loc[lambda df: df.world_rank.isin(range(1,6)), :]
# top_5_df = ranking_df.loc[lambda df: df.world_rank.isin(range(1,6))]  # same result
top_5_df.head()
top_5_df.pivot(values = "world_rank", columns = "name", index = "university_name").dropna()  # will not work
# Compute the similarity
def compute_set_similarity(s):
    pivoted = s.pivot(values = "world_rank", columns = "name", index = "university_name").dropna()
    set_similarity = 100 * len(set(pivoted["shanghai"].index) & set(pivoted["times"].index)) / 5
    return set_similarity
# Group `top_5_df` by `year`
grouped_df = top_5_df.groupby("year")
# Use compute_similarity to construct a dataframe
setsimilarity_df = pd.DataFrame({"set_similarity": grouped_df.apply(compute_set_similarity)}).reset_index()
setsimilarity_df.head()


## Vizualization ########################################################
# Plot a scatterplot with `total_score` and `alumni`
shanghai_df.plot.scatter("total_score", "alumni", c = "year", colormap = "viridis")
plt.show()

# Replace '-' entries with NaN values
times_df["total_score"] = times_df["total_score"].replace("-", "NaN").astype("float")
# Drop all rows with NaN values in num_students
times_df = times_df.dropna(subset = ["num_students"], how = "all")
# Cast the remaining rows with `num_students` as int
times_df["num_students"] = times_df["num_students"].astype("int")
# Plot a scatterplot with `total_score` and `num_students`
times_df.plot.scatter("total_score", "num_students", c = "year", colormap = "viridis")
plt.show()

# Abbreviate country names of United States and United Kingdom
times_df['country'] = times_df['country'].replace("United States of America", "USA").replace("United Kingdom", "UK")
# Count the frequency of countries
count = times_df['country'].value_counts()[:10]
# Convert the top 10 countries to a DataFrame
df = count.to_frame()
# Reset the index
df.reset_index(level = 0, inplace = True)
# Rename the columns
df.columns = ['country', 'count']
# Plot a barplot with `country` and `count`
sns.barplot(x = 'country', y = 'count', data = df)
sns.despine()
plt.show()
# Barplot with `country` and `total_score`
sns.barplot(x = 'country', y = 'total_score', data = times_df)
sns.barplot(x = 'country', y = 'total_score', data = times_df_filtered)
# pairwise relationships
import numpy as np
np.seterr(invalid = 'ignore')
sns.pairplot(times_df, hue = 'country')


## some peculiarities #########################################################
times_df[2] # will give error; since label and index are integers, confuses machine
times_df[2:3]  # gives no error

shanghai_df.iloc[:10]  # record with index 10 not included
shanghai_df.loc[:10]  # still iterable; record with label 10 also included