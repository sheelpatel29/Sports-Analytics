#  =-*` *- coding: utf-8 -*-
"""
Created on Mon Jun 22 12:44:23 2020

@author: sheel
"""
#import Required Packages 
%matplotlib inline

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
plt.style.use('bmh')                    
import re
import requests 
import bs4 
import datetime
import os
from scipy import stats
import plotly.offline as py
from plotly.offline import iplot, init_notebook_mode
import plotly.graph_objs as go
init_notebook_mode(connected = True)
from IPython.display import Markdown    


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

df=pd.read_csv("data.csv",low_memory=False)

df.columns


## Data Cleaning Steps
df.drop(columns={'Unnamed: 0','Flag','Club Logo','Photo'},inplace=True)
df=df.sort_values('Name').reset_index()
df.drop('index',axis=1,inplace=True)

# Checking the Count of Missing Values
df.isnull().sum().sort_values(ascending=False)

# Found that [Loaned From] column has 93% of the missing value 
#which should be droped bcoz it can skew the fifa
df.drop('Loaned From',axis=1,inplace=True)


# filing the nan values with forward fill of the values 
#for i in range(1,58):
    #df.iloc[:,i].fillna(method="ffill", inplace=True)

#There are special characters and certain symbolic notations which are to be removed
df["Value"]=df["Value"].apply(lambda x: x.replace('€',''))
df["Wage"]=df["Wage"].apply(lambda x: x.replace('€',''))


df.Value.head(50)

df.Value= (df.Value.replace(r'[KM]+$', '', regex=True).astype(float) *df.Value.str.extract(r'[\d\.]+([KM]+)', expand=False).fillna(1).replace(['K','M'], [10**3, 10**6]).astype(int))


df.Wage= (df.Wage.replace(r'[KM]+$', '', regex=True).astype(float) *df.Wage.str.extract(r'[\d\.]+([KM]+)', expand=False).fillna(1).replace(['K','M'], [10**3, 10**6]).astype(int))         
          
df['Release Clause'] = df['Release Clause'].str.slice(1,-1).astype(float)  
df['Release Clause']


#Choose the Columns which are useful and are enough for gaining insights 
chosen_columns = ['Name', 
                  'Age', 
                  'Nationality', 
                  'Overall', 
                  'Potential', 
                  'Club', 
                  'Value', 
                  'Wage', 
                  'Special',
                  'Preferred Foot', 
                  'International Reputation', 
                  'Weak Foot',
                  'Skill Moves', 
                  'Body Type', 
                  'Position',
                  'Jersey Number',
                  'Height', 
                  'Weight', 
                  'Crossing', 'Finishing', 'HeadingAccuracy',
                  'ShortPassing', 'Volleys', 'Dribbling', 'Curve', 'FKAccuracy',
                  'LongPassing', 'BallControl', 'Acceleration', 'SprintSpeed',
                  'Agility', 'Reactions', 'Balance', 'ShotPower', 'Jumping',
                  'Stamina', 'Strength', 'LongShots', 'Aggression', 'Interceptions',
                  'Positioning', 'Vision', 'Penalties', 'Composure', 'Marking',
                  'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling',
                  'GKKicking', 'GKPositioning', 'GKReflexes']


fifa=pd.DataFrame(df,columns=chosen_columns)
display(fifa.head())

fifa.shape

Describe=fifa.describe()


fifa['Club'].fillna('No Club', inplace = True)

#We assumed that ST should be te position for the missing values ( Would be validated by Machine Learning Algorithms)
fifa['Position'].fillna('ST', inplace = True)

#Taking the mean of the continuous variables and replacing the null values 
Mean_fifa = fifa.loc[:, ['Crossing', 'Finishing', 'HeadingAccuracy',
                                 'ShortPassing', 'Volleys', 'Dribbling', 'Curve', 'FKAccuracy',
                                 'LongPassing', 'BallControl', 'Acceleration', 'SprintSpeed',
                                 'Agility', 'Reactions', 'Balance', 'ShotPower', 'Jumping',
                                 'Stamina', 'Strength', 'LongShots', 'Aggression', 'Interceptions',
                                 'Positioning', 'Vision', 'Penalties', 'Composure', 'Marking',
                                 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling',
                                 'GKKicking', 'GKPositioning', 'GKReflexes']]
for i in Mean_fifa.columns:
    fifa[i].fillna(fifa[i].mean(), inplace = True)

fifa['International Reputation'].unique()
fifa['Height'].unique()
fifa['Jersey Number'].unique()
fifa['Weak Foot'].unique()

#categorical variables and will be filled by mode.
Mode_fifa = fifa.loc[:, ['Body Type','International Reputation', 'Height', 'Weight', 'Preferred Foot','Jersey Number']]
for i in Mode_fifa.columns:
    fifa[i].fillna(fifa[i].mode()[0], inplace = True)

#discrete numerical or continuous numerical variables.So the will be imputed by media
Median_fifa = fifa.loc[:, ['Weak Foot', 'Skill Moves', ]]
for i in Median_fifa.columns:
    fifa[i].fillna(fifa[i].median(), inplace = True)
    
#check if there are still remaining NA to be filled     
fifa.columns[fifa.isna().any()]

#Binning or quantization , to prepare numerical Dataset for assisting Machine Learning 

def general(fifa):
    return int(round((fifa[['HeadingAccuracy', 'Dribbling', 'Curve', 
                            'BallControl','Penalties']].mean()).mean()))

def mental(fifa):
    return int(round((fifa[['Aggression', 'Interceptions', 'Positioning', 
                            'Vision','Composure']].mean()).mean()))

def mobility(fifa):
    return int(round((fifa[['Acceleration', 'SprintSpeed', 
                            'Agility','Reactions']].mean()).mean()))

def power(fifa):
    return int(round((fifa[['Balance', 'Jumping', 'Stamina', 
                            'Strength']].mean()).mean()))

def shooting(fifa):
    return int(round((fifa[['Finishing', 'Volleys', 'FKAccuracy', 
                            'ShotPower','LongShots']].mean()).mean()))

def passing(fifa):
    return int(round((fifa[['Crossing', 'ShortPassing', 
                            'LongPassing']].mean()).mean()))

def defending(fifa):
    return int(round((fifa[['Marking', 'StandingTackle', 
                            'SlidingTackle']].mean()).mean()))

def goalkeeping(fifa):
    return int(round((fifa[['GKDiving', 'GKHandling', 'GKKicking', 
                            'GKPositioning', 'GKReflexes']].mean()).mean()))

def rating(fifa):
    return int(round((fifa[['Potential', 'Overall']].mean()).mean()))

fifa['General'] = fifa.apply(general, axis = 1)
fifa['Mental'] = fifa.apply(mental, axis =1)
fifa['Mobility'] = fifa.apply(mobility, axis = 1)
fifa['Power'] = fifa.apply(power, axis = 1)
fifa['Shooting'] = fifa.apply(shooting, axis = 1)
fifa['Passing'] = fifa.apply(passing, axis = 1)
fifa['Defending'] = fifa.apply(defending, axis =1)
fifa['Goalkeeping'] = fifa.apply(goalkeeping, axis = 1)
fifa['Rating'] = fifa.apply(rating, axis =1)

#Converting Weight and Height Variable into integer/float
def weight(value):
  x = value.replace('lbs', '')
  return float(x)

fifa['Weight'] = fifa['Weight'].apply(lambda x: weight(x))
def ft(x):
  calc= x.split("'")
  return int(int(calc[0])*12+int(calc[1]))
fifa['Height'] = fifa['Height'].apply(lambda x : ft(x))
  
display(fifa[['Weight', 'Height', 'Value', 'Wage']].head(3))

#Determining the Attributes to be considered while scouting a player for a given position
player_features = ('Acceleration', 'Aggression', 'Agility', 
                   'Balance', 'BallControl', 'Composure', 
                   'Crossing', 'Dribbling', 'FKAccuracy', 
                   'Finishing', 'GKDiving', 'GKHandling', 
                   'GKKicking', 'GKPositioning', 'GKReflexes', 
                   'HeadingAccuracy', 'Interceptions', 'Jumping', 
                   'LongPassing', 'LongShots', 'Marking', 'Penalties')

# Top four features for every position in football

for i, val in fifa.groupby(fifa['Position'])[player_features].mean().iterrows():
    print('Position {}: {}, {}, {}, {}'.format(i, *tuple(val.nlargest(4).index)))


display(fifa.head(2))

#Writing file 
fifa.to_csv('normalized_data.csv')












































          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          








































