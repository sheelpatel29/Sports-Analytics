# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 16:22:21 2020

@author: sheel
"""


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df=pd.read_csv('data.csv')

df.describe()
df.drop('Loaned From',axis=1,inplace=True)



cols_to_be_dropped = ['ID','Unnamed: 0','Weak Foot','Release Clause','Wage','Photo', 'Nationality', 'Flag',
                      'Club Logo', 'International Reputation',
                      'Work Rate', 'Body Type', 'Real Face','Jersey Number', 'Joined', 
                      'Contract Valid Until','LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 
                      'RW','LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LWB', 
                      'LDM','CDM', 'RDM', 'RWB', 'LB', 'LCB','CB', 'RCB', 'RB',]
df.drop(cols_to_be_dropped,axis=1,inplace=True)
df.head()

df["Value"]=df["Value"].apply(lambda x: x.replace('€',''))



data=pd.read_csv('data.csv')

attributes = data.iloc[:, 54:83]
attributes['Skill Moves'] = data['Skill Moves']
workrate = data['Work Rate'].str.get_dummies(sep='/ ')
attributes = pd.concat([attributes, workrate], axis=1)
df = attributes
attributes = attributes.dropna()
df['Name'] = data['Name']
df = df.dropna()
print(attributes.columns)

def fix(x):
    # evaluate sum
    if('+' in str(x).strip()):
        calc = x.split('+')
        return int(calc[0]) + int(calc[1])

    # convert to integer if string contains a valid number
    elif str(x).strip().isdigit():
        return int(x)
    # return as it is, for example null values
    else:
         return x
for column in df.iloc[:,0:34]:
    df[column] = df[column].apply(fix)

data.Acceleration.unique()
data.info()



attributes.head()



#KNN ANLOGRITHM FOR RECOMENDATIONS
scaled = StandardScaler()
X = scaled.fit_transform(attributes)


recommendations = NearestNeighbors(n_neighbors=6,algorithm='ball_tree')
recommendations.fit(X)

player_index = recommendations.kneighbors(X)[1]

player_index


def get_index(x):
    return df[df['Name']==x].index.tolist()[0]

def recommend_me(player):
    print("5 Players similar to {} are : ".format(player))
    index=  get_index(player)
    for i in player_index[index][1:]:
        print(df.iloc[i]['Name'])
recommend_me('E. Hazard')

import pickle 
#Save the Modle to file in the current working directory
Pkl_Filename = "Scouting_Model.pkl"  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(recommendations, file)
