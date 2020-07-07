# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 16:22:11 2020

@author: sheel
"""




# Univariate Analysis 

#Importing relevant packages 
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

df=pd.read_csv("normalized_data.csv",low_memory=False)
fifa=pd.read_csv("normalized_data.csv",low_memory=False)




#If the variable is Categorical or Numerical, We can see the distribution of data with Histogram and Density plot for numerical variable 
# and for Categorical variable we need to follow bar plot 


#Drop the features that would not be useful anymore
df.drop(columns = ['Unnamed: 0','Crossing', 'Finishing', 'HeadingAccuracy',
                     'ShortPassing', 'Volleys', 'Dribbling', 'Curve', 'FKAccuracy',
                     'LongPassing', 'BallControl', 'Acceleration', 'SprintSpeed',
                     'Agility', 'Reactions', 'Balance', 'ShotPower', 'Jumping',
                     'Stamina', 'Strength', 'LongShots', 'Aggression', 'Interceptions',
                     'Positioning', 'Vision', 'Penalties', 'Composure', 'Marking',
                     'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling',
                     'GKKicking', 'GKPositioning', 'GKReflexes', 'Overall', 'Potential'], inplace = True, axis =1)


#Features Remaining after Dropping
display(Markdown('**Features Remaining after Dropping:**'))
display(df.columns.values)


# Function to distribution plot

def distplot ( variable,color):
    global ax 
    font_size=16
    title_size=20
    plt.rcParams['figure.figsize']=(18,7)
    ax=sns.distplot(variable,color=color)
    plt.xlabel('%s' %variable.name, fontsize = font_size)
    plt.ylabel('Count of the players', fontsize = font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.title('%s' %variable.name + 'Distribution of Players',fontsize = title_size)
    plt.show()

#function to create count plot
def countplot(variable, title,  color):
    global ax
    font_size = 14
    title_size = 20
    plt.rcParams['figure.figsize'] = (18, 8)
    ax = sns.countplot(variable, palette = color)
    plt.xlabel('%s' %variable.name, fontsize = font_size)
    plt.xticks(fontsize = font_size)
    plt.yticks(fontsize = font_size)
    plt.title(title, fontsize = title_size)
    plt.show()






# Age is Positively skewed, age between 20 and 29 is where the most players are 
distplot(df['Age'], 'b')

#Value is highly skewed
distplot(df['Value'], 'g')

#Wage is highly skewed
distplot(df['Wage'], 'teal')

# height is normal distribution 
distplot(df['Height'], 'k')
#Weight seems like normal distribution
distplot(df['Weight'], 'c')

#Right is more preferred foot than left
countplot(df['Preferred Foot'], 'Most Preferred Foot of the Player', 'Set2')

# There is a balance of the weak foot distribution at 3 
countplot(df['Weak Foot'], 'Distribution of weak foot', 'Paired')
# Most players are playing ST  ( This is interesting)
countplot(df['Position'], 'Distribution of Players position', 'copper')

clubs = ('Arsenal','Liverpool', 'RC Celta', 'Empoli', 'Atlético Madrid', 'Manchester City',
             'Tottenham Hotspur', 'FC Barcelona', 'Valencia CF', 'Chelsea', 'Real Madrid')
data_clubs = df.loc[df['Club'].isin(clubs) & df['Rating']]


sns.heatmap(df[['General', 'Mental', 'Mobility', 'Power', 'Shooting',
       'Passing', 'Defending', 'Goalkeeping', 'Rating']].corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.1f')
plt.title('Heatmap of the Dataset', fontsize = 30)
plt.show()


#Top Clubs in FIFA
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
plt.rcParams['figure.figsize'] = (15, 15)
wordcloud = WordCloud(background_color = 'white', width = 1200,  height = 1200, max_words = 121).generate(' '.join(df['Club']))
plt.imshow(wordcloud)
plt.axis('off')
plt.title('Most Popular Club',fontsize = 30)
plt.show()


#Football pitch and Positions location in the game of football 

def draw_pitch(pitch, line, orientation,view):
    
    orientation = orientation
    view = view
    line = line
    pitch = pitch
    
    if view.lower().startswith("h"):
        fig,ax = plt.subplots(figsize=(20.8,13.6))
        plt.ylim(98,210)
        plt.xlim(-2,138)
    else:
        fig,ax = plt.subplots(figsize=(13.6,20.8))
        plt.ylim(-2,210)
        plt.xlim(-2,138)
    ax.axis('off') 
    
    # this hides the x and y ticks

    # side and goal lines #
    lx1 = [0,0,136,136,0]
    ly1 = [0,208,208,0,0]

    plt.plot(lx1,ly1,color=line,zorder=5)

    # boxes, 6 yard box and goals
        #outer boxes#
    lx2 = [27.68,27.68,108.32,108.32] 
    ly2 = [208,175,175,208]
    plt.plot(lx2,ly2,color=line,zorder=5)

    lx3 = [27.68,27.68,108.32,108.32] 
    ly3 = [0,33,33,0]
    plt.plot(lx3,ly3,color=line,zorder=5)

        #goals#
    lx4 = [60.68,60.68,75.32,75.32]
    ly4 = [208,208.4,208.4,208]
    plt.plot(lx4,ly4,color=line,zorder=5)

    lx5 = [60.68,60.68,75.32,75.32]
    ly5 = [0,-0.4,-0.4,0]
    plt.plot(lx5,ly5,color=line,zorder=5)

       #6 yard boxes#
    lx6 = [49.68,49.68,86.32,86.32]
    ly6 = [208,199,199,208]
    plt.plot(lx6,ly6,color=line,zorder=5)

    lx7 = [49.68,49.68,86.32,86.32]
    ly7 = [0,9,9,0]
    plt.plot(lx7,ly7,color=line,zorder=5)

    #Halfway line, penalty spots, and kickoff spot
    lx8 = [0,136] 
    ly8 = [104,104]
    plt.plot(lx8,ly8,color=line,zorder=5)


    plt.scatter(68,186,color=line,zorder=5)
    plt.scatter(68,22,color=line,zorder=5)
    plt.scatter(68,104,color=line,zorder=5)

    circle1 = plt.Circle((68,187), 18.30,ls='solid',lw=3,color=line, fill=False, zorder=1,alpha=1)
    circle2 = plt.Circle((68,21), 18.30,ls='solid',lw=3,color=line, fill=False, zorder=1,alpha=1)
    circle3 = plt.Circle((68,104), 18.30,ls='solid',lw=3,color=line, fill=False, zorder=2,alpha=1)


    ## Rectangles in boxes
    rec1 = plt.Rectangle((40, 175), 60,33,ls='-',color=pitch, zorder=1,alpha=1)
    rec2 = plt.Rectangle((40, 0), 60,33,ls='-',color=pitch, zorder=1,alpha=1)

    ## Pitch rectangle
    rec3 = plt.Rectangle((-1, -1), 140,212,ls='-',color=pitch, zorder=1,alpha=1)

    ax.add_artist(rec3)
    ax.add_artist(circle1)
    ax.add_artist(circle2)
    ax.add_artist(rec1)
    ax.add_artist(rec2)
    ax.add_artist(circle3)   
draw_pitch("#195905","#faf0e6","v","full")
x = [68, 68, 68, 32, 104, 68, 32, 104, 68, 44, 88, 20, 116, 12, 124, 68, 68, 16, 120, 16, 120, 40, 96, 32, 104, 32, 104]
y = [186, 150, 1, 150, 150, 112, 114, 114, 14, 16, 16, 24, 24, 50, 50, 50, 74, 74, 74, 130, 130, 74, 74, 186, 186, 50, 50]
n = ['ST', 'CF', 'GK', 'LF', 'RF', 'CAM', 'LAM', 'RAM', 'CB', 'LCB', 'RCB', 'LB', 'RB', 'LWB', 'RWB', 'CDM', 'CM', 'LM', 'RM', 'LW', 'RW', 'LCM', 'RCM', 'LS', 'RS', 'LDM', 'RDM']

for i,type in enumerate(n):
    x_c = x[i]
    y_c = y[i]
    plt.scatter(x_c, y_c, marker='o', color='red', edgecolors="black", zorder=10)
    plt.text(x_c-2.5, y_c+1, type, fontsize=16)




import pitch
import turtle

#A Procedure to draw a player at the given position
#def drawPlayer(color,x,y,label):
  #screen = turtle.Screen()
  #creen.tracer(0)
  #myPen = turtle.Turtle()
  #myPen.hideturtle()
  #myPen.penup()
  #myPen.goto(x,y)
  #myPen.fillcolor(color)
  #myPen.begin_fill()
  #myPen.circle(10)
  #myPen.end_fill()
  #screen.tracer(1)  
  #myPen.penup()
  #myPen.goto(x+10,y)
  #myPen.color(color)
  #myPen.write(label)

#MAIN PROGRAM STARTS HERE
#pitch.drawPitch()


#drawPlayer("blue",-0,-194,"Goal Keeper") 
#drawPlayer("yellow",-50,-120,"Centre Back") 
#drawPlayer("yellow",50,-120,"Centre Back") 
#Add more players using the relevant parameters






#Referenced from : https://github.com/khanhnamle1994/world-cup-2018/blob/master/Ideal_Team_Lineup_World_Cup_2018.ipynb



#Best Squad of a given national team or Club based on the formation , 'Overall' is the deciding factor 
# for a player to be selected in starting lineup


def get_best_squad(formation):
    FIFA18_copy = fifa.copy()
    store = []
    
    # iterate through all positions in the input formation and get players with highest overall respective to the position
    for i in formation:
        store.append([
            i,
            FIFA18_copy.loc[[FIFA18_copy[FIFA18_copy['Position'] == i]['Overall'].idxmax()]]['Name'].to_string(index = False),
            FIFA18_copy[FIFA18_copy['Position'] == i]['Overall'].idxmax(),
            FIFA18_copy.loc[[FIFA18_copy[FIFA18_copy['Position'] == i]['Overall'].idxmax()]]['Age'].to_string(index = False),
            FIFA18_copy.loc[[FIFA18_copy[FIFA18_copy['Position'] == i]['Overall'].idxmax()]]['Club'].to_string(index = False),
            FIFA18_copy.loc[[FIFA18_copy[FIFA18_copy['Position'] == i]['Overall'].idxmax()]]['Value'].to_string(index = False),
            FIFA18_copy.loc[[FIFA18_copy[FIFA18_copy['Position'] == i]['Overall'].idxmax()]]['Wage'].to_string(index = False)
        ])
                      
        FIFA18_copy.drop(FIFA18_copy[FIFA18_copy['Position'] == i]['Overall'].idxmax(), 
                         inplace = True)
    
    # return store with only necessary columns
    return pd.DataFrame(np.array(store).reshape(11,7),
                        columns = ['Position', 'Player', 'Overall', 'Age', 'Club', 'Value', 'Wage']).to_string(index = False)
squad_433 = ['GK', 'RB', 'CB', 'CB', 'LB', 'CDM', 'CM', 'CAM', 'RW', 'ST', 'LW']
print ('4-3-3')
print (get_best_squad(squad_433))



def get_best_squad_n(formation, club, measurement = 'Overall'):
    FIFA18_copy = fifa.copy()
    FIFA18_copy = FIFA18_copy[FIFA18_copy['Club'] == club]
    store = []
    
    for i in formation:
        store.append([
            FIFA18_copy.loc[[FIFA18_copy[FIFA18_copy['Position'].str.contains(i)][measurement].idxmax()]]['Position'].to_string(index = False),
            FIFA18_copy.loc[[FIFA18_copy[FIFA18_copy['Position'].str.contains(i)][measurement].idxmax()]]['Name'].to_string(index = False), 
            FIFA18_copy[FIFA18_copy['Position'].str.contains(i)][measurement].max(),
            FIFA18_copy.loc[[FIFA18_copy[FIFA18_copy['Position'].str.contains(i)][measurement].idxmax()]]['Age'].to_string(index = False),
            FIFA18_copy.loc[[FIFA18_copy[FIFA18_copy['Position'].str.contains(i)][measurement].idxmax()]]['Nationality'].to_string(index = False),
            FIFA18_copy.loc[[FIFA18_copy[FIFA18_copy['Position'].str.contains(i)][measurement].idxmax()]]['Value'].to_string(index = False),
            FIFA18_copy.loc[[FIFA18_copy[FIFA18_copy['Position'].str.contains(i)][measurement].idxmax()]]['Wage'].to_string(index = False)
        ])
        
        FIFA18_copy.drop(FIFA18_copy[FIFA18_copy['Position'].str.contains(i)][measurement].idxmax(), 
                         inplace = True)
    
    return np.mean([x[2] for x in store]).round(2), pd.DataFrame(np.array(store).reshape(11,7), 
                                                                 columns = ['Position', 'Player', measurement, 'Age', 'Nationality', 'Value', 'Wage']).to_string(index = False)




def get_summary_n(squad_list, squad_name, nationality_list):
    summary = []

    for i in nationality_list:
        count = 0
        for j in squad_list:
            
            # for overall rating
            O_temp_rating, _  = get_best_squad_n(formation = j, nationality = i, measurement = 'Overall')
            
            # for potential rating & corresponding value
            P_temp_rating, _ = get_best_squad_n(formation = j, nationality = i, measurement = 'Potential')
            
            summary.append([i, squad_name[count], O_temp_rating.round(2), P_temp_rating.round(2)])    
            count += 1
    
    return summary










squad_343_strict = ['GK', 'CB', 'CB', 'CB', 'RB|RWB', 'CM|CDM', 'CM|CDM', 'LB|LWB', 'RM|RW', 'ST|CF', 'LM|LW']
squad_442_strict = ['GK', 'RB|RWB', 'CB', 'CB', 'LB|LWB', 'RM', 'CM|CDM', 'CM|CAM', 'LM', 'ST|CF', 'ST|CF']
squad_4312_strict = ['GK', 'RB|RWB', 'CB', 'CB', 'LB|LWB', 'CM|CDM', 'CM|CAM|CDM', 'CM|CAM|CDM', 'CAM|CF', 'ST|CF', 'ST|CF']
squad_433_strict = ['GK', 'RB|RWB', 'CB', 'CB', 'LB|LWB', 'CM|CDM', 'CM|CAM|CDM', 'CM|CAM|CDM', 'RM|RW', 'ST|CF', 'LM|LW']
squad_4231_strict = ['GK', 'RB|RWB', 'CB', 'CB', 'LB|LWB', 'LDM', 'CM|CDM', 'RM|RW', 'AM', 'LM|LW', 'ST|CF']
squad_list = [squad_343_strict, squad_442_strict, squad_4312_strict, squad_433_strict, squad_4231_strict]
squad_name = ['3-4-3', '4-4-2', '4-3-1-2', '4-3-3', '4-2-3-1']


rating_4231_FR_Overall, best_list_4231_FR_Overall = get_best_squad_n(squad_4231_strict, 'Arsenal', 'Overall')
print('4-2-3-1 formation')
print('Average rating: {:.1f}'.format(rating_4231_FR_Overall))
print(best_list_4231_FR_Overall)


Spain = pd.DataFrame(np.array(get_summary_n(squad_list, squad_name, ['Spain'])).reshape(-1,4), columns = ['Nationality', 'Squad', 'Overall', 'Potential'])
Spain.set_index('Nationality', inplace = True)
Spain[['Overall', 'Potential']] = Spain[['Overall', 'Potential']].astype(float)

print (Spain)

def country(x):
    return d_f[d_f['Nationality'] == x][['Name','Overall','Potential','Position']]


# let's check the Indian Players 
country('India')

def club(x):
    return data[data['Club'] == x][['Name','Jersey Number','Position','Overall','Nationality','Age','Wage',
                                    'Value','Contract Valid Until']]

club('Manchester United')
fifa[fifa['Club']=='FC Barcelona'].iloc[:,15:20].head(30)

## fifa Visualization 


# Potential tends to fall as you grow old 

df.to_csv('Data_Visualization.csv')
