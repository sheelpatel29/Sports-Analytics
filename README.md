# Sports Analytics and Machine Learning - Project overview
 FIFA 19 is a soccer game released by EA Sports.

In this project, I have performed various data-cleaning techniques , Data-Visualization techniques and Machine Learning algorithms.
Going through this project you will recognize the "Data Science Project-from scratch" ideology and would help you understand some of the core steps involved
in as Data Science project.


# Summary of Sports Analytics Project
- Created a **recommendation** tool with (KNN Model) for Team manager to *scout for suitable replacement of a player* , during a transfer market opening
- Preformed Data Cleaning techniques on 20000 player dataset , including all the attributes of the players with reference to their preferred position
- Optimzed **Logistic regression, Decision tree Model and KNN model using GridsearchCV** to reach the max model accuracy.
- Performed **Binning or quantization** methodology to reduce the categorical variable in the dataset, which in turn are useful for Machine Learning
- Statistically presented the **best playing 11** for a given **club or a country** based on the Overall & Potential of a given player respective of the formation of the team.

# Code and Resources used:

**Python Version**: 3.7  
**Packages**: pandas, NumPy, Sklearn, matplotlib, seaborn, pickle, plotly  
**Dataset**: https://www.kaggle.com/karangadiya/fifa19  
**Articles Referenced**: https://github.com/khanhnamle1994/world-cup-2018  


# Data cleaning

After reading the data into my spyder framework, now its time to clean the dataset. I made the following cleaning methods and created a segregated data as follows:
- Dropped columns not related to a image or a link in the dataset
- Applied Lambda function to replace the currency signs and applied relevant mathematical functions on Value & Wage
- Created a loop to iterate through all the rows and find out if there is basic math's to be conducted , such as addition and Subtraction
- Choose certain columns which will be useful for gaining relevant Insights from the dataset
- Applied Binning and quantization ( to prepare numerical dataset for easy to ML), and created a new columns into the dataset
  * General.
  * Mental.
  * Mobility.
  * Power.
  * Shooting.
  * Passing.
  * Defending.
  * Goalkeeping.
  * Rating.

# Data Visualization

I looked at various interesting Insights given in the dataset through Visualization of Continuous, Numerically distinct, and combination of both variables. Few highlighted Insights are as follows:

![alt text](https://github.com/sheelpatel29/Sports-Analytics/blob/master/Univariate%20Analysis%201.png "Distribution of Age")
![alt text](https://github.com/sheelpatel29/Sports-Analytics/blob/master/Correlation%20matrix.png "Correlation matrix")
![alt text](https://github.com/sheelpatel29/Sports-Analytics/blob/master/soccer%20position%20mapping.png "Soccer Position Map")


# Model Building

Various steps performed during model building are as follows:
- Train_Test_split the dataset into 70-30 Train & Test data respectively
- Trained the data on 3 models to find out the best model to categorize the data correctly with maximum accuracy.

 Models implemented are as follows :
  *  Logistic regression : **Accuracy:  0.8807 | F1 score:  0.8806**     
  *  KNN clustering Algorithms : **Accuracy:  0.8829 | F1 score:  0.8826**  
  *  Decision tree classifier:  **Accuracy:  0.8636 | F1 score:  0.8629**  

- All the 3 models are performing pretty well , as their accuracy is so close. So far KNN clustering seems to be the best Model
# Future Scope of Scaling

I would have created an API , based on the model of recommendation, this can be considered as a future scope if anyone wants to expand the scope of the project.
