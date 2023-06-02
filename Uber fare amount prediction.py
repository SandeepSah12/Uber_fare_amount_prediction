# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 17:44:27 2023

@author: sande
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn import metrics
import math
from statsmodels.tools.eval_measures import rmse
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense


data=pd.read_csv(r'uber.csv')
data
df=data.copy()

pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)

df.head()

df.info()

df.columns

df = df.drop(['Unnamed: 0', 'key'], axis = 1)

df.shape

df.dtypes

df.info()

df.describe()

df.isnull().sum()

df.head()

df.dtypes

df.pickup_datetime = pd.to_datetime(df.pickup_datetime, errors='coerce')

# For datetime64[ns] types NaT represents missing values

df.dtypes

df= df.assign(hour = df.pickup_datetime.dt.hour,
             day= df.pickup_datetime.dt.day,
             month = df.pickup_datetime.dt.month,
             year = df.pickup_datetime.dt.year,
             dayofweek = df.pickup_datetime.dt.dayofweek)

df.info()

from math import *
def distance_transform(longitude1, latitude1, longitude2, latitude2):
    travel_dist = []
    
    for pos in range(len(longitude1)):
        long1,lati1,long2,lati2 = map(radians,[longitude1[pos],latitude1[pos],longitude2[pos],latitude2[pos]])
        dist_long = long2 - long1
        dist_lati = lati2 - lati1
        a = sin(dist_lati/2)**2 + cos(lati1) * cos(lati2) * sin(dist_long/2)**2
        c = 2 * asin(sqrt(a))*6371
        travel_dist.append(c)
       
    return travel_dist

df['dist_travel_km'] = distance_transform(df['pickup_longitude'].to_numpy(),
                                                df['pickup_latitude'].to_numpy(),
                                                df['dropoff_longitude'].to_numpy(),
                                                df['dropoff_latitude'].to_numpy())

df.head()

# drop the column 'pickup_daetime' using drop()

# 'axis = 1' drops the specified column

df = df.drop('pickup_datetime', axis=1)

df.info()

df.describe().transpose()

df.columns[df.dtypes == 'object']

df.head()

df.fare_amount.min()

plt.figure(figsize=(20, 12))
sns.boxplot(data=df)

medianFiller = lambda x : x.fillna(x.median())
numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
df[numeric_columns] = df[numeric_columns].apply(medianFiller, axis = 0)

# outliers detection using boxplot

plt.figure(figsize=(20, 30))

for i, variable in enumerate(numeric_columns):
    plt.subplot(6, 5, i +1)
    plt.boxplot(df[variable], whis=1.5)
    plt.tight_layout()
    plt.title(variable)
plt.show()

df.shape

df.head()

def remove_outlier(df1 , col):
    Q1 = df1[col].quantile(0.25)
    Q3 = df1[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_whisker = Q1-1.5*IQR
    upper_whisker = Q3+1.5*IQR
    df[col] = np.clip(df1[col] , lower_whisker , upper_whisker)
    return df1

def treat_outliers_all(df1 , col_list):
    for c in col_list:
        df1 = remove_outlier(df , c)
    return df1

df = treat_outliers_all(df, df.iloc[:, 0::])

plt.figure(figsize=(20,30))
for i , variable in enumerate(df.iloc[: , 0::]):
    plt.subplot(6,5,i+1)
    plt.boxplot(df[variable] , whis = 1.5)
    plt.tight_layout()
    plt.title(variable)
plt.show()

#We will only keep the observation where travel distance is less than or equal to 130

df = df.loc[(df.dist_travel_km>=1) | (df.dist_travel_km <=130)]
print("Remaining observastions in the dataset:", df.shape)

incorrect_coordinates = df.loc[(df.pickup_latitude > 90) |(df.pickup_latitude < -90) |
                                   (df.dropoff_latitude > 90) |(df.dropoff_latitude < -90) |
                                   (df.pickup_longitude > 180) |(df.pickup_longitude < -180) |
                                   (df.dropoff_longitude > 90) |(df.dropoff_longitude < -90)
                                    ]

df.drop(incorrect_coordinates, inplace = True, errors = 'ignore')

# sort the variables on the basis of total null values in the variable

# 'isnull().sum()' returns the number of missing values in each variable

# 'ascending = False' sorts values in the descending order

# the variable with highest number of missing values will appear first

Total = df.isnull().sum().sort_values(ascending = False)

# calculate the percentage of missing values
# 'ascending = False' sorts values in the descending order
# the variable with highest percentage of missing values will appear first

Percent = (df.isnull().sum()*100/df.isnull().count()).sort_values(ascending = False)

# concat the 'Total' and 'Percent' columns using 'concat' function
# 'keys' is the list of column names
# 'axis = 1' concats along the columns

missing_data = pd.concat([Total, Percent], axis = 1, keys = ['Total', 'Percentage of Missing Values'])    

# add the column containing data type of each variable

missing_data['Type'] = df[missing_data.index].dtypes
missing_data

# plot heatmap to visualize the null values in each column
# 'cbar = False' does not show the color axis 

sns.heatmap(df.isnull())

# display the plot

plt.show()

corr = df.corr()
corr

# set the plot size
# pass the required height and width to the parameter, 'figsize'

plt.figure(figsize = (30, 20))

# use 'mask' to plot a upper triangular correlation matrix 
# 'tril_indices_from' returns the indices for the lower-triangle of matrix
# 'k = -1' consider the diagonal of the matrix

mask = np.zeros_like(corr)
mask[np.tril_indices_from(mask, k=-1)]=True

# plot the heat map
# corr: gives the correlation matrix
# cmap: color code used for plotting
# vmax: gives a maximum range of values for the chart
# vmin: gives a minimum range of values for the chart
# annot: prints the correlation values in the chart
# annot_kws: sets the font size of the annotation
# mask: masks the upper traingular matrix values

sns.heatmap(corr, cmap = 'RdYlGn', vmax = 1.0, vmin = -1.0, annot = True, annot_kws = {"size": 20}, mask = mask)

# set the size of x and y axes labels
# set text size using 'fontsize'

plt.xticks(fontsize = 12)
plt.yticks(fontsize = 15)

# display the plot
plt.show()

# set figure size

plt.figure(figsize=(30,20))

# plot the heat map
# corr: give the correlation matrix
# pass the condition to get the strong correlation between the variables
# cmap: color code used for plotting
# vmax: gives a maximum range of values for the chart
# vmin: gives a minimum range of values for the chart
# annot: prints the correlation values in the chart
# annot_kws: sets the font size of the annotation
#sns.set(font_scale=2)

sns.heatmap(corr[(corr >= 0.7) | (corr <= -0.7)], cmap = 'RdYlGn', vmax = 1.0, vmin = -1.0, annot = True, 
            annot_kws={"size": 20})

# set the size of x and y axes labels
# set text size using 'fontsize'

plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)

# display the plot
plt.show()

# Set the figure size, grid type and color palette along with appropraite titel for the plot

plt.figure(figsize=(10,5))
plt.title('Cab trips count during week days')
sns.set_style("white")

#plot the countplot for weekdays

sns.countplot(x='hour', data=df.loc[(df.dayofweek >= 0) & (df.dayofweek <=4)], palette='Set2')

# Set the figure size, grid type and color palette along with appropraite titel for the plot

plt.figure(figsize=(10,5))
plt.title('Cab trips count during week ends')
sns.set_style("white")

#plot the countplot for weekends

sns.countplot(x='hour', data=df.loc[(df.dayofweek >= 5) & (df.dayofweek <=6)], palette='Set2')

# Set the days in the dataset as week days and week ends

week_days = df.loc[(df.dayofweek >= 0) & (df.dayofweek <= 4)]
week_ends = df.loc[(df.dayofweek >= 5) & (df.dayofweek <= 6)]

# compute the mean fare amount over the week day and week end.
# use groupby('hour') to get the mean fare for each hour

week_days_fare = week_days.groupby(['hour']).fare_amount.mean().to_frame().reset_index()
week_ends_fare = week_ends.groupby(['hour']).fare_amount.mean().to_frame().reset_index()

# hours

x=np.array(week_days_fare.hour)

# an array of week day fare 

y = np.array(week_days_fare.fare_amount)

# an array of week end fare

z = np.array(week_ends_fare.fare_amount)

# Set the figure size, title, x and y labels

plt.figure(figsize = (20,10))
plt.title('Mean Fare Amount For Each Hour - Weekdays Vs Weekends')
plt.xlabel('Hours')
plt.ylabel('Mean Fare')

# Pass the three integers. The value of these integer should be less that 10

ax=plt.subplot(1,1,1)
ax.bar(x-0.2, y, width=0.2, color='red', align='center', label = 'Week days')
ax.bar(x, z, width=0.2, color='blue', align='center', label = 'Week ends')
plt.xticks(range(0,24))
plt.legend()
plt.show()

plt.figure(figsize=(20,10))
sns.set_style("darkgrid")
plt.title("Distribution of the fare amount")
plt.xlabel("Fare Amount")
plt.ylabel("Frequency")
plt.xlim(-10,20)
plt.xticks(range(0,200,5))

snsplot = sns.kdeplot(df.fare_amount, shade=True)

plt.figure(figsize = (20,10))
sns.style = ('darkgrid')
plt.title("Distribution of the trip distance")
plt.xlabel("Distance(Km)")
plt.ylabel("Frequency")
plt.xlim(-10, 200)
plt.xticks(range(0,200,5))

sns.plot = sns.kdeplot(df[df.dist_travel_km<600].dist_travel_km, shade=True)

# select only the target variable 'amount' and store it in dataframe 'y'

y = pd.DataFrame(df['fare_amount'])

# use 'drop()' to remove the variable 'amount' from df_taxi

# 'axis = 1' drops the corresponding column(s)

x = df.drop('fare_amount',axis = 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)

# check the dimensions of the train & test subset for 
# print dimension of predictors train set

print("The shape of X_train is:",x_train.shape)

# print dimension of predictors test set

print("The shape of X_test is:",x_test.shape)

# print dimension of target train set

print("The shape of y_train is:",y_train.shape)

# print dimension of target test set

print("The shape of y_test is:",y_test.shape)

# build a full model using OLS()

# consider the log of sales price as the target variable

# use fit() to fit the model on train data

linreg_full = sm.OLS(y_train, x_train).fit()

# print the summary output

print(linreg_full.summary())

linreg_full_predictions = linreg_full.predict(x_test)
linreg_full_predictions

actual_fare = y_test['fare_amount']

actual_fare

linreg_full_rmse = (rmse(actual_fare, linreg_full_predictions))

# calculate rmse using rmse()

linreg_full_rmse = rmse(actual_fare,linreg_full_predictions )

# calculate R-squared using rsquared

linreg_full_rsquared = linreg_full.rsquared

# calculate Adjusted R-Squared using rsquared_adj

linreg_full_rsquared_adj = linreg_full.rsquared_adj 


# create the result table for all accuracy scores

# accuracy measures considered for model comparision are RMSE, R-squared value and Adjusted R-squared value

# create a list of column names

cols = ['Model', 'RMSE', 'R-Squared', 'Adj. R-Squared']

# create a empty dataframe of the colums

# columns: specifies the columns to be selected

result_tabulation = pd.DataFrame(columns = cols)


# compile the required information
linreg_full_metrics = pd.Series({'Model': "Linreg full model ",
                     'RMSE':linreg_full_rmse,
                     'R-Squared': linreg_full_rsquared,
                     'Adj. R-Squared': linreg_full_rsquared_adj     
                   })

# append our result table using append()
# ignore_index=True: does not use the index labels
# python can only append a Series if ignore_index=True or if the Series has a name

#result_tabulation = result_tabulation.append(linreg_full_metrics, ignore_index = True)
result_tabulation = pd.concat([result_tabulation, linreg_full_metrics], ignore_index=True)
# print the result table
result_tabulation

# importing library min max scaler to scale data
from sklearn.preprocessing import MinMaxScaler
#import library for implement Linear Regression
from sklearn.linear_model import SGDRegressor 
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.preprocessing import StandardScaler

scaler= StandardScaler()
scaler.fit(x)
x1= scaler.fit_transform(x)
x1
y1=y

# split data into train data and test data 

# what proportion of data should be included in test data is passed using 'test_size'

# set 'random_state' to get the same data each time the code is executed 

x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, test_size = 0.2, random_state = 1)

# check the dimensions of the train & test subset for 

# print dimension of predictors train set

print("The shape of X1_train is:",x1_train.shape)

# print dimension of predictors test set

print("The shape of X1_test is:",x1_test.shape)

# print dimension of target train set

print("The shape of y1_train is:",y1_train.shape)

# print dimension of target test set

print("The shape of y1_test is:",y1_test.shape)


from sklearn.linear_model import SGDRegressor
SGD_MODEL = SGDRegressor()
SGD_MODEL.fit(x_train, y_train)

# build the model

SGD_model = SGDRegressor(loss="squared_error",alpha = 0.0001, max_iter = 1000) #Instantiation

# fit the model

SGD_model.fit(x1_train, y1_train) #Training

# predict the values
y1_pred_SGD  = SGD_model.predict(x1_test)
y1_pred_SGD


r_squared_SGD = SGD_model.score(x1_train,y1_train)

# Number of observation or sample size

n = 159999 

# No of independent variables

p = 11

#Compute Adj-R-Squared

Adj_r_squared_SGD = 1 - (1-r_squared_SGD)*(n-1)/(n-p-1)

# Compute RMSE

rmse_SGD = sqrt(mean_squared_error(y1_test, y1_pred_SGD))


# compile the required information

linreg_full_metrics = pd.Series({'Model': "Linear regression with SGD",
                     'RMSE':rmse_SGD,
                     'R-Squared': r_squared_SGD,
                     'Adj. R-Squared': Adj_r_squared_SGD     
                   })

# append our result table using append()
# ignore_index=True: does not use the index labels
# python can only append a Series if ignore_index=True or if the Series has a name
#result_tabulation = result_tabulation.append(linreg_full_metrics, ignore_index = True)
result_tabulation = pd.concat([result_tabulation, linreg_full_metrics], ignore_index=True)

# print the result table
result_tabulation


from sklearn import tree
from sklearn.tree import export_graphviz
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

# instantiate the 'DecisionTreeRegressor' object using 'mse' criterion
# pass the 'random_state' to obtain the same samples for each time you run the code
decision_tree = DecisionTreeRegressor(criterion = 'squared_error', random_state = 10) #Max depth D.Tree gets formed

# fit the model using fit() on train data
decision_tree_model = decision_tree.fit(x_train, y_train) #fit() method is defined inside the class 'DecisionTreeClassifier'


y_pred_DT=decision_tree_model.predict(x_test)
y_pred_DT
y_test


r_squared_DT=decision_tree_model.score(x_test,y_test)
# Number of observation or sample size
n = 159999 

# No of independent variables
p = 11

#Compute Adj-R-Squared
Adj_r_squared_DT = 1 - (1-r_squared_DT)*(n-1)/(n-p-1)
Adj_r_squared_DT

# Compute RMSE
rmse_DT = sqrt(mean_squared_error(y_test, y_pred_DT))

# compile the required information
linreg_full_metrics = pd.Series({'Model': "Decision Tree Model ",
                     'RMSE':rmse_DT,
                     'R-Squared': r_squared_DT,
                     'Adj. R-Squared': Adj_r_squared_DT     
                   })

# append our result table using append()
# ignore_index=True: does not use the index labels
# python can only append a Series if ignore_index=True or if the Series has a name
result_tabulation = pd.concat([result_tabulation, linreg_full_metrics], ignore_index=True)

# print the result table
result_tabulation

# instantiate the 'DecisionTreeRegressor' object
# max_depth: maximum depth of the tree 
# max_leaf_nodes: maximum number of leaf nodes in the tree
# pass the 'random_state' to obtain the same samples for each time you run the code
prune = DecisionTreeRegressor(max_depth = 10, max_leaf_nodes = 32 , random_state = 10)

# fit the model using fit() on train data
decision_tree_prune = prune.fit(x_train, y_train)

y_pred_DT_prune=decision_tree_prune.predict(x_test)
y_pred_DT_prune

r_squared_DT_prune=decision_tree_prune.score(x_test,y_test)
# Number of observation or sample size
n = 159999  

# No of independent variables
p = 11

#Compute Adj-R-Squared
Adj_r_squared_DT_prune = 1 - (1-r_squared_DT_prune)*(n-1)/(n-p-1)
Adj_r_squared_DT_prune
# Compute RMSE
rmse_DT_prune = sqrt(mean_squared_error(y_test, y_pred_DT_prune))

# compile the required information
linreg_full_metrics = pd.Series({'Model': "Decision Tree Model after pruning ",
                     'RMSE':rmse_DT_prune,
                     'R-Squared': r_squared_DT_prune,
                     'Adj. R-Squared': Adj_r_squared_DT_prune     
                   })

# append our result table using append()
# ignore_index=True: does not use the index labels
# python can only append a Series if ignore_index=True or if the Series has a name
result_tabulation = pd.concat([result_tabulation, linreg_full_metrics], ignore_index=True)

# print the result table
result_tabulation

# import library for random forest regressor
from sklearn.ensemble import RandomForestRegressor

#intantiate the regressor
rf_reg = RandomForestRegressor(n_estimators=100, random_state=10)

# fit the regressor with training dataset
rf_reg.fit(x_train, y_train)

# predict the values on test dataset using predict()
y_pred_RF = rf_reg.predict(x_test)
y_pred_RF

r_squared_RF = rf_reg.score(x_test,y_test)
# Number of observation or sample size
n = 159999  

# No of independent variables
p = 11

#Compute Adj-R-Squared
Adj_r_squared_RF = 1 - (1-r_squared_RF)*(n-1)/(n-p-1)
Adj_r_squared_RF
# Compute RMSE
rmse_RF = sqrt(mean_squared_error(y_test, y_pred_RF))

# Calculate MAE
rf_reg_MAE = metrics.mean_absolute_error(y_test, y_pred_RF)
print('Mean Absolute Error (MAE):', rf_reg_MAE)

# Calculate MSE
rf_reg_MSE = metrics.mean_squared_error(y_test, y_pred_RF)
print('Mean Squared Error (MSE):', rf_reg_MSE)

# Calculate RMSE
rf_reg_RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pred_RF))
print('Root Mean Squared Error (RMSE):', rf_reg_RMSE)

# compile the required information
linreg_full_metrics = pd.Series({'Model': "Random Forest ",
                     'RMSE':rf_reg_RMSE,
                     'R-Squared': r_squared_RF,
                     'Adj. R-Squared': Adj_r_squared_RF     
                   })

# append our result table using append()
# ignore_index=True: does not use the index labels
# python can only append a Series if ignore_index=True or if the Series has a name
result_tabulation = pd.concat([result_tabulation, linreg_full_metrics], ignore_index=True)

# print the result table
result_tabulation

model = Sequential()
model.add(Dense(64, input_dim=4, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))

from tensorflow.keras.metrics import cosine
# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=[cosine])

# Train the model
model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, verbose=1)

# Evaluate the model
mse = model.evaluate(X_test_scaled, y_test, verbose=0)
print('Mean Squared Error:', mse)


# Make predictions
new_data = pd.DataFrame([[5.6, 30, 4, 17]], columns=['distance', 'duration', 'weekday', 'hour'])
new_data_scaled = scaler.transform(new_data)
prediction = model.predict(new_data_scaled)
print('Predicted fare amount:', prediction[0][0])



