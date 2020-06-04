# data analysis library
import pandas as pd
# scientific computing library
import numpy as np
# library to Read/Write to xlsx file
import openpyxl as op

import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans


# initialize the scaling function
sc = StandardScaler()

# store xlsx file name into string variable
excel_file = 'DSZinc.xlsx'

# load the excel file into a book
dataBook = op.load_workbook(excel_file)

# activate the book as a worksheet
dataSheet = dataBook.active

# convert worksheet into a data frame including the column names
df = pd.DataFrame(dataSheet.values, columns=['Date','Cu(hg)','Pb(hg)','Zn(hg)','Ag(hg)','Delivered to Plant',
'Milled','% Cu','Cu Recovery','% Pb', 'Pb Recovery', '% Zn','Zn Recovery','tons Ag','Ag Recovery','Source',
'Cu in Cu','Pb in Cu','Zn in Cu','Ag in Cu','pH','Oil in feed','+150 microns','eH'])

# DATA CLEANING

# delete 2nd row because it contains column names
df = df.drop([0], axis=0)
# delete last row with averages
total_rows = len(df.index)
df = df.drop([total_rows], axis=0)


# delete the date column
df = df.drop(["Date", "Source"], axis=1)

# number of missing values Pandas can detect
#print df.isnull().sum().sum()

# replace all entries with 0 by None, 0 not detected by Pandas as a missing value
df = df.replace({0: None})

# delete rows which are fully empty
df = df.dropna(how='all')


# replace empty entries with zero
#df = df.dropna()
df = df.replace({None: 0})

# create target
selected_column = df[["Cu Recovery"]]
target = selected_column.copy()

#print df.shape

# create features, 3 columns, Cu(hg), %Cu, and Cu in cu
features = df.drop(['Pb(hg)','Zn(hg)','Ag(hg)','Delivered to Plant','Milled','% Pb',
                    'Pb Recovery','% Zn','Zn Recovery','tons Ag','Ag Recovery', 'Cu Recovery',
                    'Pb in Cu','Zn in Cu','Ag in Cu','Cu in Cu','Oil in feed','+150 microns','eH'], axis=1)


#print features.tail(n=10)
#print target.tail(n=10)

# now add this code to the next cell and run to split your training and testing data to the specified ratio
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.3)
# validation data
#features_train, features_validation, target_train, target_validation = train_test_split(features_train, target_train, test_size=0.2)

#print features_train
# building an artificial neural network model
model = Sequential()
# ML model, creating layers
model.add(Dense(16, input_dim=3, activation='relu'))
#model.add(Dense(16, activation='relu'))
model.add(Dropout(0.5))
#model.add(Dense(16, activation='relu'))
model.add(Dense(1))

#print features

# to scale the training set and the test set, add this code to the notebook cell and run it:
features_train = sc.fit_transform(features_train)
#features_test = sc.transform(features_test)
target_train = sc.fit_transform(target_train)
#target_test = sc.transform(target_test)

# apply a gradient descent to the neural network
model.compile(optimizer= "adam",loss = "binary_crossentropy",metrics = ["accuracy"])

# train model using to your the training data
model.fit(features_train, target_train, batch_size=10, epochs=10)


prediction_features = model.predict(features_test)
performance = model.evaluate(features_test, target_test)

print performance


#print features_train
#print features_train.shape
#print target_train
#print target_train.shape
#print features_test.shape
#print target_test.shape



# you can now start adding layers to your network. Run this code in your next cell
#classifier.add(Dense(12, kernel_initializer = "uniform",activation = "relu", input_dim=21))
#classifier.add(Dense(8, kernel_initializer = "uniform", activation = 'relu'))
#classifier.add(Dense(1, kernel_initializer = "uniform",activation = "sigmoid"))



#model.summary()


