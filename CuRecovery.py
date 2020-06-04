# Copper Recovery Prediction using KNN regression model
# Author: Willem Kapenda
# Date: 15 May 2020
# Version: Final


# import openpyxl library to read from xlsx file
import openpyxl as op
# data analysis library
import pandas as pd
# scientific computing library, arrays
import numpy as np

# import scikit-learn module used for the KNN regression model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor


# initialize the KNN regressor model
knn = KNeighborsRegressor()
# initialize the scaling function
sc = StandardScaler()

# DATA EXTRACTION

# store xlsx file name into string variable
excel_file = 'DSZinc.xlsx'
# load the excel file into a book
dataBook = op.load_workbook(excel_file)
# activate the book as a worksheet
dataSheet = dataBook.active

# convert datasheet into a dataframe including the column names
df = pd.DataFrame(dataSheet.values, columns=['Date','Cu(hg)','Pb(hg)','Zn(hg)','Ag(hg)','Delivered to Plant',
'Milled','% Cu','Cu Recovery','% Pb', 'Pb Recovery', '% Zn','Zn Recovery','tons Ag','Ag Recovery','Source',
'Cu in Cu','Pb in Cu','Zn in Cu','Ag in Cu','pH','Oil in feed','+150 microns','eH'])


# DATA CLEANING

# delete 1st row because it contains column names which are already defined
df = df.drop([0], axis=0)
# calculate the length of the dataframe and delete the last row containing averages
total_rows = len(df.index)
df = df.drop([total_rows], axis=0)

# delete the Date and Source columns
df = df.drop(["Date", "Source"], axis=1)

# replace all 0 entries with by None, because 0 is not detected by Pandas as a missing value
df = df.replace({0: None})

# delete rows which are fully empty, which contain only None value
df = df.dropna(how='all')

# replace empty entries with zero
df = df.replace({None: 0})

# create target dataframe
selected_column = df[["Cu Recovery"]]
target = selected_column.copy()

# create the features dataframe by removing columns deemed to be redundant
features = df.drop(['Cu Recovery','Oil in feed','eH'], axis=1)

# create array containing feature labels
features_labels = [['Cu(hg)','Pb(hg)','Zn(hg)','Ag(hg)','Delivered to Plant','Milled','% Cu','% Pb','Pb Recovery',
                    '% Zn','Zn Recovery','tons Ag','Ag Recovery','Cu in Cu','Pb in Cu','Zn in Cu','Ag in Cu',
                    'pH','+150 microns']]


# MODELLING AND TRAINING DATA

# scale the features dataframe using the standard scaler fit transform function
for x in features_labels:
    features[x] = sc.fit_transform(features[x])

# split feature datframe into training (80%) and testing (20%) data to the specified ratio
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.2, random_state=4)

# fit the knn model with the training data
knn.fit(features_train, target_train)
# predict the copper recoveries using the test data
predictions = knn.predict(features_test)

# ANALYSIS

# Model score, how often is the classifier correct?
score = knn.score(features_test, target_test)
print score*100

# The knn model has an accuracy of 82.07%
