# data analysis library
import pandas as pd
# scientific computing library
import numpy as np
# library to Read/Write to xlsx file
import openpyxl as op

import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
df = df.dropna()
#df = df.replace({None: 0})

# create target
selected_column = df[["Cu Recovery"]]
target = selected_column.copy()


# create features by deleting the Cu Recovery, Oil in feed, +150 microns and eH columns
features = df.drop(["Cu Recovery",'tons Ag','Ag Recovery','Oil in feed','+150 microns','eH'], axis=1)



kmeans = KMeans(n_clusters=18).fit(features)
centroids = kmeans.cluster_centers_
print(centroids)

plt.scatter(df['x'], df['y'], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
plt.show()



# now add this code to the next cell and run to split your training and testing data to the specified ratio
#features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.2)
# validation data
#features_train, features_validation, target_train, target_validation = train_test_split(features_train, target_train, test_size=0.2)

#print features_train

# ML model, creating layers
#model.add(Dense(16, input_dim=16, activation='relu'))
#model.add(Dense(16, activation='relu'))
#model.add(Dropout(0.5))
#model.add(Dense(16, activation='relu'))
#model.add(Dense(1, activation=None))

#print features

# to scale the training set and the test set, add this code to the notebook cell and run it:
#features_train = sc.fit_transform(features_train)
#features_test = sc.fit_transform(features_test)
#target_train = sc.fit_transform(target_train)
#target_test = sc.fit_transform(target_test)

# apply a gradient descent to the neural network
#model.compile(optimizer= "adam",loss = "binary_crossentropy",metrics = ["accuracy"])

# train model using to your the training data
#model.fit(features_train, target_train, batch_size=10, epochs=10)

#validation_data=(features_validation, target_validation))



#prediction_features = model.predict(features_test)
#performance = model.evaluate(features_test, target_test)

#print prediction_features


#print features_train
#print features_train.shape
#print target_train
#print target_train.shape
#print features_test.shape
#print target_test.shape
#print features_validation.shape
#print target_validation.shape


#print target.values
#print target.dtypes
#print df.isnull()
#print df.isnull().sum()
#print features.head(n=60)
#print features.loc[2]
#print target
#print (features["Pb(hg)"])
