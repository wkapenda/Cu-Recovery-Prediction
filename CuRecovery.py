# data analysis library
import pandas as pd
# scientific computing library
import numpy as np
# library to Read/Write to xlsx file
import openpyxl as op


# store xlsx file name into string variable
excel_file = 'DSZinc.xlsx'

# load the excel file into a book
dataBook = op.load_workbook(excel_file)

# activate the book as a worksheet
dataSheet = dataBook.active

# convert worksheet into a data frame
df = pd.DataFrame(dataSheet.values, columns=['Date','Cu(hg)','Pb(hg)','Zn(hg)','Ag(hg)','Delivered to Plant','Milled','% Cu','Cu Recovery','% Pb', 'Pb Recovery',
'% Zn','Zn Recovery','tons Ag','Ag Recovery','Source','Cu in Cu','Pb in Cu','Zn in Cu','Ag in Cu','pH','Oil in feed','+150 microns','eH'])

# delete 2nd row, row is duplicate
df = df.drop([0], axis=0)

# delete the Date and Sources columns. Redundant.
df = df.drop(["Date","Source"], axis=1)

# replace all entries with 0 with None
df = df.replace({0: None})

# delete all row which are empty
df = df.dropna(how = 'all')
#df= df.dropna()


#print df.isnull()


print df.isnull().sum()
print df.head(n=20)
#print df.loc[10]

#print(a1.value)
#print(a2.value)


#dataSet = pd.DataFrame(excel_file)
