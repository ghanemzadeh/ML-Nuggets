import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np

#Reading Dataset
df = pd.read_csv("housePrice.csv")

#Take a look at the dataset
#print(df.head())
#print(df.shape)


#Data cleansing for Area and Address features
#print(df.dtypes)
df = df[pd.to_numeric(df["Area"], errors="coerce").notnull()]
df["Area"] = df["Area"].astype("int")
#print(df.shape)
df=df.dropna()
df["Address"] = df["Address"].astype("string")
#print(df.shape)
#print(df.dtypes)

#Convert "Address" values to numerical values
#print(df["Address"].value_counts())
addressValues = df["Address"].values.unique()
#print(addressValues)
"""
plt.scatter(df["Address"], df["Price(USD)"],  color='blue')
plt.xlabel("Address")
plt.ylabel("Price(USD)")
plt.show()
"""

X = df[["Area", "Room", "Parking", "Warehouse", "Elevator","Address",]].values
y = df[["Price(USD)"]].values
#print(X[0:5])
#print(y[0:5])

from sklearn import preprocessing
le_Parking = preprocessing.LabelEncoder()
le_Parking.fit(['False','True'])
X[:,2] = le_Parking.transform(X[:,2]) 

le_Warehouse = preprocessing.LabelEncoder()
le_Warehouse.fit(['False','True'])
X[:,3] = le_Warehouse.transform(X[:,3]) 

le_Elevator = preprocessing.LabelEncoder()
le_Elevator.fit(['False','True'])
X[:,4] = le_Elevator.transform(X[:,4]) 

le_Address = preprocessing.LabelEncoder()
le_Address.fit(addressValues)
X[:,5] = le_Address.transform(X[:,5]) 

print(X[0:5])

#Spliting Date into Train and Test (80/20)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=123)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

#Train
from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit (X_train, y_train)
# The coefficients
print ('Coefficients: ', regr.coef_)

#Prediction
y_hat= regr.predict(X_test)

print("Residual sum of squares: %.2f"
      % np.mean((y_hat - y_test) ** 2))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(X_test, y_test))

