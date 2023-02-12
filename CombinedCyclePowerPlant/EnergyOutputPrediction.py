import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np

# Reading Dataset
df = pd.read_csv("CCPP_data.csv")

# Take a look at the dataset
#print(df.head())
#print(df.shape)

# Data cleansing for Area and Address features
#print(df.dtypes)
#print(df.shape)
#df=df.dropna()
#print(df.shape)
#print(df.dtypes)
# Data was already clean so I just commented the code



# I used this part to take a look at scatter plot for each feature, It seems it's Linear Regression Like, But Multipe.
plt.scatter(df["AT"], df["PE"],  color='blue')
plt.xlabel("Temperature")
plt.ylabel("Net hourly electrical energy output")
#plt.show()


X = df[["AT", "V", "AP", "RH"]].values
y = df[["PE"]].values
#print(X[0:5])
#print(y[0:5])


# Spliting Date into Train, Test and validation (70/20/10)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=123)
X_test, X_valid, y_test, y_valid = train_test_split( X_test, y_test, test_size=0.33, random_state=123)
print ("Train set:", X_train.shape, y_train.shape)
print ("Test set:", X_test.shape, y_test.shape)
print ("Validation set:", X_valid.shape, y_valid.shape)

# Train
from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit (X_train, y_train)

# The coefficients
print ("Coefficients: ", regr.coef_)

# Validation & Validation Metrics
y_hat_Valid= regr.predict(X_valid)
print(f"Metrics for validation data")
print(f"(Validation) Mean Absolute Error: {np.mean(np.absolute(y_hat_Valid - y_valid)):.2f}")
print(f"(Validation) Residual sum of squares: {np.mean((y_hat_Valid - y_valid) ** 2):.2f}")
print(f"(Validation) Variance score: {regr.score(X_valid, y_valid):.2f}")

# Prediction on test data
y_hat= regr.predict(X_test)

# Metrics
print(f"Metrics for test data")
print(f"Mean Absolute Error: {np.mean(np.absolute(y_hat - y_test)):.2f}")
print(f"Residual sum of squares: {np.mean((y_hat - y_test) ** 2):.2f}")
print(f"Variance score: {regr.score(X_test, y_test):.2f}")

# Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
regrRF = RandomForestRegressor(n_estimators=1000, max_depth=6, random_state=123)
regrRF.fit(X_train, y_train)

# Prediction on test data
y_hat= regrRF.predict(X_test)

# Metrics
print(f"Metrics for Random Forest test data with n_estimators=1000, max_depth=6")
print(f"Mean Absolute Error: {np.mean(np.absolute(y_hat - y_test)):.2f}")
print(f"Residual sum of squares: {np.mean((y_hat - y_test) ** 2):.2f}")
print(f"Variance score: {regrRF.score(X_test, y_test):.2f}")

# Changing RandomForestRegressor parameters
regrRF = RandomForestRegressor(n_estimators=300, max_depth=2, random_state=123)
regrRF.fit(X_train, y_train)

# Prediction on test data
y_hat= regrRF.predict(X_test)

# Metrics
print(f"Metrics for Random Forest test data with n_estimators=300, max_depth=2")
print(f"Mean Absolute Error: {np.mean(np.absolute(y_hat - y_test)):.2f}")
print(f"Residual sum of squares: {np.mean((y_hat - y_test) ** 2):.2f}")
print(f"Variance score: {regrRF.score(X_test, y_test):.2f}")