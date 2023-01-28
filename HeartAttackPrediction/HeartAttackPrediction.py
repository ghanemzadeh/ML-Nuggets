#Load libraries
import pandas as pd
import numpy as np


df = pd.read_csv("heart.csv")

#Take a look at the dataset
#print(df.head())
#print(df.shape)

X = df[["age", "sex", "cp", "trtbps", "chol", "fbs", "restecg", "thalachh", "exng", "oldpeak", "slp", "caa", "thall"]].values
#print(X[0:5])
y = df["output"]
#print(y[0:5])

#Spliting Date into Train and Test (80/20)
from sklearn.model_selection import train_test_split #Import train_test_split function

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

#Train
from sklearn.tree import DecisionTreeClassifier #Import Decision Tree Classifier

heartAttack = DecisionTreeClassifier(criterion="gini", max_depth = 8)

#Prediction
heartAttack.fit(X_train,y_train)

y_hat = heartAttack.predict(X_test)

#print (y_hat [0:5])
#print (y_test [0:5])

#Evaluation
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_test, y_hat))

#Visualization
import matplotlib.pyplot as plt
from sklearn import tree

featureNames = df.columns[0:13]

plt.figure(dpi=300)
tree.plot_tree(heartAttack, feature_names=featureNames, filled=True)
plt.show()

