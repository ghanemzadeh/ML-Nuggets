# Importing necessary Libraries
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import seaborn as sns


# Reading Dataset
df = pd.read_csv("Customer.csv")

# Take a look at the dataset
#print(df.head(10))
#print()
#print(df.shape)

# Data cleansing for Area and Address features
#print(df.dtypes)
df = df.dropna()

# Convert "Gender" values to numerical values
#print(df["Gender"].value_counts())
df["Gender"] = df["Gender"].astype("category")
df["Gender"] = df["Gender"].cat.codes
# Data was already clean so I just commented the code

X = df[["Annual Income (k$)", "Spending Score (1-100)"]].values
#print(X[0:5])

# Normalizing Data
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
print(X[0:5])



######################
# K-Means Clustering #

km = KMeans(n_clusters = 5, init = "random", n_init = 10)
y_km = km.fit_predict(X)
#print(y_km)

print(km.cluster_centers_)

# Plot
df_scale = pd.DataFrame(X, columns = ["Annual Income (k$)", "Spending Score (1-100)"]);
df2 = df_scale
df_scale.head(5)
df_scale["Clusters"] = km.labels_
#print(df_scale[0:5])
sns.scatterplot(x = "Spending Score (1-100)", y = "Annual Income (k$)", hue = 'Clusters', data = df_scale, palette = "viridis")
plt.show() 

#########################################
# Agglomerative Hierarchical Clustering #

from sklearn.cluster import AgglomerativeClustering

ac = AgglomerativeClustering(n_clusters = 5, linkage = "complete", metric = "euclidean")
y_ac = ac.fit_predict(X)
#print(y_ac)

# Plot
plt.figure(figsize =(5, 5))
plt.scatter(df2["Spending Score (1-100)"], df2["Annual Income (k$)"], c = ac.fit_predict(X), cmap = "rainbow")
plt.show()

#########################################
# DBSCAN Clustering #

from sklearn.cluster import DBSCAN

db = DBSCAN(eps = 0.1, min_samples = 8, metric = "euclidean")
y_db = db.fit_predict(X)

# Plot
c = db.labels_
#print("c=: ", c)
plt.scatter(X[:, 0], 
X[:, 1], 
c = db.labels_)
plt.xlabel("Spending Score (1-100)")
plt.ylabel("Annual Income (k$)")
plt.show()


