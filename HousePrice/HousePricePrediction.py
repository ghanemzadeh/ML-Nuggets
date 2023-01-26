import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np

df = pd.read_csv("housePrice.csv")

# take a look at the dataset
print(df.head())
print(df.shape)


#Data cleansing for Area and Address features
print(df.dtypes)
df = df[pd.to_numeric(df["Area"], errors="coerce").notnull()]
df["Area"] = df["Area"].astype("int")
print(df.shape)
df=df.dropna()
df["Address"] = df["Address"].astype("string")
print(df.shape)
print(df.dtypes)

