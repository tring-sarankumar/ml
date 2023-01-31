Prediction of a district's median housing prices 

LOAD DATASET:

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

train_df = pd.read_csv("housing.csv")

train_df.head(10)

train_df.tail()

Exploratory Data Analysis (EDA):

list(train_df.columns)

train_df.describe()

train_df.info()

Feature Engineering:

sns.heatmap(train_df.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #data.corr()-->correlation matrix

fig=plt.gcf()

fig.set_size_inches(20,12)

plt.show()
