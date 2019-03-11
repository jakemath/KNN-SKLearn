# **Problem Statement:** You are provided with a dataset from USA Forensic Science Service which has description of 6 types of glass; defined in terms of their oxide content (i.e. Na, Fe, K, etc). Your task is to use K-Nearest Neighbor (KNN) classifier to classify the glasses.
# The original dataset is available at [Glass Identification Data Set](https://archive.ics.uci.edu/ml/datasets/glass+identification). For detailed description on the attributes of the dataset, please refer to the original link of the dataset in the UCI ML repository.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.metrics import classification_report
from scipy.spatial import distance

train = pd.read_csv("trainKNN.txt",header=None)
train.columns=["ID", "RI", "Na", "Mg","Al","Si","K","Ca","Ba","Fe", "Type of glass"]
train = train.drop(["ID"],axis=1) # Drop ID since irrelevant to predictions

test = pd.read_csv("testKNN.txt",header=None)
test.columns=["ID", "RI", "Na", "Mg","Al","Si","K","Ca","Ba","Fe", "Type of glass"]
test = test.drop(['ID'],axis=1) # Drop ID since irrelevant to predictions

# Data exploration

train.head()

test.head()

train.describe()

test.describe()

def standardize (df): # Standardize data function
    for i in df.columns:
        if i != "Type of glass": # Don't standardize the categories
            df[i] = (df[i] - df[i].mean())/df[i].std()
    return df

train = standardize(train) # Standardize each dataset
test = standardize(test)

# k = 8 neighbors
euclid_model = KNeighborsClassifier(n_neighbors=8,metric = distance.sqeuclidean) # Square Euclidean distance model
manhattan_model = KNeighborsClassifier(n_neighbors=8,metric = distance.cityblock) # Manhattan distance model

x_train = train.drop(["Type of glass"],axis=1)
y_train = train["Type of glass"]

euclid_model.fit(x_train,y_train) # Train models
manhattan_model.fit(x_train,y_train)

x_test = test.drop(["Type of glass"],axis=1) 
y_test = test["Type of glass"]

manhattan_predictions = manhattan_model.predict(x_test) # Make predictions for each model
euclid_predictions = euclid_model.predict(x_test)

count1 = 0 # Compute accuracy of each model
count2 = 0
for i, j, k in zip(manhattan_predictions, euclid_predictions, y_test):
    if i == k:
        count1 += 1
    if j == k:
        count2 += 1

print("Manhattan Accuracy:",(count1/len(y_test))*100,"%")
print(classification_report(y_test,manhattan_predictions,target_names=['1','2','3','5','6','7'])) # Classification report for each model
print ("\n")
print("Square Euclidean Accuracy:",(count2/len(y_test))*100,"%")
print(classification_report(y_test,euclid_predictions,target_names=['1','2','3','5','6','7']))
