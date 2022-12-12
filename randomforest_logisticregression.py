import pandas
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
a= "dataset-fesa-new (1).csv"
dataframe = pandas.read_csv(a)
dataset = dataframe.values
X = dataframe.copy()
y = X.pop('label')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 101)
classes = ['brown', 'dark', 'light', 'normal', 'tan', 'verydark']
n_classes = len(classes)

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

model1=RandomForestClassifier(max_features=3)
model1.fit(X_train,y_train)
y1_pred=model1.predict(X_test)

model2=LogisticRegression(max_iter=3000)
model2.fit(X_train,y_train)
y2_pred=model2.predict(X_test)

from sklearn import metrics
models=['Random Forest Classifier','Logistic Regression']
accuracy=[y1_pred,y2_pred]

for i,j in zip(models,accuracy):
  print("Accuracy for {} : {}".format(i,metrics.accuracy_score(y_test,j)))
