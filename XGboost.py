import pandas as pd
import numpy as np

df = pd.read_csv("dataset-fesa-new (1).csv")

!pip install xgboost

import xgboost as xgb

from sklearn.model_selection import train_test_split

dataframe = pd.read_csv('dataset-fesa-new (2).csv')
dataframe.info()
X = dataframe.copy()
y = X.pop('label')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)

from sklearn.tree import DecisionTreeClassifier

dTree_clf = DecisionTreeClassifier()

dTree_clf.fit(X_train,y_train)

y_pred2 = dTree_clf.predict(X_test)

from sklearn.metrics import accuracy_score, classification_report
# Use accuracy_score to get accuracy of the model
acc = accuracy_score(y_test, y_pred2)
print("Accuracy of Model::",accuracy_score(y_test,y_pred2))

xgb_classifier = xgb.XGBClassifier()

xgb_classifier.fit(X_train,y_train)

predictions = xgb_classifier.predict(X_test)

print("Accuracy of Model::",accuracy_score(y_test,predictions))
