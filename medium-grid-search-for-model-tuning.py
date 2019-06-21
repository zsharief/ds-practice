# https://towardsdatascience.com/grid-search-for-model-tuning-3319b259367e

import pandas as pd
import numpy as np
import os

data_path = "D:\Repos\ds-practice\data\BreastCancerWisconsin(Original)DataSet"
os.chdir(data_path)

## import dataset ##
data = pd.read_csv("breast-cancer-wisconsin.data", header = None)
## set column names
data.columns = ['Sample Code Number', 'Clump Thickness', 'Uniformity of Cell Size', 
                'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size', 
                'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']
data.head(10)


## clean the data ##
data = data.drop(['Sample Code Number'], axis = 1) # axis = 1 for column/ 0 for row
data = data[data['Bare Nuclei'] != '?'] # remove rows with missing data
data['Class'] = np.where(data['Class'] == 2, 0, 1) # change the class representation
data['Class'].value_counts() # class distribution


## dummy classifier ##
## predicts most frequent class; to be used as baseline
X = data.drop(['Class'], axis = 1)
y = data['Class']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)

from sklearn.dummy import DummyClassifier
clf = DummyClassifier(strategy = 'most_frequent').fit(X_train, y_train)
y_pred = clf.predict(X_test)

y_test.value_counts()
pd.Series(y_pred).value_counts()


## Calculate evaluation metrics ##
## Model evaluation metrics
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
accuracy_score(y_test, y_pred)
precision_score(y_test, y_pred)
recall_score(y_test, y_pred)
f1_score(y_test, y_pred)

## Dummy classifier confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)


## Logistic regression ##
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression().fit(X_train, y_train)
y_pred = clf.predict(X_test)

## model evaluation metrics
accuracy_score(y_test, y_pred)
precision_score(y_test, y_pred)
recall_score(y_test, y_pred)
f1_score(y_test, y_pred)
confusion_matrix(y_test, y_pred)

## We try to minimize false negatives as a malignant case
## not detected can be very dangerous. So, we maximize recall

## Grid search to maximize recall ##
from sklearn.model_selection import GridSearchCV
clf = LogisticRegression()
grid_values = {'penalty': ['l1', 'l2'], 'C': [0.001, 0.009, 0.01, 0.09, 1, 5, 10, 25]}
grid_clf_acc = GridSearchCV(clf, param_grid = grid_values, scoring = 'recall')
grid_clf_acc.fit(X_train, y_train)
## The hyperparameters tuned are:
## 1)penalty - l1 or l2
## 2)C - inverse of regularization rate lambda; smaller the value of C, more the regularization

y_pred_acc = grid_clf_acc.predict(X_test)
print(accuracy_score(y_test, y_pred_acc))
print(precision_score(y_test, y_pred_acc))
print(recall_score(y_test, y_pred_acc))
print(f1_score(y_test, y_pred_acc))
print(confusion_matrix(y_test, y_pred_acc))
## We see that the recall increases but the precision drops
## We can further tune the model to strike a balance between precision and 
## recall by using ‘f1’ score as the evaluation metric

## Grid search builds a model for every combination of hyperparameters specified 
## and evaluates each model. A more efficient technique for hyperparameter tuning
## is the Randomized search — where random combinations of the hyperparameters 
## are used to find the best solution