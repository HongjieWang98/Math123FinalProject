# -*- coding: utf-8 -*-
"""
Created on Tue May 11 17:23:44 2021

@author: 11231
"""
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data = pd.read_csv('wdbc.csv')
# Encode diagnosis label
data['diagnonis'] = data['diagnosis'].map({'M':1,'B':0})
print(data['diagnosis'].value_counts())

# Drop unnecessary columns
cols_drop = ['id']
data = data.drop(cols_drop, axis=1)
# Encode diagnosis label
data['diagnonis'] = data['diagnosis'].map({'M':1,'B':0})
# Featureset creation
X = data.drop('diagnosis', axis=1).values
X = StandardScaler().fit_transform(X) #Standarized data

y = data.diagnosis                          # M or B 
list = ['id','diagnosis']
from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
y= labelencoder_Y.fit_transform(y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=100, test_size=0.3)
print(X_train.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.40, stratify=y, random_state = 17)
from sklearn.preprocessing import StandardScaler, RobustScaler
sc = RobustScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


def models(X_train,y_train):
    
  
  #Using Logistic Regression 
    from sklearn.linear_model import LogisticRegression
    log = LogisticRegression(random_state = 0)
    log.fit(X_train, y_train)

  #Using SVC rbf
    from sklearn.svm import SVC
    svc_rbf = SVC(kernel = 'rbf', random_state = 0)
    svc_rbf.fit(X_train, y_train)
    
  
  #print model accuracy on the training data.
    print('[0]Logistic Regression Training Accuracy:', log.score(X_train, y_train))
    #print('[1]K Nearest Neighbor Training Accuracy:', knn.score(X_train, y_train))
    print('[2]Support Vector Machine (RBF Classifier) Training Accuracy:', svc_rbf.score(X_train, y_train))
  
    return log, svc_rbf
model = models(X_train,y_train)

