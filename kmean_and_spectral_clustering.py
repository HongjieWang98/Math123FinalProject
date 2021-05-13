# -*- coding: utf-8 -*-
"""
Created on Tue May 11 17:23:44 2021

@author: 11231
"""
import numpy as np # linear algebra
import pandas as pd # data processing
#from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv('wdbc.csv')
# Encode diagnosis label
data['diagnonis'] = data['diagnosis'].map({'M':1,'B':0})
print(data['diagnosis'].value_counts())
sns.countplot(data['diagnosis'],label="Count")

# Drop unnecessary columns
cols_drop = ['id']
data = data.drop(cols_drop, axis=1)
# Encode diagnosis label
data['diagnonis'] = data['diagnosis'].map({'M':1,'B':0})
# Featureset creation
X = data.drop('diagnosis', axis=1).values
X = StandardScaler().fit_transform(X) #Standarized data


#------------------------------KMeans-Clustering------------------------------------------

from sklearn.cluster import KMeans
km = KMeans(n_clusters=2, init="k-means++", n_init=10)
km_pred = km.fit_predict(X)
#labels = km.labels_

# Scatter plots
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

ax1.scatter(X[:,1], X[:,4], c=data["diagnosis"], cmap="jet", edgecolor="None", alpha=0.35)
ax1.set_title("Actual clusters")

ax2.scatter(X[:,1], X[:,4], c=km_pred, cmap="jet", edgecolor="None", alpha=0.35)
ax2.set_title("KMeans clustering plot")
from sklearn import metrics
accuracy = metrics.adjusted_rand_score(data["diagnosis"], km_pred)
print ("The accuracy of KMeans is: ",accuracy)


#------------------------------SpectralClustering------------------------------------------

from sklearn.cluster import SpectralClustering

for index, gamma in enumerate((0.001,0.01,0.1,0.5,1)):
    sc = SpectralClustering(n_clusters=2, gamma= gamma, assign_labels="discretize")
    sc_pred = sc.fit_predict(X)
    accuracy2 = metrics.adjusted_rand_score(data["diagnosis"], sc_pred)
    print ("The accuracy of Spectral Clustering is: ",accuracy2, "gamma=",gamma)
# Scatter plots
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
sc = SpectralClustering(n_clusters=2, gamma= 0.01, assign_labels="kmeans")
sc_pred = sc.fit_predict(X)
ax1.scatter(X[:,1], X[:,4], c=data["diagnosis"], cmap="jet", edgecolor="None", alpha=0.35)
ax1.set_title("Actual clusters")

ax2.scatter(X[:,1], X[:,4], c=sc_pred, cmap="jet", edgecolor="None", alpha=0.35)
ax2.set_title("Spectral clustering plot,gamma=0.01")
accuracy2 = metrics.adjusted_rand_score(data["diagnosis"], sc_pred)
print ("The best accuracy of Spectral Clustering is when gamma=0.01",accuracy2)
