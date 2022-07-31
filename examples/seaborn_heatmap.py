# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 12:17:30 2022

@author: TÚLIO
"""
from pprint import pprint

from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import fetch_openml, load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data, target = fetch_openml(data_id=41470,return_X_y = True, as_frame=True)

iris_data, iris_target = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.3, random_state=2022
)

encoded_y_test = np.empty(y_test.shape)
encoded_y_train = np.empty(y_train.shape)

le = LabelEncoder()
for i in range(y_train.shape[1]):
    
    encoded_y_train[:,i] = le.fit_transform(y_train.values[:, i])
    encoded_y_test[:, i] = le.transform(y_test.values[:, i])

rf = RandomForestClassifier(max_depth=30, n_jobs=3, random_state=2022)

rf.fit(X_train, encoded_y_train)

y_pred = rf.predict(X_test)

pprint(np.unique(target))
cm = multilabel_confusion_matrix(encoded_y_test, y_pred)
fig, axes = plt.subplots(1,7,figsize=(18,9))
for i, ax in enumerate(axes):
    sns.heatmap(
        cm[i],
        ax=ax,
        cmap='viridis', 
        annot=True, 
        cbar=False, 
        fmt="d", 
    )
    ax.set_aspect(1)
    
axes[0].set_ylabel('True labels')
axes[3].set_xlabel('Predicted labels')
plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)
plt.show()
    