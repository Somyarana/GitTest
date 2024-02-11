# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 16:17:35 2024

@author: cdac
"""

from sklearn import datasets
dataset=datasets.load_wine()
dataset['data']
dataset['target']
len(dataset['data'])
len(dataset['target'])
dataset.feature_names
dataset.target_names
#perform All EDA

#split the data into train and test data
from sklearn.model_selection import train_test_split
#x=dataset data(0 to 12)
#y=dataset target(1 dim array)
x_train,x_test,y_train,y_test=train_test_split(dataset.data,dataset.target,
                                               test_size=0.3,random_state=100)

#import the gaussian nb model
from sklearn.naive_bayes import GaussianNB
model=GaussianNB()

#train the model using the training data
model.fit(x_train,y_train)

# predict the class for the test data
y_pred=model.predict(x_test)

#evaluate the model

from sklearn import metrics

#model accuracy

metrics.accuracy_score(y_test, y_pred)





