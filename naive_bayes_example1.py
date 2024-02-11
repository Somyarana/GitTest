# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 15:52:15 2024

@author: cdac
"""

weather=['Sunny','Sunny','Overcast','Rainy','Rainy','Rainy','Overcast','Sunny','Sunny',
'Rainy','Sunny','Overcast','Overcast','Rainy']
temp=['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild']
play=['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']


from sklearn.preprocessing import LabelEncoder


le=LabelEncoder()
we=le.fit_transform(weather)
te=le.fit_transform(temp)
play=le.fit_transform(play)

we
te
play

x=list(zip(we,te))
x
from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
#train the model using the traing data
model.fit(x,play)

##predict the class label for the test data
predicted=model.predict([[2,1]])
predicted
predicted=model.predict([[0,2]])
predicted
