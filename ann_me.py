#We are going to build a neural network for a binary classification model(DV), i.e. whether a person will leave the bank or not.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values
#before splitting the data into test set and training set we need to take care of categorical variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder() #if two separate categorical variables are present then, treat them separately X_1, X_2 
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
#we now need to create dummy variables for categorical variable(it also converts X from object to float64)
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()#After this we remove 1st column(0th index) to prevent ourself from falling in dummy variable trap (why?)
X = X[:, 1:]
#splitting the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)
#Applying feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
'''Data preprocessing completed'''
#building ANN
import keras
from keras.models import Sequential
from keras.layers import Dense
#using 1st initialization method i.e. defining sequence of layers
classifier = Sequential() #classifier is the future nn we will be building
#adding 1st i/p layer and hidden layers
#choosing rectifier function for hidden layer and sigmoid function for o/p layer as this is a classification model
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
#Adding 2nd hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
#Adding the o/p layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
#compiling the whole neural network, i.e. applying stochastic gradient descent on whole neural network.
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
'''optimizer -> the algo we want to use find optimal set of weights,
adam is a type of stochastics grad. algo, adam is based on loss function that we need to optimize.
loss is same as logrithmic function of logistic regression and for two catogorical variable(yes/no) it is given as
binary_crossentropy'''
#fitting the ann to training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)
'''batch_size is the no. of observation after which we update the weights, nb_epoch is how many iterations we go 
on neural network'''

'''Accuracy = 86%'''
#now time to make prediction on test set
y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5) #converts digital value to boolean value i.e. true or false
#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
'''we have got this accuracy when we have not done any parameter 
tuning'''



