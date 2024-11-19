#!/usr/bin/env python
# coding: utf-8

# In[204]:


import json
from matplotlib import pyplot as plt
from collections import defaultdict
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import numpy
import random
import gzip
import math


# In[206]:


import warnings
warnings.filterwarnings("ignore")


# In[208]:


def assertFloat(x): # Checks that an answer is a float
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N


# In[210]:


f = gzip.open("data/young_adult_10000.json.gz")
dataset = []
for l in f:
    dataset.append(json.loads(l))


# In[212]:


len(dataset)


# In[214]:


answers = {} # Put your answers to each question in this dictionary


# In[216]:


dataset[0]


# In[218]:


### Question 1


# In[220]:


def feature(datum):
    #Won't use
    return


# In[222]:


#Extracting target variable
ratings_list = [review['rating'] for review in dataset]

#Extracting exclamation mark feature
mark_list = []
for review in dataset:
    if 'review_text' in review:
        mark_count = review['review_text'].count('!')
        mark_list.append(mark_count)
mark_df = pd.DataFrame(mark_list, columns = ['Marks'])
ratings_df = pd.DataFrame(ratings_list, columns = ['Rating'])


# In[224]:


X = mark_df
Y = ratings_df


# In[226]:


#Building model
model_1 = LinearRegression()
model_1.fit(X, Y)

#Grabbing theta values & MSE
theta0 = model_1.intercept_[0]
theta1 = (model_1.coef_[0])[0]
y_pred = model_1.predict(X)
mse = mean_squared_error(Y, y_pred)


# In[228]:


answers['Q1'] = [theta0, theta1, mse]


# In[230]:


assertFloatList(answers['Q1'], 3) # Check the format of your answer (three floats)


# In[232]:


### Question 2


# In[234]:


def feature(datum):
    #Won't use
    return


# In[236]:


#Extract feature (length) from JSON file
len_list = [len(review['review_text']) for review in dataset]

#Create dataframe for model (rather than making into matrix)
features_df = pd.DataFrame({'Length': len_list, 'Marks': mark_list})


# In[238]:


X = features_df
Y = ratings_df


# In[240]:


#Build out model
model_2 = LinearRegression()
model_2.fit(X, Y)

#Grabbing theta values & MSE 
#(includes [0/1] since .intercept_ & .coef_ return an array)

theta0 = model_2.intercept_[0]
theta1 = (model_2.coef_[0])[0]
theta2 = (model_2.coef_[0])[1]
y_pred = model_2.predict(X)
mse = mean_squared_error(Y, y_pred)


# In[242]:


answers['Q2'] = [theta0, theta1, theta2, mse]


# In[244]:


assertFloatList(answers['Q2'], 4)


# In[246]:


### Question 3


# In[248]:


def feature(datum, deg):
    # feature for a specific polynomial degree
    #Won't use
    return


# In[250]:


#Creating feature dataframes for models
mark_df_2 = mark_df.copy()
mark_df_2['Marks_2'] = mark_df['Marks'] ** 2

mark_df_3 = mark_df_2.copy()
mark_df_3['Marks_3'] = mark_df['Marks'] ** 3

mark_df_4 = mark_df_3.copy()
mark_df_4['Marks_4'] = mark_df['Marks'] ** 4

mark_df_5 = mark_df_4.copy()
mark_df_5['Marks_5'] = mark_df['Marks'] ** 5


# In[252]:


mses = []
model_3 = LinearRegression()

#Model Degree 1 (same as Model_1)
model_3.fit(mark_df, Y)
y_pred = model_3.predict(mark_df)
mses.append(mean_squared_error(Y, y_pred))

#Model Degree 2 
model_3.fit(mark_df_2, Y)
y_pred = model_3.predict(mark_df_2)
mses.append(mean_squared_error(Y, y_pred))

#Model Degree 3
model_3.fit(mark_df_3, Y)
y_pred = model_3.predict(mark_df_3)
mses.append(mean_squared_error(Y, y_pred))

#Model Degree 4
model_3.fit(mark_df_4, Y)
y_pred = model_3.predict(mark_df_4)
mses.append(mean_squared_error(Y, y_pred))

#Model Degree 5
model_3.fit(mark_df_5, Y)
y_pred = model_3.predict(mark_df_5)
mses.append(mean_squared_error(Y, y_pred))

mses


# In[254]:


answers['Q3'] = mses


# In[256]:


assertFloatList(answers['Q3'], 5)# List of length 5


# In[258]:


### Question 4


# In[260]:


model_4 = LinearRegression()

mark_df['Rating'] = ratings_df
mid = len(mark_df)//2

training = mark_df.iloc[:mid]
X_train = training[['Marks']]
Y_train = training[['Rating']]

testing = mark_df.iloc[mid:]
X_test = testing[['Marks']]
Y_test = testing[['Rating']]


# In[262]:


#Creating feature dataframes for models
training_df_2 = X_train.copy()
testing_df_2 = X_test.copy()

#Model2
training_df_2['Marks_2'] = training_df_2['Marks'] ** 2
testing_df_2['Marks_2'] = testing_df_2['Marks'] ** 2 

#Model3
training_df_3 = training_df_2.copy()
testing_df_3 = testing_df_2.copy()
training_df_3['Marks_3'] = training_df_3['Marks'] ** 3
testing_df_3['Marks_3'] = testing_df_3['Marks'] ** 3 

#Model4
training_df_4 = training_df_3.copy()
testing_df_4 = testing_df_3.copy()
training_df_4['Marks_4'] = training_df_4['Marks'] ** 4
testing_df_4['Marks_4'] = testing_df_4['Marks'] ** 4

#Model5
training_df_5 = training_df_4.copy()
testing_df_5 = testing_df_4.copy()
training_df_5['Marks_5'] = training_df_5['Marks'] ** 5
testing_df_5['Marks_5'] = testing_df_5['Marks'] ** 5


# In[264]:


model_4 = LinearRegression()
mses = []

#order 1
model_4.fit(X_train, Y_train)
y_pred = model_4.predict(X_test)
mses.append(mean_squared_error(Y_test, y_pred))

#order2
model_4.fit(training_df_2, Y_train)
y_pred = model_4.predict(testing_df_2)
mses.append(mean_squared_error(Y_test, y_pred))

#order3
model_4.fit(training_df_3, Y_train)
y_pred = model_4.predict(testing_df_3)
mses.append(mean_squared_error(Y_test, y_pred))

#order4
model_4.fit(training_df_4, Y_train)
y_pred = model_4.predict(testing_df_4)
mses.append(mean_squared_error(Y_test, y_pred))

#order5
model_4.fit(training_df_5, Y_train)
y_pred = model_4.predict(testing_df_5)
mses.append(mean_squared_error(Y_test, y_pred))


# In[266]:


answers['Q4'] = mses


# In[268]:


assertFloatList(answers['Q4'], 5)


# In[270]:


### Question 5


# In[272]:


#The best predictor in terms of the MAE is the median
theta0 = np.median(Y_train)
y_pred = np.full_like(Y_test, theta0)

mae = mean_absolute_error(Y_test, y_pred)


# In[274]:


answers['Q5'] = mae


# In[276]:


assertFloat(answers['Q5'])


# In[278]:


### Question 6


# In[280]:


f = open("data/beer_50000.json")
dataset = []
for l in f:
    if 'user/gender' in l: #only appends recoreds with gender specified
        dataset.append(eval(l))


# In[281]:


len(dataset)


# In[374]:


X = [[1, d['review/text'].count('!')] for d in dataset]
y = [d['user/gender'] == 'Female' for d in dataset]


# In[378]:


model_5 = LogisticRegression()
model_5.fit(X, y)
y_pred = model_5.predict(X) #binary vector of predictions 


# In[398]:


from sklearn.metrics import confusion_matrix

tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel() #ravel() lays matrix out into 1D array

fpr = fp / (fp + tn) if (fp + tn) != 0 else 0
fnr = fn / (fn + tp) if (fn + tp) != 0 else 0
ber = (fpr + fnr) / 2


# In[400]:


TP = tp
TN = tn
FP = fp
FN = fn
BER = ber


# In[402]:


answers['Q6'] = [TP, TN, FP, FN, BER]


# In[404]:


assertFloatList(answers['Q6'], 5)


# In[ ]:


### Question 7


# In[408]:


model_6 = LogisticRegression(class_weight = 'balanced')
model_6.fit(X, y)
y_pred = model_6.predict(X)


# In[410]:


tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel() #ravel() lays matrix out into 1D array

fpr = fp / (fp + tn) if (fp + tn) != 0 else 0
fnr = fn / (fn + tp) if (fn + tp) != 0 else 0
ber = (fpr + fnr) / 2


# In[412]:


TP = tp
TN = tn
FP = fp
FN = fn
BER = ber


# In[416]:


answers["Q7"] = [TP, TN, FP, FN, BER]


# In[418]:


assertFloatList(answers['Q7'], 5)


# In[ ]:


### Question 8


# In[430]:


k_values = [1, 10, 100, 1000, 10000]
precisionList = []


# In[426]:


#We may only have a fixed budget of results taht can be returned to a user and we might be
#interested in evaluating the precision and recall when our classifier returns only its K most
#confident predictions.

confidences = model_6.decision_function(X) #real vector of confidences 

sortedByConfidence = list(zip(confidences, y))
sortedByConfidence.sort(reverse = True)


# In[486]:


for index in range(0, len(k_values)):
    retrievedLabels = [x[1] for x in sortedByConfidence[:k_values[index]]]
    precisionK = sum( retrievedLabels ) / len( retrievedLabels )
    precisionList.append(precisionK)


# In[488]:


answers['Q8'] = precisionList


# In[490]:


assertFloatList(answers['Q8'], 5) #List of five floats


# In[492]:


f = open("answers_hw1.txt", 'w') # Write your answers to a file
f.write(str(answers) + '\n')
f.close()

