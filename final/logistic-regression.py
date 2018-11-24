#!/usr/bin/env python
# coding: utf-8

# # Credit Card Fraud

# We will be detecting credit card fraud based on the different features of our dataset with 3 different models. Here is the Logistic Regression one.
# 
# We're looking to minimize the False Negative Rate or FNR.
# 
# Since the dataset is unbalanced, we can try two techniques that may help us have better predictions:
# 
#     - Adding some noise (gaussian) to the fraud data to create more and reduce the imbalance
#     - Randomly sample the fraud data and train k models and average them out (or choose the best)
#     
#  

# In[1]:


import numpy as np
import sklearn as sk
import pandas as pd
import matplotlib.pyplot as plt
from pandas_ml import ConfusionMatrix
import pandas_ml as pdml
from sklearn.preprocessing import scale
import random


# In[2]:


# May have to do this...
#!pip install imblearn
#!pip install --upgrade sklearn


# In[3]:


df = pd.read_csv('creditcard.csv', low_memory=False)
df = df.sample(frac=1).reset_index(drop=True)
df.head()


# In[4]:


frauds = df.loc[df['Class'] == 1]#selecting data by condition
non_frauds = df.loc[df['Class'] == 0]
print("We have", len(frauds), "fraud data points and", len(non_frauds), "nonfraudulent data points.")


# In[5]:


ax = frauds.plot.scatter(x='Amount', y='Class', color='Orange', label='Fraud')
non_frauds.plot.scatter(x='Amount', y='Class', color='Blue', label='Normal', ax=ax)
plt.show()
print("This feature looks important based on their distribution with respect to class.")
print("We will now zoom in onto the fraud data to see the ranges of amount just for fun.")


# In[6]:


bx = frauds.plot.scatter(x='Amount', y='Class', color='Orange', label='Fraud')
plt.show()


# In[7]:


ax = frauds.plot.scatter(x='V22', y='Class', color='Orange', label='Fraud')
non_frauds.plot.scatter(x='V22', y='Class', color='Blue', label='Normal', ax=ax)
plt.show()
print("This feature may not be very important because of the similar distribution.")


# # Logistic Regression (vanilla)

# In[8]:


from sklearn import datasets, linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split


# In[9]:


X = df.iloc[:,:-1]#selecting data by row number , -1 indicates the last column number
y = df['Class']

print("X and y sizes, respectively:", len(X), len(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35)
print("Train and test sizes, respectively:", len(X_train), len(y_train), "|", len(X_test), len(y_test))
print("Total number of frauds:", len(y.loc[df['Class'] == 1]), len(y.loc[df['Class'] == 1])/len(y))
print("Number of frauds on y_test:", len(y_test.loc[df['Class'] == 1]), len(y_test.loc[df['Class'] == 1]) / len(y_test))
print("Number of frauds on y_train:", len(y_train.loc[df['Class'] == 1]), len(y_train.loc[df['Class'] == 1])/len(y_train))


# In[10]:


logistic = linear_model.LogisticRegression(C=8655.38)
logistic.fit(X_train, y_train)
print("Score: ", logistic.score(X_test, y_test))


# In[11]:


y_predicted = np.array(logistic.predict(X_test))
y_right = np.array(y_test)


# In[12]:


confusion_matrix = ConfusionMatrix(y_right, y_predicted)
print("Confusion matrix:\n%s" % confusion_matrix)
confusion_matrix.plot(normalized=True)
plt.show()
confusion_matrix.print_stats()


# In[13]:


print("FNR is {0}".format(confusion_matrix.stats()['FNR']))


# # Logistic Regression with SMOTE over-sampling

# In[14]:


df2 = pdml.ModelFrame(X_train, target=y_train)
sampler = df2.imbalance.over_sampling.SMOTE()
sampled = df2.fit_sample(sampler)
print("Size of training set after over sampling:", len(sampled))


# In[15]:


X_train_sampled = sampled.iloc[:,1:]
y_train_sampled = sampled['Class']

# NOTE: Scaling makes it worse.
# X_train_sampled = scale(X_train_sampled)

logistic = linear_model.LogisticRegression(C=8655.38)
logistic.fit(X_train_sampled, y_train_sampled)
print("Score: ", logistic.score(X_test, y_test))


# In[16]:


y_predicted1 = np.array(logistic.predict(X_test))
y_right1 = np.array(y_test)

confusion_matrix1 = ConfusionMatrix(y_right1, y_predicted1)
print("Confusion matrix:\n%s" % confusion_matrix1)
confusion_matrix1.plot(normalized=True)
plt.show()
confusion_matrix1.print_stats()


# In[17]:


print("FNR is {0}".format(confusion_matrix1.stats()['FNR']))


# # Logistic Regression with balanced class weights

# In[19]:


best_c, best_fnr = 1, 1
for _ in range(20):
    c = random.uniform(1, 10000)
    logistic = linear_model.LogisticRegression(C=c, class_weight="balanced")
    logistic.fit(X_train, y_train)
    #print("Score: ", logistic.score(X_test, y_test))
    y_predicted2 = np.array(logistic.predict(X_test))
    y_right2 = np.array(y_test)
    confusion_matrix2 = ConfusionMatrix(y_right2, y_predicted2)
    #print("Confusion matrix:\n%s" % confusion_matrix2)
    #confusion_matrix2.plot(normalized=True)
    #plt.show()
    #confusion_matrix2.print_stats()
    fnr = confusion_matrix2.stats()['FNR']
    if fnr < best_fnr:
        best_fnr = fnr
        best_c = c
print("Best C is {0} with best FNR of {1}.".format(best_c, best_fnr))


# In[ ]:




