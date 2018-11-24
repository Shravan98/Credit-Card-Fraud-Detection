#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, Flatten, Dropout, MaxPooling2D
from keras.optimizers import SGD
import tensorflow as tf

import pandas_ml as pdml
import imblearn


# In[ ]:


df = pd.read_csv('creditcard.csv', low_memory=False)
X = df.iloc[:,:-1]
y = df['Class']


# In[ ]:


df.head()


# In[ ]:


frauds = df.loc[df['Class'] == 1]
non_frauds = df.loc[df['Class'] == 0]
print("We have", len(frauds), "fraud data points and", len(non_frauds), "regular data points.")


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33) #93987 test points and 190820 training points


# In[ ]:


print("Size of training set: ", X_train.shape)


# # Simplest Neural Network (for testing)

# In[ ]:


model = Sequential()
model.add(Dense(30, input_dim=30, activation='relu'))     # kernel_initializer='normal'   //input layer has 30 neurons , hidden layer has 30 neurons , output layer has one neuron
model.add(Dense(1, activation='sigmoid'))                 # kernel_initializer='normal'
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


# In[ ]:


model.fit(X_train.as_matrix(), y_train, epochs=1)


# In[ ]:


print("Loss: ", model.evaluate(X_test.as_matrix(), y_test, verbose=0))


# In[ ]:


y_predicted = model.predict(X_test.as_matrix()).T[0].astype(int)


# In[ ]:


from pandas_ml import ConfusionMatrix
y_right = np.array(y_test)
confusion_matrix = ConfusionMatrix(y_right, y_predicted) #93987 test data points
print("Confusion matrix:\n%s" % confusion_matrix)
confusion_matrix.plot(normalized=True)
plt.show()


# In[ ]:


confusion_matrix.print_stats() #actually fraud but predicted as non fraud are 166 points.


# # Oversampling with gaussian noise (commented out)

# In[ ]:


# noise = np.random.normal(0,.1,30)

# # 0 is the mean of the normal distribution you are choosing from
# # 1 is the standard deviation of the normal distribution
# # 100 is the number of elements you get in array noise
# noise


# In[ ]:


# frauds.head()


# In[ ]:


# for i in range(300):
#     #frauds.iloc[i] += noise[i]
#     frauds.append(frauds.iloc[i % 30] + noise[i % 30])


# In[ ]:


# NEED TO ADD A (DIFFERENT BUT SIMILAR) RANDOM NOISE ARRAY TO EVERY ROW OF FRAUDS TABLE (WITHOUT CLASS)
# THEN ADD THIS TO ORIGINAL FRAUDS TABLE (MAKING MORE DATA POINTS)
# AND RE SPLIT DATA AND DO NEURAL NET
# ALSO TRY FORCING 50% OF FRAUDS INTO TRAINING SET AND 50% INTO TEST SET


# # Neural Network after Oversampling, Scaling, and PCA (10 components)

# In[ ]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

df2 = pdml.ModelFrame(X_train, target=y_train)
sampler = df2.imbalance.over_sampling.SMOTE()
oversampled = df2.fit_sample(sampler)
X2, y2 = oversampled.iloc[:,:-1], oversampled['Class']

data = scale(X2)
pca = PCA(n_components=10)
X2 = pca.fit_transform(data)
X2


# In[ ]:


model2 = Sequential()
model2.add(Dense(10, input_dim=10, activation='relu')) #10 input neurons and 10 neurons in the first hidden layer
model2.add(Dense(27, activation='relu')) # 27 neurons in second hidden layer
model2.add(Dense(20, activation='relu')) #20 neurons in third hidden layer
model2.add(Dense(15, activation='relu')) # 15 neurons in fourth hidden layer
model2.add(Dense(1, activation='sigmoid')) 
model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model2.summary()


# In[ ]:


X2_test = pca.fit_transform(X_test)
h = model2.fit(X2, y2, epochs=5, validation_data=(X2_test, y_test))


# In[ ]:


print("Loss: ", model2.evaluate(X2_test, y_test, verbose=2))


# In[ ]:


y2_predicted = np.round(model2.predict(X2_test)).T[0]
y2_correct = np.array(y_test)


# In[43]:


confusion_matrix2 = ConfusionMatrix(y2_correct, y2_predicted)
print("Confusion matrix:\n%s" % confusion_matrix2)
confusion_matrix2.plot(normalized=True)
plt.show()
confusion_matrix2.print_stats()


# In[ ]:




