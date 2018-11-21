#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score,auc
from sklearn.model_selection import train_test_split
from sklearn import svm

# In[2]:


data = pd.read_csv('creditcard.csv')
data.head()


# In[3]:


data.info()


# In[4]:


data[data['Class']==1].describe()    # Describing the data


# In[5]:


# Scaling the Time and Amount features
data['Scaled_Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))
data['Scaled_Time'] = StandardScaler().fit_transform(data['Time'].values.reshape(-1,1))
data.drop(['Time', 'Amount'], axis=1, inplace=True)
data.head()


# In[6]:


# Splitting the data into input features (X), and output target (Y)
X = data.iloc[:, data.columns != "Class"]
Y = data.iloc[:, data.columns == "Class"]


# In[7]:


X.head()


# In[8]:


Y.head()


# In[9]:


# Splitting the training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=50)


# In[10]:


# Decision Tree

#clf= DecisionTreeClassifier()
#clf.fit(X_train, y_train)	
#pred = clf.predict(X_test)
clf = svm.SVC()
clf.fit(X_train, y_train)
pred = clf.predict(X_test)

# In[11]:


print(classification_report(y_test, pred))


# In[12]:


matrix = confusion_matrix(y_test, pred)
print(matrix)


# In[13]:


sns.heatmap(matrix, cmap="coolwarm_r", annot=True, linewidths=0.5)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Real Class")
plt.show()


# In[14]:


# Calculating the Area Under the Curve
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, pred)
roc_auc = auc(false_positive_rate, true_positive_rate)
print (roc_auc)


# In[15]:


# Plotting the ROC Curve
plt.title('ROC')
plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[16]:



