#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import necessary packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


#Reading the data

dta = pd.read_csv("D:\AIML\Dataset\Cellsamples.csv")


# In[3]:


#Removing unnecessary columns

a_dta = dta.drop('ID',axis=1)


# In[4]:


#Removing the unnecessary columns

b_data = a_dta.drop('BareNuc',axis = 1)


# In[5]:


#Selecting the columns required for the model

cell_data = b_data[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize',
       'BlandChrom', 'NormNucl', 'Mit', 'Class']]


# In[6]:


#Declaring Independant variable

x = np.asarray(cell_data[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize',
       'BlandChrom', 'NormNucl', 'Mit']]) #Independant variable


# In[7]:


y = np.asarray(cell_data['Class']) #Dependant variable


# In[8]:


#Normalizing the data
#Import necessary package for normalizing the data

from sklearn import preprocessing

x = preprocessing.StandardScaler().fit(x).transform(x)


# In[9]:


#Splitting the data into train & test
#Import necessary package for splitting 

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state = 200)

print("Train set:",x_train.shape,y_train.shape)
print("Test set:",x_test.shape,y_test.shape)


# In[10]:


#Modeling

from sklearn.tree import DecisionTreeClassifier

cancer_dtc = DecisionTreeClassifier(criterion='entropy',max_depth=4)

cancer_dtc.fit(x_train,y_train)


# In[11]:


#Evaluating the 'DecisionTree' model using test data

dtc_pred = cancer_dtc.predict(x_test)


# In[ ]:




