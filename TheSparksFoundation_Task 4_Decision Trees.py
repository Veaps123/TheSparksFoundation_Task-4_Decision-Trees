#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
training_data = pd.read_csv("c:\\Users\\hp\\Desktop\\iris.csv")
training_data


# In[2]:


import sklearn.datasets as datasets
iris = datasets.load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
print(data.head(5))

y = iris.target
print(y)


# In[3]:


from sklearn.tree import DecisionTreeClassifier
DTree = DecisionTreeClassifier()
model = DTree.fit(data,y)
model


# In[5]:


text_representation = tree.export_text(DTree)
print(text_representation)


# In[8]:


from sklearn.tree import plot_tree
fig = plt.figure()
_ = tree.plot_tree(DTree, feature_names = iris.feature_names,class_names = iris.target_names,filled = True)


# In[9]:


from sklearn.tree import DecisionTreeRegressor
regr = DecisionTreeRegressor(max_depth=3, random_state=1234)
model = regr.fit(data, y)


# In[10]:


fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(regr, feature_names=iris.feature_names, filled=True)


# In[ ]:




