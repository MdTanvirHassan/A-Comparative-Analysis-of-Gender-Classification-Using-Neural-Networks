#!/usr/bin/env python
# coding: utf-8

# In[22]:


import pandas as p
import numpy as n
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[32]:


data = p.read_csv("gender_classification.csv")
data


# In[43]:


data.head(3)
# display


# In[33]:


features = data.columns
features


# In[34]:


features = [x for x in features if x != 'gender']
features


# In[35]:


train, test = train_test_split(data, test_size = 0.4)
print(len(file))
print(len(train))
print(len(test))


# In[36]:


mlp = MLPClassifier(hidden_layer_sizes = (8,8,8),max_iter = 900, activation = 'relu')


# In[38]:


x_train = train[features]
y_train = train["gender"]

x_test = test[features]
y_test = test["gender"]


# In[39]:


mlp = mlp.fit(x_train, y_train)


# In[40]:


y_pred = mlp.predict(x_test)


# In[41]:


y_pred #These are the predicted values.


# In[42]:


from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, y_pred) * 100
print("Accuracy using desicion Tree: ", round(score, 2), "%" )


# In[ ]:




