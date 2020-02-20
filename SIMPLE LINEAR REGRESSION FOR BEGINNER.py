#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv(r"C:\Users\Lenovo\Documents\1 PT Pos Indonesia (Persero)\KERJA\5 Training Rosebay\Regression\Data dari Kaggle\Summary of Weather.csv")


# In[3]:


print(df.info())
df.head()


# In[4]:


deleted_columns = ['FT', 'FB', 'FTI', 'ITH', 'SD3', 'RHX', 'RHN', 'RVG', 'WTE']
df.drop(deleted_columns, axis=1, inplace=True)
df.info()


# #### CARA 1

# In[6]:


lr = LinearRegression()
x = df.MinTemp.values
x


# In[7]:


x.shape


# In[8]:


x = x.reshape(-1,1)
x


# In[9]:


x.shape


# In[10]:


y = df.MaxTemp.values.reshape(-1,1)
y


# In[11]:


lr.fit(x,y)


# In[12]:


# X is min temperatures given from us.
X = np.array([10,20,30,40,50]).reshape(-1,1)
X


# In[14]:


print("Results")
for i in X:
    print("Min Temp:",i,"Predicted Max Temp:",lr.predict([i]))


# In[16]:


#Visualize

plt.scatter(x,y)
plt.show()


# In[17]:


y_head = lr.predict(X)

plt.scatter(X, y_head, color="red")
plt.show()


# #### CARA 2

# In[18]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.0005)

lr.fit(x_train,y_train)
a = lr.predict(x_test)
a


# In[19]:


mean_absolute_error(a,y_test)


# In[20]:


fig,ax = plt.subplots()

plt.scatter(x_test,y_test)
plt.scatter(a,y_test)

plt.show()


# #### CARA 3

# In[21]:


#import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
import seaborn as sb
#from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LinearRegression
from sklearn import metrics
get_ipython().run_line_magic('matplotlib', 'inline')

#weatherSummary = pd.read_csv("../input/weatherww2/Summary of Weather.csv")
#weatherStationLocations = pd.read_csv("../input/weatherww2/Weather Station Locations.csv")

#weatherSummary.shape  # It will give number of Rows and Columns 

#weatherSummary.describe()

df.plot(x='MinTemp',y='MaxTemp',style='o')
plt.title('MinTemp Vs MaxTemp')
plt.xlabel('MinTemp')
plt.ylabel('MaxTemp')
plt.show()

plt.figure(figsize=(15,10))
plt.tight_layout()
sb.distplot(df['MaxTemp'])

x = df['MinTemp'].values.reshape(-1,1)
y = df['MaxTemp'].values.reshape(-1,1)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=0)

# print(len(x_train), len(y_train), len(x_test), len(y_test))

#linearRegressor = LinearRegression()
lr.fit(x_train,y_train)

print(lr.intercept_)

print(lr.coef_)

print(lr.score(x,y))


y_predict = lr.predict(x_test)

val = pd.DataFrame({'Actual':y_test.flatten(),'predicted':y_predict.flatten()})

val1 = val.head(25)
val1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major',linestyle='-',linewidth='0.5',color='green')
plt.grid(which='minor',linestyle=':',linewidth='0.5',color='black')
plt.show()

plt.scatter(x_test, y_test,  color='gray')
plt.plot(x_test, y_predict, color='red', linewidth=2)
plt.show()


print('Mean Absolute Error: ', metrics.mean_absolute_error(y_test, y_predict))  
print('Mean Squared Error: ', metrics.mean_squared_error(y_test, y_predict))  
print('Root Mean Squared Error: ', np.sqrt(metrics.mean_squared_error(y_test, y_predict)))

