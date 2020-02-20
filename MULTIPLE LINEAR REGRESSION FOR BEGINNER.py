#!/usr/bin/env python
# coding: utf-8

# ##### IMPORT LIBRARY

# In[1]:


import numpy as np 
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sb


# ##### LOAD RAW DATA TO DATAFRAME

# In[2]:


df = pd.read_csv(r"C:\Users\Lenovo\Documents\1 PT Pos Indonesia (Persero)\KERJA\5 Training Rosebay\Regression\Data dari Kaggle\Summary of Weather.csv")


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


df.describe(include=['O'])


# In[7]:


df.describe(exclude=['O'])


# ##### DATA CLEANING

# In[8]:


df.isnull().sum()


# In[10]:


print (df.shape)


# In[12]:


#Lets see what percentage of each column has null values
#It means count number of nulls in every column and divide by total num of rows.

print (df.isnull().sum()/df.shape[0] * 100)


# In[13]:


cols = [col for col in df.columns if (df[col].isnull().sum()/df.shape[0] * 100 < 100)]
cols


# In[16]:


df = df[cols]
print ('Legitimate columns after dropping null columns: %s' % df.shape[1])


# In[17]:


df.isnull().sum()


# In[18]:


df.dtypes


# In[19]:


#Looks like some columns needs to be converted to numeric field
df['Date'] = pd.to_datetime(df['Date'])
df['Precip'] = pd.to_numeric(df['Precip'], errors='coerce')
df['Snowfall'] = pd.to_numeric(df['Snowfall'], errors='coerce')
df['PoorWeather'] = pd.to_numeric(df['PoorWeather'], errors='coerce')
df['PRCP'] = pd.to_numeric(df['PRCP'], errors='coerce')
df['SNF'] = pd.to_numeric(df['SNF'], errors='coerce')
df['TSHDSBRSGF'] = pd.to_numeric(df['TSHDSBRSGF'], errors='coerce')


# In[20]:


#Fill remaining null values. FOr the moment lts perform ffill

df.fillna(method='ffill', inplace=True)
df.fillna(method='bfill', inplace=True)
df.isnull().sum()


# In[21]:


df.dtypes


# In[24]:


#Plot couple of columns to see how the data is scaled

fig, ax = plt.subplots(4, 2, figsize=(15, 15))

sb.distplot(df['Precip'], ax=ax[0][0])
#sb.distplot(df['Precip_scaled'], ax=ax[0][1])

sb.distplot(df['MeanTemp'], ax=ax[1][0])
#sb.distplot(df['MeanTemp_scaled'], ax=ax[1][1])

sb.distplot(df['Snowfall'], ax=ax[2][0])
#sb.distplot(df['Snowfall_scaled'], ax=ax[2][1])

sb.distplot(df['MAX'], ax=ax[3][0])
#sb.distplot(df['MAX_scaled'], ax=ax[3][1])


# #### CORRELATION

# In[25]:


pearsoncorr = df.corr(method='pearson')
pearsoncorr


# In[51]:


plt.figure(figsize=(12, 10))

matrix = np.triu(df.corr())
sb.heatmap(pearsoncorr, 
           fmt='.3g', 
           vmin=-1, vmax=1, center= 0,
           annot = False, 
           square = True, 
           cmap = 'coolwarm',
           linewidth = 1)


# In[28]:


df.SND.value_counts()


# #### MODEL 1
# ##### Independent variables: DR, PGT
# ##### Dependent variables: WindGustSpd

# In[52]:


lr = LinearRegression()
x = df[['DR','PGT']] # here we have 2 variables for multiple regression
                        # if you just want to use one variable for simple linear regression
x


# In[53]:


x.shape


# In[54]:


y = df.WindGustSpd.values.reshape(-1,1)
y


# In[55]:


y.shape


# In[56]:


lr.fit(x,y)


# In[57]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.0005)

lr.fit(x_train,y_train)
a = lr.predict(x_test)
a


# In[58]:


mean_absolute_error(a,y_test)


# In[59]:


x = sm.add_constant(x) # adding a constant
x


# In[60]:


model = sm.OLS(y, x).fit()
model


# In[61]:


predictions = model.predict(x) 
predictions


# In[62]:


print(model.summary())


# #### MODEL 2
# ##### Independent variables: DR, TSHDSBRSGF
# ##### Dependent variables: WindGustSpd

# In[63]:


x2 = df[['DR','TSHDSBRSGF']]
x2


# In[65]:


lr.fit(x2,y)

x2_train,x2_test,y2_train,y2_test = train_test_split(x2,y,test_size=0.0005)

lr.fit(x2_train,y2_train)
a2 = lr.predict(x2_test)
a2


# In[66]:


mean_absolute_error(a2,y2_test)


# In[67]:


x2 = sm.add_constant(x2) # adding a constant
x2


# In[68]:


model2 = sm.OLS(y, x2).fit()
model2


# In[69]:


predictions2 = model.predict(x2) 
predictions2


# In[70]:


print(model2.summary())

