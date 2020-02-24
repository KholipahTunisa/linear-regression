#!/usr/bin/env python
# coding: utf-8

# ##### IMPORT LIBRARY

# In[227]:


import numpy as np 
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sb
from numpy.linalg import inv
import scipy.stats as stats
import os
import random 
import math


# ##### LOAD RAW DATA TO DATAFRAME

# In[228]:


# Load data
os.chdir("D://FILE//konsultan l//learning and predict//weatherww2a//")
df = pd.read_csv('Summary of Weather.csv', delimiter=",")


# In[229]:


df.head()


# In[230]:


df.info()


# In[231]:


df.describe()


# In[232]:


df.describe(include=['O'])


# In[233]:


df.describe(exclude=['O'])


# In[234]:


pearsoncorr_raw = df.corr(method='pearson')
pearsoncorr_raw


# In[235]:


plt.figure(figsize=(12, 10))

matrix_raw = np.triu(df.corr())
sb.heatmap(pearsoncorr_raw, 
           fmt='.3g', 
           vmin=-1, vmax=1, center= 0,
           annot = False, 
           square = True, 
           cmap = 'coolwarm',
           linewidth = 1)


# In[236]:


plt.plot(df.DR, df.WindGustSpd, 'o', color='magenta');
plt.title("Checking Linear Relationship")


# In[237]:


plt.plot(df.MIN, df.MinTemp, 'o', color='magenta');
plt.title("Checking Linear Relationship")


# In[238]:


plt.plot(df.MaxTemp, df.MinTemp, 'o', color='magenta');
plt.title("Checking Linear Relationship")

### YEAY LINEAR !!!!!!!!!


# In[239]:


plt.plot(df.MIN, df.MaxTemp, 'o', color='magenta');
plt.title("Checking Linear Relationship")

### YEAY LINEAR !!!!!!!!!


# In[240]:


plt.plot(df.MIN, df.WindGustSpd, 'o', color='magenta');
plt.title("Checking Linear Relationship")


# In[241]:


df.MIN.unique()


# Temporary conclusions:
# 1. Y: MinTemp, X: MaxTemp, MIN

# ##### DATA CLEANING

# In[242]:


df.isnull().sum()


# In[243]:


print (df.shape)


# In[244]:


#Lets see what percentage of each column has null values
#It means count number of nulls in every column and divide by total num of rows.

print (df.isnull().sum()/df.shape[0] * 100)


# In[245]:


cols = [col for col in df.columns if (df[col].isnull().sum()/df.shape[0] * 100 < 100)]
cols


# In[246]:


df_new = df[cols]
print ('Legitimate columns after dropping null columns: %s' % df_new.shape[1])


# In[247]:


df_new.isnull().sum()


# In[248]:


df_new.dtypes


# In[249]:


#Looks like some columns needs to be converted to numeric field
df_new['Date'] = pd.to_datetime(df_new['Date'])
df_new['Precip'] = pd.to_numeric(df_new['Precip'], errors='coerce')
df_new['Snowfall'] = pd.to_numeric(df_new['Snowfall'], errors='coerce')
df_new['PoorWeather'] = pd.to_numeric(df_new['PoorWeather'], errors='coerce')
df_new['PRCP'] = pd.to_numeric(df_new['PRCP'], errors='coerce')
df_new['SNF'] = pd.to_numeric(df_new['SNF'], errors='coerce')
df_new['TSHDSBRSGF'] = pd.to_numeric(df_new['TSHDSBRSGF'], errors='coerce')


# In[250]:


#Fill remaining null values. FOr the moment lts perform ffill

df_new.fillna(method='ffill', inplace=True)
df_new.fillna(method='bfill', inplace=True)
df_new.isnull().sum()


# In[251]:


df_new.dtypes


# In[252]:


#Plot couple of columns to see how the data is scaled

fig, ax = plt.subplots(4, 2, figsize=(15, 15))

sb.distplot(df_new['Precip'], ax=ax[0][0])
#sb.distplot(df['Precip_scaled'], ax=ax[0][1])

sb.distplot(df_new['MeanTemp'], ax=ax[1][0])
#sb.distplot(df['MeanTemp_scaled'], ax=ax[1][1])

sb.distplot(df_new['Snowfall'], ax=ax[2][0])
#sb.distplot(df['Snowfall_scaled'], ax=ax[2][1])

sb.distplot(df_new['MAX'], ax=ax[3][0])
#sb.distplot(df['MAX_scaled'], ax=ax[3][1])


# #### CORRELATION

# In[253]:


pearsoncorr_new = df_new.corr(method='pearson')
pearsoncorr_new


# In[254]:


plt.figure(figsize=(12, 10))

matrix_new = np.triu(df_new.corr())
sb.heatmap(pearsoncorr_new, 
           fmt='.3g', 
           vmin=-1, vmax=1, center= 0,
           annot = False, 
           square = True, 
           cmap = 'coolwarm',
           linewidth = 1)


# In[255]:


df_new.SND.value_counts()


# #### MODEL 1
# ##### Independent variables: MIN, MaxTemp
# ##### Dependent variables: MinTemp

# ##### CHECKING LINEAR REGRESSION ASSUMPTIONS

# 1. Investigating a Linear Relationship

# In[256]:


plt.plot(df_new.MIN, df_new.MinTemp, '.', color='magenta');
plt.title("Checking Linear Relationship MIN and MinTemp")


# In[257]:


plt.plot(df_new.MaxTemp, df_new.MinTemp, '.', color='green');
plt.title("Checking Linear Relationship MaxTemp and MinTemp")


# 2. Variables should follow a Normal Distribution ############# WRONG! YOU JUST HAVE TO CHECK NORMALITY AFTER REGRESSION BECAUSE THE THING THAT HAS TO BE NORMAL IS THE ERROR! OMG HELLOOOO :)

# In[258]:


stats.probplot(df_new.MinTemp, dist="norm", plot=plt)
plt.show()


# In[259]:


stats.probplot(df_new.MIN, dist="norm", plot=plt)
plt.show()


# In[260]:


stats.probplot(df_new.MaxTemp, dist="norm", plot=plt)
plt.show()


# In[261]:


# Anderson-Darling Test

from scipy.stats import anderson

# normality test
result = anderson(df_new.MinTemp)
print('Statistic: %.3f' % result.statistic)
p = 0
for i in range(len(result.critical_values)):
    sl, cv = result.significance_level[i], result.critical_values[i]
    if result.statistic < result.critical_values[i]:
        print('%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))
    else:
        print('%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))


# In[262]:


# Anderson-Darling Test

from scipy.stats import anderson

# normality test
result = anderson(df_new.MaxTemp)
print('Statistic: %.3f' % result.statistic)
p = 0
for i in range(len(result.critical_values)):
    sl, cv = result.significance_level[i], result.critical_values[i]
    if result.statistic < result.critical_values[i]:
        print('%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))
    else:
        print('%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))


# In[263]:


# Anderson-Darling Test

# normality test
result = anderson(df_new.MIN)
print('Statistic: %.3f' % result.statistic)
p = 0
for i in range(len(result.critical_values)):
    sl, cv = result.significance_level[i], result.critical_values[i]
    if result.statistic < result.critical_values[i]:
        print('%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))
    else:
        print('%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))


# In[264]:


# Shapiro-Wilk Test:  the test may be suitable for smaller samples of data, e.g. thousands of observations or fewer.
from scipy.stats import shapiro
# normality test
stat, p = shapiro(df_new.MinTemp)
print('Statistics=%.3f, p=%.3f' % (stat, p))

# interpret
alpha = 0.05
if p > alpha:
    print('Sample looks Gaussian (fail to reject H0)')
else:
    print('Sample does not look Gaussian (reject H0)')


# In[265]:


# D'Agostino and Pearson's Test

from scipy.stats import normaltest

# normality test
stat, p = normaltest(df_new.MinTemp)
print('Statistics=%.3f, p=%.3f' % (stat, p))

# interpret
alpha = 0.05
if p > alpha:
    print('Sample looks Gaussian (fail to reject H0)')
else:
    print('Sample does not look Gaussian (reject H0)')


# ***WHAT AM I DOING THO HAHAHA*** SKIP

# 2. Multicollinearity Test #### DONE BEFORE: CORRELATION

# 3. Multicollinearity Test #### DONE BEFORE: CORRELATION

# ##### USING SCIKIT LEARN PACKAGE

# In[266]:


lr = LinearRegression()
x = df_new[['MIN','MaxTemp']] # here we have 2 variables for multiple regression
                        # if you just want to use one variable for simple linear regression
x


# In[267]:


x.shape


# In[268]:


y = df.MinTemp.values.reshape(-1,1)
y


# In[269]:


y.shape


# In[270]:


lr.fit(x,y)


# In[271]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.0005)

lr.fit(x_train,y_train)
a = lr.predict(x_test)
a


# In[272]:


mean_absolute_error(a,y_test)


# In[273]:


x = sm.add_constant(x) # adding a constant
x


# In[274]:


model = sm.OLS(y, x).fit()
model


# In[275]:


predictions = model.predict(x) 
predictions


# In[276]:


print(model.summary())


# ### SPLIT DATA TRAIN AND TEST

# In[281]:


df_new=df_new.reset_index()
sample=[]
nrow=len(df.index)
train=0.8
datasample=df_new['index'].tolist()
data_test=random.sample(datasample, int(nrow*train))
df_test = pd.DataFrame({'index':data_test})
df_train=pd.merge(df_test, df_new, how = 'left', left_on='index', right_on='index')
df_test=df_new[(~df_new.index.isin(df_train.index))]


# In[278]:


df_train.info()


# ##### MANUALLY

# ### TRAIN DATA

# In[306]:


data1 = df_train[['MIN','MaxTemp','MinTemp']]
dataM = data1.as_matrix()
X_manual, y_manual = dataM[:,[1,2]], dataM[:,0]
x0 = np.ones((len(X_manual), 1), dtype=int)
x_manual = np.concatenate((x0, X_manual), axis=1)
b = inv(x_manual.T.dot(x_manual)).dot(x_manual.T).dot(y_manual)
#b = np.linalg.solve(np.dot(x_manual.T, x_manual),np.dot(x_manual.T,y_manual))

yhat = x_manual.dot(b)

#MSE
error = y_manual-yhat
e2=error**2
mse=np.mean(e2)
print("MSE: "+str(mse))

#R SQUARE
sumE = np.sum(e2)
y_s = np.sum((y_manual-(np.mean(y_manual)*len(y_manual)))**2)
R2=(1-(sumE/y_s))*100
print('R-square: '+str(R2)+"%")


# ### TEST DATA

# In[312]:


data1 = df_test[['MIN','MaxTemp','MinTemp']]
dataM = data1.as_matrix()
X_manual, y_manual = dataM[:,[1,2]], dataM[:,0]
x0 = np.ones((len(X_manual), 1), dtype=int)
x_manual = np.concatenate((x0, X_manual), axis=1)

#MSE
y_test=b.dot(x_manual.T)
error=y_manual-y_test
e2=error**2
mse=np.mean(e2)
print("MSE: "+str(mse))


# In[ ]:




