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
from numpy.linalg import inv
import scipy.stats as stats


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


# In[8]:


pearsoncorr_raw = df.corr(method='pearson')
pearsoncorr_raw


# In[9]:


plt.figure(figsize=(12, 10))

matrix_raw = np.triu(df.corr())
sb.heatmap(pearsoncorr_raw, 
           fmt='.3g', 
           vmin=-1, vmax=1, center= 0,
           annot = False, 
           square = True, 
           cmap = 'coolwarm',
           linewidth = 1)


# In[10]:


plt.plot(df.DR, df.WindGustSpd, 'o', color='magenta');
plt.title("Checking Linear Relationship")


# In[11]:


plt.plot(df.MIN, df.MinTemp, 'o', color='magenta');
plt.title("Checking Linear Relationship")


# In[12]:


plt.plot(df.MaxTemp, df.MinTemp, 'o', color='magenta');
plt.title("Checking Linear Relationship")

### YEAY LINEAR !!!!!!!!!


# In[13]:


plt.plot(df.MIN, df.MaxTemp, 'o', color='magenta');
plt.title("Checking Linear Relationship")

### YEAY LINEAR !!!!!!!!!


# In[14]:


plt.plot(df.MIN, df.WindGustSpd, 'o', color='magenta');
plt.title("Checking Linear Relationship")


# In[15]:


df.MIN.unique()


# Temporary conclusions:
# 1. Y: MinTemp, X: MaxTemp, MIN

# ##### DATA CLEANING

# In[16]:


df.isnull().sum()


# In[17]:


print (df.shape)


# In[18]:


#Lets see what percentage of each column has null values
#It means count number of nulls in every column and divide by total num of rows.

print (df.isnull().sum()/df.shape[0] * 100)


# In[19]:


cols = [col for col in df.columns if (df[col].isnull().sum()/df.shape[0] * 100 < 100)]
cols


# In[20]:


df_new = df[cols]
print ('Legitimate columns after dropping null columns: %s' % df_new.shape[1])


# In[21]:


df_new.isnull().sum()


# In[22]:


df_new.dtypes


# In[23]:


#Looks like some columns needs to be converted to numeric field
df_new['Date'] = pd.to_datetime(df_new['Date'])
df_new['Precip'] = pd.to_numeric(df_new['Precip'], errors='coerce')
df_new['Snowfall'] = pd.to_numeric(df_new['Snowfall'], errors='coerce')
df_new['PoorWeather'] = pd.to_numeric(df_new['PoorWeather'], errors='coerce')
df_new['PRCP'] = pd.to_numeric(df_new['PRCP'], errors='coerce')
df_new['SNF'] = pd.to_numeric(df_new['SNF'], errors='coerce')
df_new['TSHDSBRSGF'] = pd.to_numeric(df_new['TSHDSBRSGF'], errors='coerce')


# In[24]:


#Fill remaining null values. FOr the moment lts perform ffill

df_new.fillna(method='ffill', inplace=True)
df_new.fillna(method='bfill', inplace=True)
df_new.isnull().sum()


# In[25]:


df_new.dtypes


# In[26]:


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

# In[27]:


pearsoncorr_new = df_new.corr(method='pearson')
pearsoncorr_new


# In[28]:


plt.figure(figsize=(12, 10))

matrix_new = np.triu(df_new.corr())
sb.heatmap(pearsoncorr_new, 
           fmt='.3g', 
           vmin=-1, vmax=1, center= 0,
           annot = False, 
           square = True, 
           cmap = 'coolwarm',
           linewidth = 1)


# In[29]:


df_new.SND.value_counts()


# #### MODEL 1
# ##### Independent variables: MIN, MaxTemp
# ##### Dependent variables: MinTemp

# ##### CHECKING LINEAR REGRESSION ASSUMPTIONS

# 1. Investigating a Linear Relationship

# In[30]:


plt.plot(df_new.MIN, df_new.MinTemp, '.', color='magenta');
plt.title("Checking Linear Relationship MIN and MinTemp")


# In[31]:


plt.plot(df_new.MaxTemp, df_new.MinTemp, '.', color='green');
plt.title("Checking Linear Relationship MaxTemp and MinTemp")


# 2. Variables should follow a Normal Distribution ############# WRONG! YOU JUST HAVE TO CHECK NORMALITY AFTER REGRESSION BECAUSE THE THING THAT HAS TO BE NORMAL IS THE ERROR! OMG HELLOOOO :)

# In[ ]:


stats.probplot(df_new.MinTemp, dist="norm", plot=plt)
plt.show()


# In[ ]:


stats.probplot(df_new.MIN, dist="norm", plot=plt)
plt.show()


# In[ ]:


stats.probplot(df_new.MaxTemp, dist="norm", plot=plt)
plt.show()


# In[ ]:


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


# In[ ]:


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


# In[ ]:


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


# In[ ]:


# Shapiro-Wilk Test:  the test may be suitable for smaller samples of data, e.g. thousands of observations or fewer.

# normality test
stat, p = shapiro(df_new.MinTemp)
print('Statistics=%.3f, p=%.3f' % (stat, p))

# interpret
alpha = 0.05
if p > alpha:
    print('Sample looks Gaussian (fail to reject H0)')
else:
    print('Sample does not look Gaussian (reject H0)')


# In[ ]:


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

# In[32]:


lr = LinearRegression()
x = df_new[['MIN','MaxTemp']] # here we have 2 variables for multiple regression
                        # if you just want to use one variable for simple linear regression
x


# In[33]:


x.shape


# In[34]:


y = df.MinTemp.values.reshape(-1,1)
y


# In[35]:


y.shape


# In[36]:


lr.fit(x,y)


# In[37]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.0005)

lr.fit(x_train,y_train)
a = lr.predict(x_test)
a


# In[38]:


mean_absolute_error(a,y_test)


# In[39]:


x = sm.add_constant(x) # adding a constant
x


# In[40]:


model = sm.OLS(y, x).fit()
model


# In[41]:


predictions = model.predict(x) 
predictions


# In[42]:


print(model.summary())


# ##### MANUALLY

# In[43]:


data1 = df_new[['MIN','MaxTemp','MinTemp']]
data1


# In[44]:


#convert dataframe to numpy matrix
dataM = data1.as_matrix()
dataM


# In[45]:


X_manual, y_manual = dataM[:,[1,2]], dataM[:,0]


# In[46]:


X_manual


# In[47]:


y_manual


# In[48]:


x0 = np.ones((len(X_manual), 1), dtype=int)
x0


# In[49]:


x_manual = np.concatenate((x0, X_manual), axis=1)
x_manual


# In[50]:


b = inv(x_manual.T.dot(x_manual)).dot(x_manual.T).dot(y_manual)
b


# In[51]:


yhat = x_manual.dot(b)
yhat


# In[54]:


d1 = y_manual-yhat
d1


# In[55]:


d2 = y_manual - y_manual.mean()
d2


# In[56]:


rsquared = 1 - d1.dot(d1) / d2.dot(d2)
rsquared

