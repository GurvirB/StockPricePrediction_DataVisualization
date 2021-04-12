#!/usr/bin/env python
# coding: utf-8

# In[2]:


from google.colab import files
import numpy as np
import matplotlib.pyplot as plt  # To visualize
import pandas as pd  # To read data
from sklearn.linear_model import LinearRegression
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib.dates
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# In[3]:


uploaded = files.upload()


# In[4]:


jets = pd.read_csv('JETS.csv')
#arkw = pd.read_csv('ARKW.csv')
#xph = pd.read_csv('XPH.csv')

jets.describe


# In[5]:


#change 'Date' type from Object to DateTime

jets['Date']= pd.to_datetime(jets['Date'])
jets.info()


# In[6]:


#Check correlation
corr = jets.corr()
corr


# In[7]:


sb.heatmap(corr,xticklabels=corr.columns, yticklabels=corr.columns,cmap='RdBu_r', annot=True, linewidth=0.5)


# In[8]:


#prepare dataset to work with 
#Dropping columns: Adj Close and Volume
jets_df=jets[['Date','High','Low','Open','Close']]
jets_df.head(10)


# In[9]:


#plot "Close" price vs "Year"

jets_df['Date'] = pd.to_datetime(jets_df['Date'], format='%Y-%m-%d')
plt.figure(figsize=(15,8))
plt.title('Jets Stock closing price history')
plt.plot(jets_df['Date'],jets_df['Close'])
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close Price',fontsize=18)
#plt.style.use('fivethirtyeight')
plt.show()


# In[10]:


'''
#Plot Open vs Close
jets_df[['Open','Close']].head(20).plot(kind='bar',figsize=(16,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
'''


# In[11]:


'''
#Plot High vs Close

jets_df[['High','Close']].head(20).plot(kind='bar',figsize=(16,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
'''


# In[12]:


#Plot Low vs Close
'''
jets_df[['Low','Close']].head(20).plot(kind='bar',figsize=(16,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
'''


# In[13]:


jets_df['Year']=jets_df['Date'].dt.year
jets_df['Month']=jets_df['Date'].dt.month
jets_df['Day']=jets_df['Date'].dt.day


# In[14]:


#jets_df=jets_df[['Day','Month','Year','High','Low','Open','Close']]
jets_df.head(10)


# In[15]:


plt.figure(figsize=(15, 5))
plt.plot(jets_df.Open.values, color='red', label='open')
plt.plot(jets_df.Close.values, color='green', label='close')
plt.plot(jets_df.Low.values, color='blue', label='low')
plt.plot(jets_df.High.values, color='black', label='high')
plt.title('stock price')
plt.xlabel('time [days]')
plt.ylabel('price')
plt.legend(loc='best')
#plt.show()


# In[16]:


#jets_X = jets_df.iloc[:,jets_df.columns !='Close']
#jets_y= jets_df.iloc[:, 6]


# In[17]:


#print(jets_X.shape)
#print(jets_y.shape)


# In[18]:


#Data Normalization

scaler=MinMaxScaler()
jets_df_scaled=jets_df.copy()
jets_df_scaled.iloc[:,1:5] = scaler.fit_transform(jets_df_scaled.iloc[:,1:5])
jets_df_scaled.head()


# In[19]:


#PLot to see the overall trend for all the features
plt.figure(figsize=(15, 5))
plt.plot(jets_df_scaled.Open.values, color='red', label='open')
plt.plot(jets_df_scaled.Close.values, color='green', label='close')
plt.plot(jets_df_scaled.Low.values, color='blue', label='low')
plt.plot(jets_df_scaled.High.values, color='black', label='high')
plt.title('Normalized stock price')
plt.xlabel('time [days]')
plt.ylabel('price')
plt.legend(loc='best')
#plt.show()


# In[20]:


#Modifying all the rows of the dataframe to get the difference of each row from the first one.
#Value bigger than 1 means the price increased as compared to the first row and less than 1 means that the price dropped.
jets=jets_df_scaled[['Date','Close']] 
jets['Close']=jets['Close'].div(jets['Close'].iloc[0])
jets.head


# In[35]:


jets['MA4']=jets['Close'].rolling(10).mean()   #Using Moving average of 4 days find the trend
#jets[['Close','MA10']].plot(label='Jets',figsize=(16,8))
plt.figure(figsize=(30, 8))
#plt.plot(grid=True)
plt.plot(jets['Date'],jets['Close'], 'g-', label="Close")
plt.plot(jets['Date'],jets['MA4'], 'r-', label="MA4")
plt.axhline(y=1,color = "black")
plt.title('Jets Stock closing price history')
#plt.plot(jets['Date'],jets[['Close','MA4']])
plt.legend(loc='best')


# In[27]:





# In[36]:


#Close and Moving average over 4 days zoomed for last 500 days

plt.figure(figsize=(20, 8))
plt.plot(jets['Date'].iloc[-500:],jets['Close'].iloc[-500:], 'g-', label="Close")
plt.plot(jets['Date'].iloc[-500:],jets['MA4'].iloc[-500:], 'r--', label="MA4")
plt.axhline(y=1,color = "black")
plt.legend(loc='best')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


#print("shape of jets_X:",jets_X.shape)
#print("shape of jets_y:",jets_y.shape)


# In[ ]:


#Select one feature at a time
#jets_X_high=jets_X['High']
#jets_X_low=jets_X['Low']
#jets_X_open=jets_X['Open']

#jets_X_high = jets_X.loc[:, ['High']]
#jets_X_open = jets_X.loc[:, ['Open']]
#jets_X_low = jets_X.loc[:, ['Low']]

#print("Shape of jets_X_high:",jets_X_high.shape)
#print("Shape of jets_y:",jets_y.shape)

#print("Shape of jets_X_open:",jets_X_open.shape)
#print("Shape of jets_y:",jets_y.shape)

#print("Shape of jets_X_low:",jets_X_low.shape)
#print("Shape of jets_y:",jets_y.shape)


#print("Printing values now: -----------------")
#print("jets_X_high:",jets_X_high)
#print("jets_X_open:",jets_X_open)
#print("jets_X_low:",jets_X_low)
#print("jets_y:",jets_y)


# In[ ]:


#model=LinearRegression()

#model.fit(jets_X_low, jets_y)
#y_predict = model.predict(jets_X_low)
#print("jets_y:",jets_y)
#print("jets_predict:",y_predict)


# In[ ]:


#plt.scatter(jets_X_low, jets_y)
#plt.plot(jets_X_low, y_predict, color='red')
#plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




