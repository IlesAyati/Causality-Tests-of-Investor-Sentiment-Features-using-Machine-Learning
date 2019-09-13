#!/usr/bin/env python
# coding: utf-8

# # Pandas
# 
# 
# <a id='index-1'></a>

# ## Contents
# 
# - [Pandas](#Pandas)  
#   - [Overview](#Overview)  
#   - [Series](#Series)  
#   - [DataFrames](#DataFrames)  
#   - [On-Line Data Sources](#On-Line-Data-Sources)  
#   - [Exercises](#Exercises)  
#   - [Solutions](#Solutions)  

# ## Overview
# 
# [Pandas](http://pandas.pydata.org/) is a package of fast, efficient data analysis tools for Python
# 
# Its popularity has surged in recent years, coincident with the rise
# of fields such as data science and machine learning
# 
# Here’s a popularity comparison over time against STATA and SAS, courtesy of Stack Overflow Trends
# 
# <img src="https://s3-ap-southeast-2.amazonaws.com/lectures.quantecon.org/py/_static/figures/pandas_vs_rest.png" style="width:90%;height:90%">
# 
#   
# Just as [NumPy](http://www.numpy.org/) provides the basic array data type plus core array operations, pandas
# 
# 1. defines fundamental structures for working with data and  
# 1. endows them with methods that facilitate operations such as  
#   
#   - reading in data  
#   - adjusting indices  
#   - working with dates and time series  
#   - sorting, grouping, re-ordering and general data munging <sup><a href=#mung id=mung-link>[1]</a></sup>  
#   - dealing with missing values, etc., etc.  
#   
# 
# 
# More sophisticated statistical functionality is left to other packages, such
# as [statsmodels](http://www.statsmodels.org/) and [scikit-learn](http://scikit-learn.org/), which are built on top of pandas
# 
# This lecture will provide a basic introduction to pandas
# 
# Throughout the lecture we will assume that the following imports have taken
# place

# In[1]:


import pandas as pd
import numpy as np


# ## Series
# 
# 
# <a id='index-2'></a>
# Two important data types defined by pandas are  `Series` and `DataFrame`
# 
# You can think of a `Series` as a “column” of data, such as a collection of observations on a single variable
# 
# A `DataFrame` is an object for storing related columns of data
# 
# Let’s start with Series

# In[2]:


s = pd.Series(np.random.randn(4), name='daily returns')
s


# Here you can imagine the indices `0, 1, 2, 3` as indexing four listed
# companies, and the values being daily returns on their shares
# 
# Pandas `Series` are built on top of NumPy arrays, and support many similar
# operations

# In[3]:


s * 100


# In[4]:


np.abs(s)


# But `Series` provide more than NumPy arrays
# 
# Not only do they have some additional (statistically oriented) methods

# In[5]:


s.describe()


# But their indices are more flexible

# In[6]:


s.index = ['AMZN', 'AAPL', 'MSFT', 'GOOG']
s


# Viewed in this way, `Series` are like fast, efficient Python dictionaries
# (with the restriction that the items in the dictionary all have the same
# type—in this case, floats)
# 
# In fact, you can use much of the same syntax as Python dictionaries

# In[7]:


s['AMZN']


# In[8]:


s['AMZN'] = 0
s


# In[9]:


'AAPL' in s


# ## DataFrames
# 
# 
# <a id='index-3'></a>
# While a `Series` is a single column of data, a `DataFrame` is several columns, one for each variable
# 
# In essence, a `DataFrame` in pandas is analogous to a (highly optimized) Excel spreadsheet
# 
# Thus, it is a powerful tool for representing and analyzing data that are naturally organized  into rows and columns, often with  descriptive indexes for individual rows and individual columns
# 
# Let’s look at an example that reads data from the CSV file `pandas/data/test_pwt.csv` that can be downloaded
# <a href=/_static/code/pandas/data/test_pwt.csv download>here</a>
# 
# Here’s the contents of `test_pwt.csv`

# ```text
# "country","country isocode","year","POP","XRAT","tcgdp","cc","cg"
# "Argentina","ARG","2000","37335.653","0.9995","295072.21869","75.716805379","5.5788042896"
# "Australia","AUS","2000","19053.186","1.72483","541804.6521","67.759025993","6.7200975332"
# "India","IND","2000","1006300.297","44.9416","1728144.3748","64.575551328","14.072205773"
# "Israel","ISR","2000","6114.57","4.07733","129253.89423","64.436450847","10.266688415"
# "Malawi","MWI","2000","11801.505","59.543808333","5026.2217836","74.707624181","11.658954494"
# "South Africa","ZAF","2000","45064.098","6.93983","227242.36949","72.718710427","5.7265463933"
# "United States","USA","2000","282171.957","1","9898700","72.347054303","6.0324539789"
# "Uruguay","URY","2000","3219.793","12.099591667","25255.961693","78.978740282","5.108067988"
# ```
# 

# Supposing you have this data saved as test_pwt.csv in the present working directory (type %pwd in Jupyter to see what this is), it can be read in as follows:

# In[10]:


df = pd.read_csv('https://github.com/QuantEcon/QuantEcon.lectures.code/raw/master/pandas/data/test_pwt.csv')
type(df)


# In[11]:


df


# We can select particular rows using standard Python array slicing notation

# In[12]:


df[2:5]


# To select columns, we can pass a list containing the names of the desired columns represented as strings

# In[13]:


df[['country', 'tcgdp']]


# To select both rows and columns using integers, the `iloc` attribute should be used with the format `.iloc[rows,columns]`

# In[14]:


df.iloc[2:5, 0:4]


# To select rows and columns using a mixture of integers and labels, the `loc` attribute can be used in a similar way

# In[15]:


df.loc[df.index[2:5], ['country', 'tcgdp']]


# Let’s imagine that we’re only interested in population and total GDP (`tcgdp`)
# 
# One way to strip the data frame `df` down to only these variables is to overwrite the dataframe using the selection method described above

# In[16]:


df = df[['country', 'POP', 'tcgdp']]
df


# Here the index `0, 1,..., 7` is redundant, because we can use the country names as an index
# 
# To do this, we set the index to be the `country` variable in the dataframe

# In[17]:


df = df.set_index('country')
df


# Let’s give the columns slightly better names

# In[18]:


df.columns = 'population', 'total GDP'
df


# Population is in thousands, let’s revert to single units

# In[19]:


df['population'] = df['population'] * 1e3
df


# Next we’re going to add a column showing real GDP per capita, multiplying by 1,000,000 as we go because total GDP is in millions

# In[20]:


df['GDP percap'] = df['total GDP'] * 1e6 / df['population']
df


# One of the nice things about pandas `DataFrame` and `Series` objects is that they have methods for plotting and visualization that work through Matplotlib
# 
# For example, we can easily generate a bar plot of GDP per capita

# In[21]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

df['GDP percap'].plot(kind='bar')
plt.show()


# At the moment the data frame is ordered alphabetically on the countries—let’s change it to GDP per capita

# In[22]:


df = df.sort_values(by='GDP percap', ascending=False)
df


# Plotting as before now yields

# In[23]:


df['GDP percap'].plot(kind='bar')
plt.show()


# ## On-Line Data Sources
# 
# 
# <a id='index-4'></a>
# Python makes it straightforward to query on line databases programmatically
# 
# An important database for economists is [FRED](https://research.stlouisfed.org/fred2/) — a vast collection of time series data maintained by the St. Louis Fed
# 
# For example, suppose that we are interested in the [unemployment rate](https://research.stlouisfed.org/fred2/series/UNRATE)
# 
# Via FRED, the entire series for the US civilian unemployment rate can be downloaded directly by entering
# this URL into your browser (note that this requires an internet connection)

# ```text
# https://research.stlouisfed.org/fred2/series/UNRATE/downloaddata/UNRATE.csv
# ```
# 

# (Equivalently, click here: [https://research.stlouisfed.org/fred2/series/UNRATE/downloaddata/UNRATE.csv](https://research.stlouisfed.org/fred2/series/UNRATE/downloaddata/UNRATE.csv))
# 
# This request returns a CSV file, which will be handled by your default application for this class of files
# 
# Alternatively, we can access the CSV file from within a Python program
# 
# This can be done with a variety of methods
# 
# We start with a relatively low level method, and then return to pandas

# ### Accessing Data with requests
# 
# 
# <a id='index-6'></a>
# One option is to use [requests](http://docs.python-requests.org/en/master/), a standard Python library for requesting data over the Internet
# 
# To begin, try the following code on your computer

# In[24]:


import requests

r = requests.get('http://research.stlouisfed.org/fred2/series/UNRATE/downloaddata/UNRATE.csv')


# If there’s no error message, then the call has succeeded
# 
# If you do get an error, then there are two likely causes
# 
# 1. You are not connected to the Internet — hopefully this isn’t the case  
# 1. Your machine is accessing the Internet through a proxy server, and Python isn’t aware of this  
# 
# 
# In the second case, you can either
# 
# - switch to another machine  
# - solve your proxy problem by reading [the documentation](http://docs.python-requests.org/en/master/)  
# 
# 
# Assuming that all is working, you can now proceed to using the `source` object returned by the call `requests.get('http://research.stlouisfed.org/fred2/series/UNRATE/downloaddata/UNRATE.csv')`

# In[25]:


url = 'http://research.stlouisfed.org/fred2/series/UNRATE/downloaddata/UNRATE.csv'
source = requests.get(url).content.decode().split("\n")
source[0]


# In[26]:


source[1]


# In[27]:


source[2]


# We could now write some additional code to parse this text and store it as an array
# 
# But this is unnecessary — pandas’ `read_csv` function can handle the task for us
# 
# We use `parse_dates=True` so that pandas recognizes our dates column, allowing for simple date filtering

# In[28]:


data = pd.read_csv(url, index_col=0, parse_dates=True)


# The data has been read into a pandas DataFrame called `data` that we can now manipulate in the usual way

# In[29]:


type(data)


# In[30]:


data.head()  # A useful method to get a quick look at a data frame


# In[31]:


pd.set_option('precision', 1)
data.describe()  # Your output might differ slightly


# We can also plot the unemployment rate from 2006 to 2012 as follows

# In[32]:


data['2006':'2012'].plot()
plt.show()


# ### Accessing World Bank Data
# 
# 
# <a id='index-7'></a>
# Let’s look at one more example of downloading and manipulating data — this
# time from the World Bank
# 
# The World Bank [collects and organizes data](http://data.worldbank.org/indicator) on a huge range of indicators
# 
# For example, [here’s](http://data.worldbank.org/indicator/GC.DOD.TOTL.GD.ZS/countries) some data on government debt as a ratio to GDP
# 
# If you click on “DOWNLOAD DATA” you will be given the option to download the
# data as an Excel file
# 
# The next program does this for you, reads an Excel file into a pandas
# DataFrame, and plots time series for the US and Australia

# In[33]:


import matplotlib.pyplot as plt
import requests
import pandas as pd

# == Get data and read into file gd.xls == #
wb_data_query = "http://api.worldbank.org/v2/en/indicator/gc.dod.totl.gd.zs?downloadformat=excel"
r = requests.get(wb_data_query)
with open('gd.xls', 'wb') as output:
    output.write(r.content)

# == Parse data into a DataFrame == #
govt_debt = pd.read_excel('gd.xls', sheet_name='Data', skiprows=3, index_col=1)

# == Take desired values and plot == #
govt_debt = govt_debt.transpose()
govt_debt = govt_debt[['AUS', 'USA']]
govt_debt = govt_debt[38:]
govt_debt.plot(lw=2)
plt.show()


# (The file is `pandas/wb_download.py`, and can be downloaded
# <a href=/_static/code/pandas/wb_download.py download>here</a>

# ## Exercises
# 
# 
# <a id='pd-ex1'></a>

# ### Exercise 1
# 
# Write a program to calculate the percentage price change over 2013 for the following shares

# In[34]:


ticker_list = {'INTC': 'Intel',
               'MSFT': 'Microsoft',
               'IBM': 'IBM',
               'BHP': 'BHP',
               'TM': 'Toyota',
               'AAPL': 'Apple',
               'AMZN': 'Amazon',
               'BA': 'Boeing',
               'QCOM': 'Qualcomm',
               'KO': 'Coca-Cola',
               'GOOG': 'Google',
               'SNE': 'Sony',
               'PTR': 'PetroChina'}


# A dataset of daily closing prices for the above firms can be found in `pandas/data/ticker_data.csv`, and can be downloaded
# <a href=/_static/code/pandas/data/ticker_data.csv download>here</a>
# 
# Plot the result as a bar graph like follows
# 
# <img src="https://s3-ap-southeast-2.amazonaws.com/lectures.quantecon.org/py/_static/figures/pandas_share_prices.png" style="width:70%;height:70%">

# ## Solutions

# ### Exercise 1

# In[35]:


ticker = pd.read_csv('https://github.com/QuantEcon/QuantEcon.lectures.code/raw/master/pandas/data/ticker_data.csv')
ticker.set_index('Date', inplace=True)

ticker_list = {'INTC': 'Intel',
               'MSFT': 'Microsoft',
               'IBM': 'IBM',
               'BHP': 'BHP',
               'TM': 'Toyota',
               'AAPL': 'Apple',
               'AMZN': 'Amazon',
               'BA': 'Boeing',
               'QCOM': 'Qualcomm',
               'KO': 'Coca-Cola',
               'GOOG': 'Google',
               'SNE': 'Sony',
               'PTR': 'PetroChina'}

price_change = pd.Series()

for tick in ticker_list:
    change = 100 * (ticker.loc[ticker.index[-1], tick] - ticker.loc[ticker.index[0], tick]) / ticker.loc[ticker.index[0], tick]
    name = ticker_list[tick]
    price_change[name] = change

price_change.sort_values(inplace=True)
fig, ax = plt.subplots(figsize=(10,8))
price_change.plot(kind='bar', ax=ax)
plt.show()


# **Footnotes**
# 
# <p><a id=mung href=#mung-link><strong>[1]</strong></a> Wikipedia defines munging as cleaning data from one raw form into a structured, purged one.
