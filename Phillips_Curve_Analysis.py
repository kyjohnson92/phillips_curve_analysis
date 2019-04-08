
# coding: utf-8

# In[778]:


#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
get_ipython().magic('matplotlib inline')
import warnings
warnings.filterwarnings('ignore')
import datetime


# In[779]:


#import data and examine columns
data = pd.read_csv('FI Case Data.csv')
data.columns.tolist()


# In[780]:


#examine unique metrics
data.Data.unique()


# In[781]:


#reformat 
data = data.replace('Consumer Price Index', 'CPI')
data.Data.unique()


# In[782]:


#examine datatypes
data.info()


# In[783]:


#examine statistical qualities of dataset
data.describe()


# In[784]:


#create function to create dataframe of metric data by annual means
def dfcreator(df, my_list=[]):
    new_df = pd.DataFrame()
    new_df['Year'] =(df['Year'].unique)
    new_df = new_df.set_index('Year')
    for i in my_list:
        temp_df = df.loc[(df['Data']) == i]
        temp_df = temp_df.groupby('Year').mean()
        temp_df = temp_df.drop('Month ',axis=1)
        temp_df[str(i)+'_Value'] = temp_df
        new_df = pd.concat([new_df, temp_df])
        temp_df = pd.DataFrame()
    new_df = new_df.groupby('Year').mean()
    new_df = new_df.drop('Value', axis=1)
    return new_df


# In[785]:


#create and examine new dataframe
metric_list =['CPI','Unemployment Level', 'Civilian Labor Force']
metric_data = dfcreator(data,my_list=metric_list)
metric_data.tail()


# In[786]:


#define and create inflation and unemployment rate functions
def inflation(cpi_initial, cpi_end):
    return ((cpi_end - cpi_initial)/ cpi_initial) * 100

def unemployment_rate(unemployment_level, labor_force):
    return (unemployment_level / labor_force) * 100


# In[787]:


#pass inflation and unemployment rate functions over dataframe
metric_data['Inflation'] = inflation(metric_data.CPI_Value.shift(), metric_data.CPI_Value)
metric_data['Unemployment_Rate'] = unemployment_rate(metric_data['Unemployment Level_Value'], metric_data['Civilian Labor Force_Value'])
metric_data.tail()


# In[788]:


#inflation and unemployment plot
fig = plt.figure(figsize=(16, 6))
sns.set(style='darkgrid')
plt.plot(metric_data.Inflation)
plt.plot(metric_data.Unemployment_Rate)
plt.xlabel("Year")
plt.ylabel("Percentage")
plt.title('U.S Inflation & Unemployment Over Time')
plt.legend()
plt.show()


# ![phillips curve](phillipcurve.jpg)

# In[789]:


#inflation/unemployment rate relationship plot
fig = plt.figure(figsize=(12, 8))
sns.set(style='darkgrid')
plt.scatter(metric_data.Unemployment_Rate, metric_data.Inflation)
plt.xlabel("Unemployment Rate(Percentage)")
plt.ylabel("Inflation Rate(Percentage)")
plt.title('US Inflation/Unemployment Relationship')
plt.show()


# In[790]:


#inflation/unemployment rate relationship plot with line of best fit
fig = plt.figure(figsize=(12, 8))
sns.set(style='darkgrid')
sns.regplot(metric_data.Unemployment_Rate, metric_data.Inflation)
plt.title('US Inflation/Unemployment Relationship')


# In[772]:


#load in euro cpi data
euro_cpi = pd.read_csv('EU_CPI_Data.csv')
euro_cpi.DATE = pd.to_datetime(euro_cpi.DATE) 
euro_cpi = euro_cpi.groupby(euro_cpi['DATE'].map(lambda x:x.year)).mean()
euro_cpi.reset_index(inplace=True)


# In[773]:


#load in euro unemployment data
euro_unemp = pd.read_csv('EU_Unemp_Data.csv')
euro_unemp.DATE = pd.to_datetime(euro_unemp.DATE)
euro_unemp = euro_unemp.groupby(euro_unemp['DATE'].map(lambda x:x.year)).mean()
euro_unemp.reset_index(inplace=True)


# In[776]:


#combine dataframes and calculate inflation rate
euro_data = pd.merge(euro_cpi,euro_unemp, on=['DATE'])
euro_data['Inflation'] = inflation(euro_data.EURO_CPI.shift(),euro_data.EURO_CPI)
euro_data.tail(2)


# In[775]:


#eurozone inflation and unemployment plot
fig = plt.figure(figsize=(16, 6))
sns.set(style='darkgrid')
plt.plot(euro_data.DATE, euro_data.Inflation)
plt.plot(euro_data.DATE, euro_data.EU_Unemployment_Rate)
plt.xlabel("Year")
plt.ylabel("Percentage")
plt.title('EU Inflation & Unemployment Over Time')
plt.legend()
plt.show()


# In[777]:


#eurozone inflation/unemployment relationship
fig = plt.figure(figsize=(12, 8))
sns.set(style='darkgrid')
sns.regplot(euro_data.EU_Unemployment_Rate, euro_data.Inflation)
plt.title('EU Inflation/Unemployment Relationship')
plt.show()


# In[627]:


#create arrays of metrics by decade
seventies_inflation = metric_data.Inflation[(metric_data.index > 1970) & (metric_data.index < 1980)]
eighties_inflation = metric_data.Inflation[(metric_data.index > 1980) & (metric_data.index < 1990) ]
nineties_inflation = metric_data.Inflation[(metric_data.index > 1990) & (metric_data.index < 2000) ]
millenium_inflation = metric_data.Inflation[(metric_data.index > 2000)]

seventies_unemployment = metric_data.Unemployment_Rate[(metric_data.index > 1970) & (metric_data.index < 1980)]
eighties_unemployment = metric_data.Unemployment_Rate[(metric_data.index > 1980) & (metric_data.index < 1990) ]
nineties_unemployment = metric_data.Unemployment_Rate[(metric_data.index > 1990) & (metric_data.index < 2000) ]
millenium_unemployment = metric_data.Unemployment_Rate[(metric_data.index > 2000)]


# In[679]:


#inflation/unemployment relationship by decade with line of best fit
fig = plt.figure(figsize=(12, 8))
sns.set(style='darkgrid', palette='muted')
sns.regplot(seventies_unemployment,seventies_inflation, marker='+')
sns.regplot(eighties_unemployment,eighties_inflation, marker='+')
sns.regplot(nineties_unemployment,nineties_inflation, marker='+')
sns.regplot(millenium_unemployment,millenium_inflation, marker='+')
plt.legend(labels=['70s','80s','90s','00s'])
plt.title("Relationship by Decade")
plt.show()


# In[711]:


#correlation heatmap
corr = metric_data[['Inflation','Unemployment_Rate']].corr()

cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, cmap=cmap)


# In[694]:


#calculate and display correlations between inflation and unemployment
def correlation_finder(x,y):
    return x.corr(y)

print("Correlation 1970s: "+str(correlation_finder(seventies_inflation,seventies_unemployment)))
print("Correlation 1980s: "+str(correlation_finder(eighties_inflation,eighties_unemployment)))
print("Correlation 1990s: "+str(correlation_finder(nineties_inflation,nineties_unemployment)))
print("Correlation 2000s: "+str(correlation_finder(millenium_inflation,millenium_unemployment)))
print("Correlation Overall "+str(correlation_finder(metric_data.Inflation,metric_data.Unemployment_Rate)))
print("Correlation In EU Since 1996: "+str(correlation_finder(euro_data.Inflation,euro_data.EU_Unemployment_Rate)))

