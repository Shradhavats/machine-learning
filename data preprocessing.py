#!/usr/bin/env python
# coding: utf-8

# In[2]:


#importing data set and basic insights 

#importing libraries
import pandas as pd 
import numpy as np 


# In[3]:


#headers for the file data 
headers_data = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]
print("headers\n", type(headers_data))


# In[4]:


#data set 
df=pd.read_csv("dataset_1.data",names=headers_data)
df


# In[5]:


#to show specific data from the beginning 
df.head(15)
#to show specific data from the end 
#df.tail(5)


# In[6]:


#pandas dataframe has object, float, int , bool and datetime64 
#in numpy array dtype is used 
df.dtypes


# In[7]:


#to know the statistics like mean ,standard deviation and count etc  (describe method is only used for int and float (I.e, value for Nan not a number is not taken into account) )
df.describe()


# In[8]:


#to include other data types 
df.describe(include="all")


# In[9]:


#to know the null count , datatype, memory usage 
df.info()


# In[10]:


#due to the presence of ? which is considered as string the data types of a few are object 
#replacing ? with nan so that the null values can be known to the system, 
df.replace("?",np.nan,inplace =True)#inplace true implies that the changes made are set in the original data set 
df.info()


# In[45]:


#to save data into a new file 
df.to_csv("saved_data")


# **Handling missing data values 
# three types : standard(empty spaces or NAN) 
# non standard : in the form of string 
# unexpected: numbers when the output is in binary form 
# using Pandas and numpy 

# In[12]:


missing_data =df.isnull() #true=nullvalue 


# In[14]:


df.notnull()#false=nullvalue 


# In[28]:


#to know about the missing value headers 
for column in headers_data:
    print(column)
    print(missing_data[column].value_counts())#value_counts()is used to ditribute unique values, it does that in descending order 
    print(" ")


# In[29]:


#to replace with mean
avg_normalized_losses=df['normalized-losses'].astype("float").mean()
df['normalized-losses'].replace(np.nan, avg_normalized_losses, inplace= True )
df.head(10)


# In[38]:


avg_bore=df['bore'].astype("float").mean()
df['bore'].replace(np.nan,avg_bore, inplace=True)

"""
to check if the null values are removed with mean 
data=df.isnull()
for column in headers_data:
    print(column)
    print(data[column].value_counts())
    print(" ")
"""


# In[42]:


avg_horsepower=df['horsepower'].astype("float").mean()
df['horsepower'].replace(np.nan,avg_horsepower, inplace=True)
avg_peak_rpm=df['peak-rpm'].astype("float").mean()
df['peak-rpm'].replace(np.nan,avg_peak_rpm,inplace=True)


# In[44]:


#to replace with mode (most frequent data)
df['num-of-doors'].replace(np.nan,df['num-of-doors'].value_counts().idxmax(),inplace=True)


# In[47]:


#to drop(usually done to drop rows or columns of target variable )
df.dropna(subset=['price'],inplace=True)#subset method is used to specify in which column we need to look for 
#reseting the index is important 
df.reset_index(drop=True , inplace =True)


# **Handling missing data using Scikit Learn 
# data
# In[ ]:




