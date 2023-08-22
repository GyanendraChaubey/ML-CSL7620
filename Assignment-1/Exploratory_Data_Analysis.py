import numpy as np
import pandas as pd

data=pd.read_csv('D:\Machine Learning Assignments\ML-CSL7620\Assignment-1\student_data.csv')

data.head()

#Describe data
data.describe()

#check shape of data
data.shape

#columns in data
data.columns

#Check null values in data
data.isnull().sum()

#Check unique values
data['Hours Studied'].unique()

data['Previous Scores'].unique()

data['Duration of Sleep'].unique()

data['Sample Question Papers Practiced'].unique()

data['Performance'].unique()

data['Extracurricular Activities'].unique()

#Mapping Yes with 1 and No with zero
data['Extracurricular Activities']=data['Extracurricular Activities'].map({'Yes':1,'No':0})

#Checking tail of data
data.tail()

#First checking the distribution of Performance
import seaborn as sns
sns.distplot(data['Performance'])

#Hours Studied
sns.boxplot(data['Hours Studied'])

#Previous Scores
sns.boxplot(data['Previous Scores'])

#Previous Scores
sns.boxplot(data['Sample Question Papers Practiced'])

#Duration of Sleep
sns.boxplot(data['Duration of Sleep'])

#Extracurricular 
sns.countplot(x='Extracurricular Activities',data=data)

y=data['Performance']
x=data.drop(columns=['Performance'])

