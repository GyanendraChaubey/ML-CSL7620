import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv('D:\Machine Learning Assignments\ML-CSL7620\Assignment-1\student_data.csv')

#Mapping Yes with 1 and No with zero
data['Extracurricular Activities']=data['Extracurricular Activities'].map({'Yes':1,'No':0})

y=data['Performance']
x=data.drop(columns=['Performance'])

#Check the shape of x, y
print(x.shape)
print(y.shape)

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

sc.fit(x)

x=sc.transform(x)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

#print the shape of x_train, x_test, y_train, y_test
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

from sklearn.linear_model import LinearRegression

reg_lib=LinearRegression()

reg_lib.fit(x_train,y_train)

y_pred=reg_lib.predict(x_test)

from sklearn.metrics import r2_score, mean_squared_error

mse=mean_squared_error(y_pred,y_test)
print(mse)

R2=r2_score(y_pred,y_test)
print(R2)

n,p=x_test.shape
print(n,p)

Adj_r2 = 1-(1-R2)*(n-1)/(n-p-1)
print(Adj_r2)

#Take prediction on new data
x_new=np.array([7,95,1,7,6]).reshape(1,5)
x_new=sc.transform(x_new)
y_new=reg_lib.predict(x_new)
print(y_new)