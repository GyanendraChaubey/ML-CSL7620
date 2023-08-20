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

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

#print the shape of x_train, x_test, y_train, y_test
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

class LinearRegression:

    def __init__(self):
        self.eta=0.0001
        self.iterations=1500000
        self.thetas=None
        self.errors=[]
    
    def fit(self,x_train,y_train):
        n, f= x_train.shape
        self.thetas=np.zeros(f + 1)
        x_intercept = np.hstack((np.ones((n, 1)), x_train))
        for _ in range(self.iterations):
            pred = np.dot(x_intercept, self.thetas)
            e = pred - y_train
            self.errors.append(e)
            grad = np.dot(x_intercept.T, e) / n
            self.thetas -= self.eta * grad
    
    def predict(self, x_test):
            if self.thetas is None:
                raise Exception("Train your model first")
        
            n = x_test.shape[0]
            x_intercept = np.hstack((np.ones((n, 1)), x_test))
            return np.dot(x_intercept, self.thetas)
    
    def loss_curve(self):
        pass


reg=LinearRegression()

reg.fit(x_train,y_train)

y_pred=reg.predict(x_test)

print(y_pred)

#Mean square error
mse=np.mean((y_test-y_pred)**2)
print(mse)

#R2 score
def r2_score(y_test, y_pred):
    numerator= np.sum((y_test - y_pred) ** 2)
    denominator = np.sum((y_test - np.mean(y_test)) ** 2)
    return 1 - (numerator / denominator)

print(r2_score(y_test,y_pred))

reg.loss_curve()


#Take prediction no new data
x_new=np.array([7,95,1,7,6]).reshape(1,5)
y_new=reg.predict(x_new)
print(y_new)