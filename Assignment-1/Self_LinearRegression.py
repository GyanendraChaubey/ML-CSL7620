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

class LinearRegression:

    def __init__(self):
        self.eta=0.01
        self.iterations=1000
        self.thetas=None
        self.e=None
        self.mse=[]
        self.theta_zero=None
    
    def fit(self,x_train,y_train):
        n, f= x_train.shape
        self.thetas=np.zeros(f)
        self.theta_zero=0
        #x_intercept = np.hstack((np.ones((n, 1)), x_train))
        for _ in range(self.iterations):
            pred = np.dot(x_train, self.thetas)+self.theta_zero
            self.e = pred - y_train
            self.mse.append(np.mean(self.e**2))
            grad = np.dot(x_train.T, self.e) / n
            self.thetas -= self.eta * grad
            self.theta_zero -=self.eta*np.mean(self.e)
    
    def loss_curve(self):
        plt.plot(self.mse)
        plt.xlabel('Iteration')
        plt.ylabel('Mean Squared Error (MSE)')
        plt.title('Loss Curve')
        plt.show()
 
    def predict(self, x_test):
            if self.thetas is None:
                raise Exception("Train your model first")
        
            n = x_test.shape[0]
            #x_intercept = np.hstack((np.ones((n, 1)), x_test))
            return np.dot(x_test, self.thetas)+self.theta_zero

reg=LinearRegression()

print("Fitting the model")
reg.fit(x_train,y_train)

print("Predicting the output on test data")
y_pred=reg.predict(x_test)

#print(y_pred)

#Mean square error
mse=np.mean((y_test-y_pred)**2)
print(mse)

#draw loss curve
reg.loss_curve()

#R2 score
def r2_score(y_test, y_pred):
    numerator= np.sum((y_test - y_pred) ** 2)
    denominator = np.sum((y_test - np.mean(y_test)) ** 2)
    return 1 - (numerator / denominator)

print(r2_score(y_test,y_pred))

#Take prediction on new data
x_new=np.array([7,95,1,7,6]).reshape(1,5)
x_new=sc.transform(x_new)
y_new=reg.predict(x_new)
print(y_new)