# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and read the dataframe.
2. Assign hours to X and scores to Y.
3. Implement training set and test set of the dataframe.
4. Plot the required graph both for test data and training data.
5. Find the values of MSE , MAE and RMSE.
 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: SOMEASVAR.R
RegisterNumber:  212221230103
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error , mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
X=df.iloc[;,1].values
X
Y=df.iloc[:,1].values
Y
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
Y_pred
Y_test
plt.scatter(X_train,Y_train,color='blue')
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color='black')
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs scores(Test set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE= ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE= ',mae)
rmse=np.sqrt(mse)
print("RMSE =",rmse)
```

## Output:
### Head:
![image](https://user-images.githubusercontent.com/93434149/229068426-b5dae7a5-eb88-42e3-a08d-413c9141d451.png)

### Tail:
![image](https://user-images.githubusercontent.com/93434149/229068496-a2665a6d-817f-4e72-8225-3e4f72556aab.png)

### Value of X:
![image](https://user-images.githubusercontent.com/93434149/229068561-9b4d801f-87f1-4a01-8878-30f6ad23c156.png)

### Value of Y:
![image](https://user-images.githubusercontent.com/93434149/229068599-6eea1043-7def-4030-9d7d-99149575f2ea.png)

### Predicted value of Y:
![image](https://user-images.githubusercontent.com/93434149/229068658-d50f255d-bd3b-4e6c-9a4b-b3730d071b23.png)

### Tested value of Y:
![image](https://user-images.githubusercontent.com/93434149/229068708-f80490f2-2bdd-48f3-bedc-c3a6b6e5407e.png)

### Graph of Training Set:
![image](https://user-images.githubusercontent.com/93434149/229068760-9dd45a11-afc1-45da-a089-06b174873bf4.png)

### Graph of Test Set:
![image](https://user-images.githubusercontent.com/93434149/229068831-8766a5eb-4fe3-4dbf-baea-bfe3f6499223.png)

### Value for MSE, MAE, RMSE:
![image](https://user-images.githubusercontent.com/93434149/229068887-e4cc4833-3dc4-4a03-9330-5962926ecb34.png)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
