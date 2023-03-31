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
![image](https://user-images.githubusercontent.com/93434149/229063971-6531e292-48c2-499d-898a-b5485083b82b.png)
### Tail:
![image](https://user-images.githubusercontent.com/93434149/229065093-f79565bb-a686-4ad5-88c7-5e7f1d296220.png)
### Value of X:
![image](https://user-images.githubusercontent.com/93434149/229065264-fdd817d6-7a17-482c-975a-c7afea4bdfc9.png)
### Value of Y:
![image](https://user-images.githubusercontent.com/93434149/229065354-aa17541e-d244-44ee-9f48-313cb1feb3cf.png)
### Predicted value of Y:
![image](https://user-images.githubusercontent.com/93434149/229065573-4f472e28-52d8-42a3-b110-10628d4ce595.png)
### Tested value of Y:
![image](https://user-images.githubusercontent.com/93434149/229065830-10ddc9b4-5ac6-4b88-ab52-b83376473364.png)
### Graph for Training Set:
![image](https://user-images.githubusercontent.com/93434149/229066019-1e792834-fc32-4d7c-a62e-b7338c84b0f9.png)
### Graph for Test Set:
![image](https://user-images.githubusercontent.com/93434149/229066105-4bdc34b8-e8a8-4186-afe1-35c58ef846e1.png)
### Value for MSE, MAE, RMSE:
![image](https://user-images.githubusercontent.com/93434149/229066891-ca2dcb4e-ca21-4bca-89f0-ef0dcb939e5a.png)
## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
