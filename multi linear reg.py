import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
df=pd.read_csv("boston.csv")
x=df[['RM','AGE']]
y=df['MEDV']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
mse=mean_squared_error(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print(f"Mean Squared Error(MSE):{mse:.2f}")
print(f"Mean Absolute Error(MAE):{mae:.2f}")
print(f"R-squared (R**2) Score: {r2:.2f}")
print("Accuracy=",model.score(x_test,y_test)*100)
plt.figure(figsize=(10,6))
plt.scatter(y_test,y_test,color='blue',label='Actual Values')
plt.scatter(y_test,y_pred,color='red',label='Predicted values')
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],'k--',lw=2) plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs. Predicted values') plt.legend()
plt.show()
