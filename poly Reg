import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
df=pd.read_csv(r"boston.csv")
x=df[['RM']]
y=df['MEDV']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
poly=PolynomialFeatures(degree=3)
x_train_poly=poly.fit_transform(x_train)
x_test_poly=poly.transform(x_test)
model=LinearRegression()
model.fit(x_train_poly,y_train)
y_pred=model.predict(x_test_poly)
mse=mean_squared_error(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print(f"Mean Squared Error(MSE): {mse:.2f}")
print(f"Mean Absolute Error(MAE): {mae:.2f}")
print(f"R-square(R**2) score: {r2:.2f}")
plt.figure(figsize=(10,6))
x_range=np.linspace(x_test.min(),x_test.max(),100).reshape(-1,1) x_range_poly=poly.transform(x_range)
y_range=model.predict(x_range_poly)
plt.scatter(x_test,y_test,color='blue',label='Actual values')
plt.scatter(x_range,y_range,color='red',label='Polynomial Regression') plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('Polynomial Regression') plt.legend()
plt.show()
