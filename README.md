# Exercise_Linear_Regression
## Normal Equation
- [Part 0: Normal Equation(Sklearn)](https://github.com/Seriphap/Exercise_Linear_Regression/edit/main/README.md#part-0-normal-equationsklearn)

## Machine Learning: Gradient Descent 
- [Part 1: Batch Gradient Descent](https://github.com/Seriphap/Exercise_Linear_Regression/edit/main/README.md#part-1-batch-gradient-descent)
- [Part 2: Stochastic Gradient Descent](#part-2-stochastic-gradient-descent)
- [Part 3: Mini-Batch Gradient Descent](#part-3-mini-batch-gradient-descent)
<br>
<br>

## Part 0: Normal Equation(Sklearn) 
```python
import numpy as np
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error
x = np.array([[1,0],
              [1,2],
              [1,3]])
y = np.array([1,1,4])

# Create an instance (object) of the LinearRegression class
lin_reg = LinearRegression()  

# Train (fit) the model using the input features x and target y
lin_reg.fit(x, y)             
print(f'Intercept: {lin_reg.intercept_:.2f}')
print('Coefficient:', [float(f'{coef:.2f}') for coef in lin_reg.coef_])

#prediction
y_p = lin_reg.predict(x)
mse = mean_squared_error(y, y_p)
print(f"MSE = {mse:.2f}")
```

## Part 1: Batch Gradient Descent
### Cost Function: Half Mean Square Error
- Minimize Half MSE
<img src="https://github.com/user-attachments/assets/4825ea56-626c-45d7-9d35-f5f45d98d45e" width="20%"> 

```python
def cost_function(X, y, theta):
    N = len(y)
    Hypothesis = np.dot(X, theta)
    cost = (1/(2*N)) * np.sum(np.square(Hypothesis - y))
    return cost
```


## Part 2: Stochastic Gradient Descent
## Part 3: Mini-Batch Gradient Descent
