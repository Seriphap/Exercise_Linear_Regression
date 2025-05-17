# Exercise_Linear_Regression
## Normal Equation
- [Part 0: Normal Equation(Sklearn)](#part-0-normal-equationsklearn)

## Machine Learning: Gradient Descent 
- [Part 1: Batch Gradient Descent](#part-1-batch-gradient-descent)
- [Part 2: Stochastic Gradient Descent](#part-2-stochastic-gradient-descent)
- [Part 3: Mini-Batch Gradient Descent](#part-3-mini-batch-gradient-descent)
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
- Minimize half-MSE
  
<img src="https://github.com/user-attachments/assets/4825ea56-626c-45d7-9d35-f5f45d98d45e" width="20%"> 

```python
def cost_function(X, y, theta):
    N = len(y)
    Hypothesis = np.dot(X, theta)
    cost = (1/(2*N)) * np.sum(np.square(Hypothesis - y))
    return cost
```

### Gradient Descent Function
- Fine theta
  
  <img src="https://github.com/user-attachments/assets/8598baca-53d0-497c-8443-74c6467c31af" width="40%"> 

```python
def gradient_descent(X, y, theta, learning_rate, iterations, ep):
    N = len(y)
    cost_history = []
    for i in range(iterations):
        Hypothesis = np.dot(X, theta)
        theta = theta - (learning_rate / N) * np.dot(X.transpose(), (Hypothesis - y))
        cost_history.append(cost_function(X, y, theta))
        if i > 0 and np.abs(cost_history[i] - cost_history[i - 1]) <= ep:
          return theta, cost_history, i+1
    return theta, cost_history, iterations
```

### Training Process
- 1. Input data for X and y
```python
X = np.array([[1,0],
              [1,2],
              [1,3]])
y = np.array([1,1,4])
```
<br>

- 2. Define theta for first iteration 
```python
theta = np.array([0.1,0.1]) 
```
<br>

- 3. Define other parameters    
```python
learning_rate = 0.01
iterations = 10000
limited_iterations = iterations
ep = 0.0000000001
```
<br>

- 4. Perform gradient descent function (gradient descent funtion will calculate theta and call cost function to update half-MSE)
```python
theta, cost_history, actual_iterations = gradient_descent(X, y, theta, learning_rate, iterations, ep)
```
<br>

- 5. Loop utill converge (delta of cost function <= ep)/ Loop utill iterations setup
```python
if i > 0 and np.abs(cost_history[i] - cost_history[i - 1]) <= ep:
          return theta, cost_history, i+1
```
<br>

- 6. Print result
```python
print("Coefficients :", theta)
print("Iterations:", str(actual_iterations),"/",str(limited_iterations))
print("half-mean square error :", cost_history[-1])
```
<br>

## Part 2: Stochastic Gradient Descent
### Cost Function: Half Mean Square Error
- Minimize half-MSE (N=1)
  
<img src="https://github.com/user-attachments/assets/4825ea56-626c-45d7-9d35-f5f45d98d45e" width="20%"> 

```python
def cost_function(X, y, theta):
    N = 1
    Hypothesis = np.dot(X, theta)
    cost = (1/(2*N)) * np.sum(np.square(Hypothesis - y)) 
    return cost
```

### Gradient Descent Function (N=1)
- Fine theta
  
  <img src="https://github.com/user-attachments/assets/8598baca-53d0-497c-8443-74c6467c31af" width="40%"> 

```python
def stochastic_gradient_descent(X, y, theta, learning_rate, iterations, ep):
    N = len(y)
    cost_history = []
    for i in range(iterations):
      for j in range(N):
        x_j = X[j]  # Single training example
        y_j = y[j]  # Single target value
        Hypothesis = np.dot(x_j, theta)
        theta = theta - (learning_rate / 1) * np.dot(x_j.transpose(), (Hypothesis - y_j))
      cost_history.append(cost_function(X, y, theta))
      if i > 0 and np.abs(cost_history[i] - cost_history[i - 1]) <= ep:
        return theta, cost_history, i+1
    return theta, cost_history, iterations
```

### Training Process
- 1. Input data for X and y (need to shuffle data first)
```python
X = np.array([[1,0],
              [1,2],
              [1,3]])
y = np.array([1,1,4])
#shuffle data
Random_row=np.random.permutation(X.shape[0])
X=X[Random_row]
y=y[Random_row]
```
<br>

- 2. Define theta for first iteration 
```python
theta = np.array([0.1,0.1]) 
```
<br>

- 3. Define other parameters    
```python
learning_rate = 0.01
iterations = 10000
limited_iterations = iterations
ep = 0.0000000001
```
<br>

- 4. Perform stochastic gradient descent function (gradient descent funtion will calculate theta and call cost function to update half-MSE)
```python
theta, cost_history, actual_iterations = stochastic_gradient_descent(X, y, theta, learning_rate, iterations, ep)
```
<br>

- 5. Loop utill converge (delta of cost function <= ep)/ Loop utill iterations setup
```python
if i > 0 and np.abs(cost_history[i] - cost_history[i - 1]) <= ep:
          return theta, cost_history, i+1
```
<br>

- 6. Print result
```python
print("Coefficients :", theta)
print("Iterations:", str(actual_iterations),"/",str(limited_iterations))
print("half-mean square error :", cost_history[-1])
```
<br>


## Part 3: Mini-Batch Gradient Descent
### Cost Function: Half Mean Square Error
- Minimize half-MSE (N=Batch Size(R))
  
<img src="https://github.com/user-attachments/assets/4825ea56-626c-45d7-9d35-f5f45d98d45e" width="20%"> 

```python
def cost_function(X, y, theta,R):
    Hypothesis = np.dot(X, theta)
    cost = (1/(2*R)) * np.sum(np.square(Hypothesis - y))
    return cost
```

### Gradient Descent Function (N=Batch Size(R))
- Fine theta
  
  <img src="https://github.com/user-attachments/assets/8598baca-53d0-497c-8443-74c6467c31af" width="40%"> 

```python
def mini_batch_gradient_descent(X, y, theta, learning_rate, iterations, ep, R):
    N = len(y)
    cost_history = []
    for i in range(iterations):
      for j in range(0,N,R): # Start=0, Stop=N(exclude), Step=R
        x_batch = X[j:j+R]  # Select batch of data
        y_batch = y[j:j+R]  # Select batch of target value
        Hypothesis = np.dot(x_batch, theta)
        theta = theta - (learning_rate / R) * np.dot(x_batch.transpose(), (Hypothesis - y_batch))
      cost_history.append(cost_function(X, y, theta,R))
      if i > 0 and np.abs(cost_history[i] - cost_history[i - 1]) <= ep:
        return theta, cost_history, i+1
    return theta, cost_history, iterations
```

### Training Process
- 1. Input data for X and y (need to shuffle data first)
```python
X = np.array([[1,0],
              [1,2],
              [1,3]])
y = np.array([1,1,4])
#shuffle data
Random_row=np.random.permutation(X.shape[0])
X=X[Random_row]
y=y[Random_row]
```
<br>

- 2. Define theta for first iteration 
```python
theta = np.array([0.1,0.1]) 
```
<br>

- 3. Define other parameters    
```python
learning_rate = 0.01
iterations = 10000
limited_iterations = iterations
ep = 0.0000000001
R = 2 #Mini-batch size (choose R from N)
```
<br>

- 4. Perform mini-batch gradient descent function (gradient descent funtion will calculate theta and call cost function to update half-MSE)
```python
theta, cost_history, actual_iterations = mini_batch_gradient_descent(X, y, theta, learning_rate, iterations, ep, R)
```
<br>

- 5. Loop utill converge (delta of cost function <= ep)/ Loop utill iterations setup
```python
if i > 0 and np.abs(cost_history[i] - cost_history[i - 1]) <= ep:
          return theta, cost_history, i+1
```
<br>

- 6. Print result
```python
print("Coefficients :", theta)
print("Iterations:", str(actual_iterations),"/",str(limited_iterations))
print("half-mean square error :", cost_history[-1])
```
<br>
