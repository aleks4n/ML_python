import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

regr = linear_model.LinearRegression()

theta = [1,1,1];
theta_0 = 1;
alpha = 0.001;
X_train = 0
Y_train = 0;
iteration = 10000;
num_points = 100
a = np.array([3, 1.5, 2])
b = 3
noise_mean = 0
noise_std = 1

def generate_dataset(num_points, a, b, noise_mean=0, noise_std=0.00002):
    x1 = np.random.rand(num_points)
    x2 = np.random.rand(num_points)
    x3 = np.random.rand(num_points)
    x = np.stack((x1, x2, x3), axis=-1)
    noise = np.random.normal(noise_mean, noise_std, num_points)
    y = a.dot(x.T) + b + noise
    return x, y

def gradientDescent(theta, theta_0, alpha, X, y, iteration):
    m = len(y)
    loss_history = []

    for i in range(iteration):
        predict = theta_0 + X.dot(theta)
        loss_history.append((1/(2*m))*np.sum(np.square(predict-y)))
        theta = theta - alpha * (1/m)*(X.T.dot(predict-y))
        theta_0 = theta_0 - alpha * (1/m)*np.sum(predict-y)

    return theta, theta_0, loss_history



def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    print(f"MAE: {mae}")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"R2: {r2}")


# Generate dataset
x, y = generate_dataset(num_points, a, b, noise_mean, noise_std)
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3, random_state=2)
theta, theta_0, loss = gradientDescent(theta, theta_0, alpha, X_train, Y_train, iteration)


#Comparison with sklearn
regr.fit(X_train, Y_train)
sklearnpredict = regr.predict(X_test)

Y_pred = X_test.dot(theta) + theta_0

evaluate_model(Y_test, Y_pred)

plt.figure(300)
plt.plot(loss, label='Loss Over Iterations')
plt.legend()



fig1 = plt.figure(200)
plt.plot(Y_pred,label='Predicted')
plt.plot(Y_test,label='Actual')
plt.plot(sklearnpredict,label='Sklearn')
plt.legend()

plt.show()
