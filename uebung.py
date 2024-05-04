import scipy
import numpy as np
import matplotlib.pyplot as plt
import pandas
import sklearn
from sklearn.model_selection import train_test_split

theta = 1;
theta_0 = 1;
alpha = 0.001;
X_train = 0;
Y_train = 0;
iteration = 10000;

def gradientDescent(theta, theta_0, alpha, X, y, iteration):
    m = len(y)
    loss_history = []

    for i in range(iteration):
        predict = theta_0 + X.dot(theta)
        loss_history.append((1/(2*m))*np.sum(np.square(predict-y)))
        theta = theta - alpha * (1/m)*(X.T.dot(predict-y))
        theta_0 = theta_0 - alpha * (1/m)*np.sum(predict-y)

    return theta, theta_0, loss_history
 


num_samples = 500  # Number of data points
mean = 50  # Mean of the distribution
std_dev = 10  # Standard deviation of the distribution
data = np.random.normal(mean, std_dev, num_samples)




def generate_dataset(num_points, a, b, noise_mean=0, noise_std=0.00002):
    
    x = np.random.rand(num_points)
    noise = np.random.normal(noise_mean, noise_std, num_points)
    y = a * x + b + noise
    
    return x, y

# Parameters
num_points = 1000
a = 2
b = 3
noise_mean = 0
noise_std = 1

# Generate dataset
x, y = generate_dataset(num_points, a, b, noise_mean, noise_std)
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=2)
theta, theta_0, loss = gradientDescent(theta, theta_0, alpha, X_train, Y_train, iteration)

Y_pred = X_test.dot(theta) + theta_0

plt.figure(100)
plt.plot(X_test,Y_pred)
plt.scatter(X_test, Y_test, color='blue', marker='o', s=1)


plt.figure(300)
plt.plot(loss, label='Loss Over Iterations')
plt.legend()

plt.show()