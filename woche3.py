import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#generate dataset
np.random.seed(0)
samples = 100
noise = np.random.normal(-2,3,samples)
X = 3 + np.random.uniform(0, 1, 100)
y = 3 * X + noise

#split dataset
print(X.shape)
start_rows = np.random.randint(10, 30)
#end_rows = int(0.8 * samples)
end_rows = 100 - start_rows


X_train = X[start_rows:end_rows]
X_test = np.concatenate([X[end_rows:,],X[: start_rows,]]) 

Y_train = y[start_rows:end_rows]
Y_test = np.concatenate([y[end_rows:,],y[: start_rows,]]) 


#Y_train = y[start_rows:end_rows]
print("train", X_train.shape)
print("test", X_test.shape)

class LinearRegression():
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.theta_1 = None
        self.theta_0 = None

        self.theta_1_list = []
        self.theta_0_list = []
        self.mse_list = []

    def fit(self, X, y):
        #number of samples and features
        
        n_samples, n_features = X.shape
        self.theta_1 = np.zeros(n_features)
        self.theta_0 = 0

        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.theta_1) + self.theta_0
            d_theta_1 = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            d_theta_0 = (1 / n_samples) * np.sum(y_predicted - y)
            self.theta_1 -= self.lr * d_theta_1
            self.theta_0 -= self.lr * d_theta_0

            self.mse_list.append(np.mean((y_predicted - y) ** 2))
            self.theta_0_list.append(self.theta_0)
            self.theta_1_list.append(self.theta_1)
    def predict(self, X):
        return np.dot(X, self.theta_1) + self.theta_0
    

X_test = X_test.reshape(-1, 1)    
X_train = X_train.reshape(-1, 1)

#Model initialization
regressor = LinearRegression(lr=0.001, n_iters=1000)
regressor.fit(X_train, Y_train)
predictions = regressor.predict(X_test)

#evaluate the model
mse = np.mean((Y_test - predictions) ** 2)
print("MSE:", mse)

#visualize the model
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.scatter(X_train, Y_train, color='blue') 
plt.plot(X_test, predictions, color='red')
plt.xlabel('X')
plt.ylabel('y')


#Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_train, Y_train, color='blue')
ax.scatter(X_test, predictions, color='red')
plt.show()

mse_values = np.zeros((samples, samples))
theta_1_val = np.linspace(-1, 4, samples)
theta_1_grid, theta_0_grid = np.meshgrid(theta_1_val, regressor.theta_0_list)
for i in range(theta_1_grid.shape[0]):
    for j in range(theta_1_grid.shape[1]):
        y_pred = theta_1_grid[i, j] * X_train + theta_0_grid[i, j]
        mse_values[i, j] = ((Y_train.reshape(-1,1))-y_pred).mean


ax.plot_surface(theta_1_grid, theta_0_grid, mse_values, cmap='viridis', alpha=0.5)
ax.plot([theta_1[0] for theta_1 in regressor.theta_1_hist], regressor.theta_0_hist, regressor.mse_hist, color='red', marker='*', linewidth=2)