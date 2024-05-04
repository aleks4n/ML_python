import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model

num_variables = 3
theta = [[1],[1],[1]];
theta_0 = 1;
alpha = 0.002;
test_size = 0.2
num_points = 100
iteration = 1000;
X_train = np.empty((num_variables,int(num_points*(1-test_size))));
X_test = np.empty((num_variables,int(num_points*test_size)));
Y_test = np.empty((num_variables,int(num_points*test_size)));
Y_train = np.empty((num_variables,int(num_points*(1-test_size))));


def generate_dataset(num_points, a, b, num_variables, noise_mean=0, noise_std=0.02):
    # Generate random values for x
    matrix = np.empty((num_variables,num_points))
    noisematrix = np.empty((num_variables,num_points))
    x = np.array([])
    for i in range(matrix.shape[0]):
        matrix[i] = np.random.rand(num_points)
        noisematrix[i] = np.random.normal(noise_mean, noise_std, num_points)
        #x.np.append(np.random.rand(num_points))
    
    # Generate noise
    
    # Calculate y using the linear relationship Y = aX + b + noise
    y = a * matrix + b + noisematrix
    
    return matrix, y

def gradientDescent(theta, theta_0, num_variables, alpha, X, y, iteration):
    num = y.shape[1]
    loss_history = []
    predict = [[],[],[]]
    for m in range(num_variables):
        for i in range(iteration):
            predict[m] = theta_0 + X[m] * theta[m]
            loss_history.append((1/(2*num))*np.sum(np.square(predict[m]-y[m])))
            theta[m] = theta[m] - alpha * (1/num)*(X[m].T.dot(predict[m]-y[m]))
            theta_0 = theta_0 - alpha * (1/num)*np.sum(predict[m]-y[m])
    

    return theta, theta_0, loss_history



a = 2
b = 3
noise_mean = 0
noise_std = 1


x, y = generate_dataset(num_points, a, b, num_variables, noise_mean, noise_std)

for i in range(num_variables):
    X_train[i], X_test[i], Y_train[i], Y_test[i] = train_test_split(x[i], y[i], test_size=0.2, random_state=2)

theta, theta_0, loss = gradientDescent(theta, theta_0, num_variables, alpha, X_train, Y_train, iteration)

#Y_pred = theta.T.dot(X_test) + theta_0

Y_pred = theta[0]*X_test[0] + theta[1]*X_test[1] + theta[2]*X_test[2] + theta_0

print(theta)
#plt.plot(X_test[0],Y_pred[0],label='y1', color='blue')
#plt.plot(X_test[1],Y_pred[1],label='y2',color='red')
#plt.plot(X_test[2],Y_pred[2],label='y3',color='black')
#plt.scatter(X_test[0], Y_test[0], color='blue', marker='o', s=1)
#plt.scatter(X_test[1], Y_test[1], color='red', marker='o', s=1)
#plt.scatter(X_test[2], Y_test[2], color='black', marker='o', s=1)
plt.plot(Y_pred)
plt.plot(Y_test)
#plt.legend() 
#plt.grid(True)
plt.show()

