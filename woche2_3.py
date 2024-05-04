import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np


df = pd.read_csv('/Users/aliihsangungoren/Downloads/predictive_maintenance.csv')
theta = [1]*(len(df.columns)-2);
theta_0 = 1;

df = df.drop('Product ID', axis=1)
df = df.drop('Type', axis=1)
df = df.drop('Failure Type', axis=1)

scaler = StandardScaler()
columns_to_standardize = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]']
df[columns_to_standardize] = scaler.fit_transform(df[columns_to_standardize])


def sigmoid(z):
    return 1 / (1 + np.exp(-z))



def logistic_regressionn(X, y, learning_rate=0.1, num_iterations=50, lambda_=0.01):
    m, n = X.shape
    theta = np.zeros(n)
    theta_0 = 0
    
    for i in range(num_iterations):
        z = np.dot(X, theta) + theta_0
        h = sigmoid(z)
        gradient = np.dot(X.T, (h - y)) / m + (lambda_ * theta**2 / m)
        theta -= learning_rate * gradient
        #theta_0 -= learning_rate * np.sum(h - y) / m

    return theta, theta_0


def log_loss(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def split(df):
    # Define the proportion of data to be used for testing
    test_ratio = 0.2

    # Shuffle the DataFrame
    #df = df.sample(frac=1, random_state=42)
   
    # Calculate the index at which to split the DataFrame
    split_idx = int(len(df) * (1 - test_ratio))

    X = df.loc[:, ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]']]
    Y = df.loc[:,'Target']



    # Split the DataFrame
    X_train = X.iloc[:split_idx]
    #X_train = X_train.values.tolist()
    X_test = X.iloc[split_idx:]
    #X_test = X_test.values.tolist()

    Y_train = Y.iloc[:split_idx]
    #Y_train = Y_train.values.tolist()
    Y_test = Y.iloc[split_idx:]
    #Y_test = Y_test.values.tolist()

    # Now df_train and df_test are your training and testing sets, respectively
    return X_train, X_test, Y_train, Y_test

X_train, X_test, Y_train, Y_test = split(df)



from sklearn.linear_model import LogisticRegression

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

def plot_logistic_regression(X_train, Y_train):
    # Fit logistic regression model
    model = LogisticRegression()
    model.fit(X_train, Y_train)

    # Create a mesh to plot in
    h = .02  # step size in the mesh
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Predict the function value for the whole grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train, cmap=plt.cm.Spectral)
    plt.show()


#plot_logistic_regression(X_train.values, Y_train.values)



#print(X_train.values.tolist())
#print(Y_train)
theta, theta_0 = logistic_regressionn(X_train.values, Y_train.values)


print(X_train.values.shape)
print(Y_train.values.shape)

scaler_Y = StandardScaler()
scaler_Y.fit(Y_train.values.reshape(-1, 1))

Y_pred = X_test.dot(theta) + theta_0
Y_pred = np.array(Y_pred).reshape(-1, 1)
Y_pred = scaler_Y.inverse_transform(Y_pred)
fig1 = plt.figure(200)
plt.plot(Y_pred,label='Predicted')
plt.plot(Y_test.values,label='Actual')
plt.show()