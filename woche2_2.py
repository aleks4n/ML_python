import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np

df = pd.read_csv('/Users/aliihsangungoren/Downloads/Realestate.csv')
theta = [1]*(len(df.columns)-2);
theta_0 = 1;


def sigmoid(z):
    return 1 / (1 + np.exp(-z))



def logistic_regressionn(X, y, learning_rate=0.01, num_iterations=50, lambda_=0.01):
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

    X = df.iloc[:, 1:7]
    Y = df.iloc[:, 7]

    mean_price = np.mean(Y)
    Y = pd.Series(np.where(Y < mean_price, 1, 0))

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


# Create a scaler object
scaler = StandardScaler()

# Fit on training set only.
scaler.fit(X_train)

# NOTE: Needed to transform the date becuase it was way too big and gave overflow errors, it was not said in the scrpt
# but couldnt find any other way to fix it
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Reshape Y_train to be a 2D array, which is required for the StandardScaler
Y_train = np.array(Y_train).reshape(-1, 1)

# Create a scaler object for Y
scaler_Y = StandardScaler()
scaler_Y.fit(Y_train)

# Scale Y_train and convert it back to a 1D array
Y_train = scaler_Y.transform(Y_train).ravel()


# Now 'data' is a list of lists, where each inner list represents a row from the first 7 columns of the CSV file
theta, theta_0= logistic_regressionn(X_train, Y_train)



Y_pred = X_test.dot(theta) + theta_0
Y_pred = np.array(Y_pred).reshape(-1, 1)
Y_pred = scaler_Y.inverse_transform(Y_pred)
fig1 = plt.figure(200)
plt.plot(Y_pred,label='Predicted')
plt.plot(Y_test.values,label='Actual') #Giving Y_test.values instead of Y_test because Y_test is a pandas series and it gives the indices

#plt.plot(h, color='red', label='Predicted')



X5_list = df.iloc[:, 5].values.tolist()

print(Y_test)

#plt.legend()

plt.show()