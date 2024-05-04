#You add a regularization term Teta in order to avoid under/overfitting (It is a way to penalize the weights)
# in the exam you might need to calculate linear regression with hand
# Logistic Regression
# False negative, False positive, True negative, True positive
# classifying a patient as non-patient when he is a patient is a false negative
#sigmoid function is used to classify the output of the linear regression
#1/(1+np.exp(-x))
#Logistic regression is used to classify the output of the linear regression
#We are not able to use linear regression in logistic loss function
#The loss function of logistic regression is the cross entropy loss function: -y*log(y_pred)-(1-y)*log(1-y_pred)
#The y is either 0 or 1
# use log loss function to calculate the loss
# plot the convergence of the parameters
# do the splitting without train_test_split
# Also plot the testing loss


# Use only failure and nonfuilare dingens
# Remove the column with multivariate dingens
import pandas as pd
import numpy as np
