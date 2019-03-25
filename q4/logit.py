'''
Logistic regression implementation
'''
import numpy as np
import pickle
from metrics import *

class VanillaLogisticRegression:
    def __init__(self, lr=0.01, max_iter=100):
        '''
        A simple implementation of Logit using NumPy
        Default values used from sklearn
        '''
        self.lr = lr
        self.max_iter = max_iter

    def _sigmoid(self, z):
        '''
        sig = 1/(1+e**(-z))
        '''
        return 1 / (1 + np.e**(-z))

    def _loss(self, y, y_hat):
        '''
        vectorised loss
        '''
        # average (Sum of error when label = 1 and error when label = 0)
        logLoss = -np.mean((y * np.log(y_hat)) + ((1 - y) * np.log(1 - y_hat)))
        return logLoss

    def _gradientDescent(self, X, y, theta):
        '''
        Vectorised gradient descent
        X: features
        y: labels
        theta: weights
        '''
        z = np.dot(X, theta)
        h = self._sigmoid(z)
        gradient = np.dot(X.T, (h - y))

        # Average cost derivative for each feature
        gradient /= y.size
        gradient *= self.lr
        theta -= gradient

        return theta

    def fit(self, X, y):
        # Value of z when x == 0 is determined by intercept
        intercept = np.ones((X.shape[0], 1))
        X = np.concatenate((intercept, X), axis=1)

        self.theta = np.zeros(X.shape[1])

        for i in range(self.max_iter):
            self.theta = self._gradientDescent(X, y, self.theta)

    def predict_proba(self, X):
        intercept = np.ones((X.shape[0], 1))
        X = np.concatenate((intercept, X), axis=1)

        return self._sigmoid(np.dot(X, self.theta))


    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs > 0.5).astype(int)



X_train_f = open('data/balanced_train_test/X_train.pickle', 'rb')
X_train = pickle.load(X_train_f)
X_train_f.close()

y_train_f = open('data/balanced_train_test/y_train.pickle', 'rb')
y_train = pickle.load(y_train_f)
y_train_f.close()

X_test_f = open('data/balanced_train_test/X_test_balanced.pickle', 'rb')
X_test = pickle.load(X_test_f)
X_test_f.close()

y_test_f = open('data/balanced_train_test/y_test_balanced.pickle', 'rb')
y_test = pickle.load(y_test_f)
y_test_f.close()

def accuracy(predicted_labels, actual_labels):
    diff = predicted_labels - actual_labels
    return 1.0 - (float(np.count_nonzero(diff)) / len(diff))

clf = VanillaLogisticRegression(max_iter=1000, lr=0.1)
clf.fit(X_train, y_train)
res = clf.predict(X_test)

print(accuracy(res, y_test))

precisionVal = precision(y_test, res)
recallVal = recall(y_test, res)
f1Score = f1_score(y_test, res)

print('precision: ', precisionVal)
print('recall: ', recallVal)
print('f1 score: ', f1Score)
