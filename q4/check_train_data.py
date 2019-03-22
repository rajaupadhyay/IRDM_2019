import numpy as np
import pickle
import sqlite3
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split

X_train_f = open('X_train.pickle', 'rb')
X_train = pickle.load(X_train_f)
X_train_f.close()

y_train_f = open('y_train.pickle', 'rb')
y_train = pickle.load(y_train_f)
y_train_f.close()


X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=58)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)


y_pred = logreg.predict(X_test)
print(y_pred[:50])
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
