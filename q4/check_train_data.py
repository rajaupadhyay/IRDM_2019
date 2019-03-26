import numpy as np
import pickle
import sqlite3
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt



# y_pred_proba_f = open('y_pred_proba.pkl', 'rb')
# y_pred_proba = pickle.load(y_pred_proba_f)
# y_pred_proba_f.close()
#
#
# y_test_f = open('data/imbalanced_train_test/y_test.pickle', 'rb')
# y_test = pickle.load(y_test_f)
# y_test_f.close()
#
# fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
# auc = metrics.roc_auc_score(y_test, y_pred_proba)
# plt.title('ROC Curve (Balanced Train - Imbalanced Test)')
# plt.plot(fpr,tpr,label="data, auc="+str(round(auc, 4)), color='navy')
# plt.plot([0, 1], [0, 1], color='red', linestyle='--')
# plt.legend(loc=4)
# plt.show()


# X_train_f = open('data/imbalanced_train_test/X_train.pickle', 'rb')
# X_train = pickle.load(X_train_f)
# X_train_f.close()
#
# y_train_f = open('data/imbalanced_train_test/y_train.pickle', 'rb')
# y_train = pickle.load(y_train_f)
# y_train_f.close()
#
# X_test_f = open('data/imbalanced_train_test/X_test.pickle', 'rb')
# X_test = pickle.load(X_test_f)
# X_test_f.close()
#
# y_test_f = open('data/imbalanced_train_test/y_test.pickle', 'rb')
# y_test = pickle.load(y_test_f)
# y_test_f.close()
#
# logreg = LogisticRegression()
# logreg.fit(X_train, y_train)
#
# y_pred = logreg.predict(X_test)
# print('predictions ', y_pred[:10])
# print('actual ', y_test[:10])
# print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
#
# number_of_ones = np.count_nonzero(y_test)
#
# ctr = 0
#
# for itx in range(len(y_pred)):
#     if y_test[itx] == 1:
#         if y_pred[itx] == 1:
#             ctr += 1
#
# print(ctr/number_of_ones)
#
#
# print(precision_recall_fscore_support(y_test, y_pred, average=None, labels=[0, 1]))


# conf_mat = confusion_matrix(y_test, y_pred)
#
# plt.clf()
# plt.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.tab20c)
# classNames = ['Negative','Positive']
# plt.ylabel('True label')
# plt.xlabel('Predicted label')
# tick_marks = np.arange(len(classNames))
# plt.xticks(tick_marks, classNames, rotation=45)
# plt.yticks(tick_marks, classNames)
# s = [['TN','FP'], ['FN', 'TP']]
# for i in range(2):
#     for j in range(2):
#         plt.text(j,i, str(s[i][j])+" = "+str(conf_mat[i][j]))
#
# plt.tight_layout()
# plt.show()
