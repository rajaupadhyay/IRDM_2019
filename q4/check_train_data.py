import numpy as np
import pickle
import sqlite3
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
from matplotlib import style
style.use('ggplot')


y_pred_proba_f = open('q4/roc_auc_helpers/y_pred_proba_imb.pkl', 'rb')
y_pred_proba = pickle.load(y_pred_proba_f)
y_pred_proba_f.close()
print(len(y_pred_proba))

y_test_f = open('q4/roc_auc_helpers/y_test_imbalanced.pickle', 'rb')
y_test = pickle.load(y_test_f)
y_test_f.close()

# print('log_loss: ', log_loss(y_test, y_pred_proba))

fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.title('ROC Curve (Balanced Train - Imbalanced Test)')
plt.plot(fpr,tpr,label="data, auc="+str(round(auc, 4)), color='navy')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.legend(loc=4)
plt.show()



# X_vals = list(np.linspace(0.00001, 1, 9))
# y_vals = [0.6941514256316456, 0.6675577463338476, 0.669738554554836, 0.6733282011437942, 0.676165598258556, 0.678382734587441, 0.6801818746878227, 0.6816951566537858, 0.6830038846621984]
#
#
# plt.title('Log loss vs learning rate (Imbalanced Dev set)')
# plt.xlabel('Learning rate')
# plt.ylabel('Log loss')
# plt.plot(X_vals, y_vals)
# plt.show()
#
















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
