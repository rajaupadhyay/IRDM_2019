'''
Precision, Recall and F1 metrics
'''
import numpy as np

def getCMValues(y, y_pred):
    zippedY = list(zip(y, y_pred))
    truePositives = sum([1 for y_tuple in zippedY if y_tuple[0] == 1 and y_tuple[1] == 1])
    falsePositives = sum(y_pred) - truePositives
    falseNegatives = sum([1 for y_tuple in zippedY if y_tuple[0] == 1 and y_tuple[1] == 0])

    return truePositives, falsePositives, falseNegatives

def precision(y, y_pred):
    tp, fp, _ = getCMValues(y, y_pred)
    return tp/(tp+fp)

def recall(y, y_pred):
    tp, _, fn = getCMValues(y, y_pred)
    return tp/(tp+fn)

def f1_score(y, y_pred):
    precisionVal = precision(y, y_pred)
    recallVal = recall(y, y_pred)
    f1 = (2 * precisionVal * recallVal)/(precisionVal + recallVal)
    return f1
