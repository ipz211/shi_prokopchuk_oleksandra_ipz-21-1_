import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np

# Load dataset
df = pd.read_csv('data_metrics.csv')
df.head()

# Set threshold and make predictions
thresh = 0.5
df['predicted_RF'] = (df.model_RF >= thresh).astype('int')
df['predicted_LR'] = (df.model_LR >= thresh).astype('int')
df.head()

# Define helper functions for confusion matrix components
def find_TP(y_true, y_pred):
    # counts the number of true positives (y_true = 1, y_pred = 1)
    return sum((y_true == 1) & (y_pred == 1))
def find_FN(y_true, y_pred):
    # counts the number of false negatives (y_true = 1, y_pred = 0)
    return sum((y_true == 1) & (y_pred == 0))
def find_FP(y_true, y_pred):
    # counts the number of false positives (y_true = 0, y_pred = 1)
    return sum((y_true == 0) & (y_pred == 1))
def find_TN(y_true, y_pred):
    # counts the number of true negatives (y_true = 0, y_pred = 0)
    return sum((y_true == 0) & (y_pred == 0))
# Testing the helper functions
print('TP:', find_TP(df.actual_label.values, df.predicted_RF.values))
print('FN:', find_FN(df.actual_label.values, df.predicted_RF.values))

print('FP:', find_FP(df.actual_label.values, df.predicted_RF.values))
print('TN:', find_TN(df.actual_label.values, df.predicted_RF.values))

# Confusion matrix implementation
def find_conf_matrix_values(y_true, y_pred):
    # calculate TP, FN, FP, TN
    TP = find_TP(y_true, y_pred)
    FN = find_FN(y_true, y_pred)
    FP = find_FP(y_true, y_pred)
    TN = find_TN(y_true, y_pred)
    return TP, FN, FP, TN
def prokopchuk_confusion_matrix(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return np.array([[TN, FP], [FN, TP]])

# Compare custom confusion matrix with sklearn's
assert np.array_equal(prokopchuk_confusion_matrix(df.actual_label.values, df.predicted_RF.values),
 confusion_matrix(df.actual_label.values,df.predicted_RF.values)), 'my_confusion_matrix() is not correct for RF'
assert np.array_equal(prokopchuk_confusion_matrix(df.actual_label.values, df.predicted_LR.values),
 confusion_matrix(df.actual_label.values, df.predicted_LR.values)), 'my_confusion_matrix() is not correct for LR'

# Calculate accuracy
accuracy_rf = accuracy_score(df.actual_label.values,
df.predicted_RF.values)
print(f'Accuracy (Random Forest): {accuracy_rf}')

def prokopchuk_accuracy_score(y_true, y_pred):
# calculates the fraction of samples
    TP,FN,FP,TN = find_conf_matrix_values(y_true,y_pred)
    return (TP + TN) / (TP + TN + FP + FN)
assert prokopchuk_accuracy_score(df.actual_label.values,
df.predicted_RF.values) == accuracy_score(df.actual_label.values,
df.predicted_RF.values), 'my_accuracy_score failed on RF'

assert prokopchuk_accuracy_score(df.actual_label.values,
df.predicted_LR.values) == accuracy_score(df.actual_label.values,
df.predicted_LR.values), 'my_accuracy_score failed on LR'

print('Accuracy LR: %.3f'%(prokopchuk_accuracy_score(df.actual_label.values,
df.predicted_LR.values)))

from sklearn.metrics import recall_score
recall_score(df.actual_label.values, df.predicted_RF.values)

def prokopchuk_recall_score(y_true, y_pred):
# calculates the fraction of positive samples predicted correctly
    TP,FN,FP,TN = find_conf_matrix_values(y_true,y_pred)
    return TP / (TP + FN) if (TP + FN) != 0 else 0
assert prokopchuk_recall_score(df.actual_label.values,
df.predicted_RF.values) == recall_score(df.actual_label.values,
df.predicted_RF.values), 'prokopchuk_accuracy_score failed on RF'
assert prokopchuk_recall_score(df.actual_label.values,
df.predicted_LR.values) == recall_score(df.actual_label.values,
df.predicted_LR.values), 'prokopchuk_accuracy_score failed on LR'

print('Recall RF: %.3f'%(prokopchuk_recall_score(df.actual_label.values, df.predicted_RF.values)))
print('Recall LR: %.3f'%(prokopchuk_recall_score(df.actual_label.values, df.predicted_LR.values)))

from sklearn.metrics import precision_score
precision_score(df.actual_label.values, df.predicted_RF.values)

def prokopchuk_precision_score(y_true, y_pred):
 # calculates the fraction of predicted positives samples that are actually positive
    TP,FN,FP,TN = find_conf_matrix_values(y_true,y_pred)
    return TP / (TP + FP) if (TP + FP) != 0 else 0

assert prokopchuk_precision_score(df.actual_label.values,
df.predicted_RF.values) == precision_score(df.actual_label.values,
df.predicted_RF.values), 'my_accuracy_score failed on RF'
assert prokopchuk_precision_score(df.actual_label.values,
df.predicted_LR.values) == precision_score(df.actual_label.values,
df.predicted_LR.values), 'my_accuracy_score failed on LR'

print('Precision RF: %.3f'%(prokopchuk_precision_score(df.actual_label.values, df.predicted_RF.values)))
print('Precision LR: %.3f'%(prokopchuk_precision_score(df.actual_label.values, df.predicted_LR.values)))

from sklearn.metrics import f1_score
f1_score(df.actual_label.values, df.predicted_RF.values)

def prokopchuk_f1_score(y_true, y_pred):
    # calculates the F1 score
    recall = prokopchuk_recall_score(y_true,y_pred)
    precision = prokopchuk_precision_score(y_true,y_pred)
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
assert np.isclose(prokopchuk_f1_score(df.actual_label.values, df.predicted_RF.values), f1_score(df.actual_label.values,
df.predicted_RF.values)), 'prokopchuk_f1_score failed on RF'
assert np.isclose(prokopchuk_f1_score(df.actual_label.values, df.predicted_LR.values), f1_score(df.actual_label.values,
df.predicted_LR.values)), 'prokopchuk_f1_score failed on LR'
print('F1 RF: %.3f'%(prokopchuk_f1_score(df.actual_label.values, df.predicted_RF.values)))
print('F1 LR: %.3f'%(prokopchuk_f1_score(df.actual_label.values, df.predicted_LR.values)))
print('scores with threshold = 0.5')
print('Accuracy RF: %.3f'%(prokopchuk_accuracy_score(df.actual_label.values, df.predicted_RF.values)))
print('Recall RF: %.3f'%(prokopchuk_recall_score(df.actual_label.values,
df.predicted_RF.values)))
print('Precision RF: %.3f'%(prokopchuk_precision_score(df.actual_label.values, df.predicted_RF.values)))
print('F1 RF: %.3f'%(prokopchuk_f1_score(df.actual_label.values, df.predicted_RF.values)))

print('')
print('scores with threshold = 0.25')
print('Accuracy RF: %.3f'%(prokopchuk_accuracy_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))
print('Recall RF: %.3f'%(prokopchuk_recall_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))
print('Precision RF: %.3f'%(prokopchuk_precision_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))
print('F1 RF: %.3f'%(prokopchuk_f1_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))

from sklearn.metrics import roc_curve
fpr_RF, tpr_RF, thresholds_RF = roc_curve(df.actual_label.values, df.model_RF.values)
fpr_LR, tpr_LR, thresholds_LR = roc_curve(df.actual_label.values, df.model_LR.values)

import matplotlib.pyplot as plt
plt.plot(fpr_RF, tpr_RF,'r-',label = 'RF')
plt.plot(fpr_LR,tpr_LR,'b-', label= 'LR')
plt.plot([0,1],[0,1],'k-',label='random')
plt.plot([0,0,1,1],[0,1,1,1],'g-',label='perfect')
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

from sklearn.metrics import roc_auc_score
auc_RF = roc_auc_score(df.actual_label.values, df.model_RF.values)
auc_LR = roc_auc_score(df.actual_label.values, df.model_LR.values)
print('AUC RF:%.3f'% auc_RF)
print('AUC LR:%.3f'% auc_LR)

import matplotlib.pyplot as plt
plt.plot(fpr_RF, tpr_RF,'r-',label = 'RF AUC: %.3f'%auc_RF)
plt.plot(fpr_LR,tpr_LR,'b-', label= 'LR AUC: %.3f'%auc_LR)
plt.plot([0,1],[0,1],'k-',label='random')
plt.plot([0,0,1,1],[0,1,1,1],'g-',label='perfect')
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()