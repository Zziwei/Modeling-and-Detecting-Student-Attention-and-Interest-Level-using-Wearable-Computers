import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm
import pandas as pd
import os
from sklearn.model_selection import LeavePOut
from sklearn.model_selection import KFold
from sklearn import metrics


train_raw = pd.read_csv('train.csv', header=None)
train = train_raw.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 48, 49, 50, 51, 52, 53, 54, 55]]
test_rand_raw = pd.read_csv('test-rand.csv', header=None)
test_rand = test_rand_raw.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 48, 49, 50, 51, 52, 53, 54, 55]]
test_writing_raw = pd.read_csv('test-writing.csv', header=None)
test_writing = test_writing_raw.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 48, 49, 50, 51, 52, 53, 54, 55]]

# train = train_raw.iloc[:, [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 48, 49, 50, 51, 52, 53]]
# test_rand_raw = pd.read_csv('test-rand.csv', header=None)
# test_rand = test_rand_raw.iloc[:, [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 48, 49, 50, 51, 52, 53]]
# test_writing_raw = pd.read_csv('test-writing.csv', header=None)
# test_writing = test_writing_raw.iloc[:, [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 48, 49, 50, 51, 52, 53]]

# fit the model
clf = svm.OneClassSVM(nu=0.01, kernel='rbf', gamma=0.01)
clf.fit(train)
y_pred_train = clf.predict(train)
test_frame = [test_rand, test_writing]
test = pd.concat(test_frame)
y_pred_test = clf.predict(test)
y_exp_list = []
for i in range(len(test_rand)):
    y_exp_list.append(-1)
for i in range(len(test_writing)):
    y_exp_list.append(1)
y_exp_test = pd.DataFrame(y_exp_list)
# n_error_train = y_pred_train[y_pred_train == -1].size / float(y_pred_train.size)
# n_error_test = y_pred_test[y_pred_test == -1].size / float(y_pred_test.size)
# n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size / float(y_pred_outliers.size)
# print 'error in train set: ' + str(n_error_train)
# print 'error in test set: ' + str(n_error_test)
# print 'error in outlier set: ' + str(n_error_outliers)
print(metrics.confusion_matrix(y_exp_test, y_pred_test))
print (metrics.classification_report(y_exp_test, y_pred_test, digits=5))


# clf = svm.OneClassSVM(nu=0.01, kernel='rbf', gamma=0.01)
# lpo = LeavePOut(p=100)
# te_error_list = []
# outlier_error_list = []
# x = range(640)
# kf = KFold(n_splits=100)
# for tr, te in kf.split(x):
#     clf.fit(train.loc[tr])
#     y_pred_te = clf.predict(train.loc[te])
#     n_error_te = y_pred_te[y_pred_te == -1].size / float(y_pred_te.size)
#     print 'error in test set: ' + str(n_error_te)
#     te_error_list.append(n_error_te)
#     y_pred_outliers = clf.predict(test_rand)
#     n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size / float(y_pred_outliers.size)
#     print 'error in outlier set: ' + str(n_error_outliers)
#     outlier_error_list.append(n_error_outliers)
# print 'test error mean: ' + str(np.mean(te_error_list))
# print 'test error std: ' + str(np.std(te_error_list))


# for filename in os.listdir('writing_features_everyone_wed'):
#     fileid = filename.split('.')[0]
#     print fileid
#     current_output_file = open('writing_results_everyone_wed/' + fileid + '.txt', 'a')
#     current_data = pd.read_csv('writing_features_everyone_wed/' + filename, header=None)
#     current_input = current_data.iloc[:, range(16)]
#     current_output = clf.predict(current_input)
#     for i in range(len(current_output)):
#         current_output_file.write(str(current_output[i]) + ' ' + str(current_data.iloc[i, 16]) + ' '
#                                   + str(current_data.iloc[i, 17]) + '\r\n')
#     current_output_file.close()