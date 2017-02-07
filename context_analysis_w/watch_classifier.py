from imblearn.over_sampling import SMOTE
from sklearn import metrics
import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import precision_recall_fscore_support
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import LeavePOut
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import copy
import Queue
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import StratifiedKFold


c = 'i'
min_list = []
if c == 'd':
    min_list = [0, 8, 9, 12]
else:
    min_list = [3, 4, 7, 10, 11, 13]


# fit a CART model to the data
data = pd.read_csv('input_' + c + '_2_hrv_c.csv', header=None)
decisionTree = DecisionTreeClassifier()
knnClf = KNeighborsClassifier(n_neighbors=3)    # default:k = 5,defined by yourself:KNeighborsClassifier(n_neighbors=10)
svc = svm.SVC(kernel='linear', C=1) #(kernel='linear', C=1)   #(kernel='rbf') #(kernel='poly', degree=5)
naive_bayes = GaussianNB()
rand_forrest = RandomForestClassifier(n_estimators=25)
lpo = LeavePOut(p=3)
X_raw = data.iloc[:, :data.shape[1] - 1]
y = data.iloc[:, data.shape[1] - 1]
# X = X_raw.iloc[:, best_features_list]

# lsvc = LinearSVC(C=0.7, penalty="l1", dual=False).fit(X_old, y)
# model = SelectFromModel(lsvc, prefit=True)
# X = model.transform(X_old)
# print X.shape

# model_name_list = ['decision tree', 'knn', 'svm', 'naive bayes'] #, 'random forrest']
# model_list = [decisionTree, knnClf, svc, naive_bayes]# , rand_forrest]
model_name_list = ['knn'] #, 'random forrest']
model_list = [svc]# , rand_forrest]

from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedShuffleSplit
ss = StratifiedShuffleSplit(n_splits=150, test_size=0.25, random_state=0)
# for train_index, test_index in ss.split(X_raw, y):
#     print("%s %s" % (train_index, test_index))


best_features_list = [[3, 38, 43, 45, 49, 67, 80, 81, 86, 87, 98, 99, 107, 109]]#, [11, 15, 44, 46, 72, 51, 59], [5, 6, 17, 25]]
X_new = SelectKBest(mutual_info_classif, k=30).fit_transform(X_raw, y)
# this part is just computing scores with different models using given features list
scores = []
for i in range(len(model_list)):
    scores.append(cross_val_score(model_list[i], X_raw.iloc[:, best_features_list[i]], y, cv=lpo))

for i in range(len(scores)):
    print("%s Accuracy: %0.10f (+/- %0.4f)" % (model_name_list[i], scores[i].mean(), scores[i].std()))
    # print scores
    print
    # result_file.write(str(scores[i].mean()) + ' ' + str(scores[i].std()) + '\r\n')
import numpy as np

for i in range(len(model_list)):
    expected = pd.DataFrame()
    predicted = np.array([])
    for train, test in lpo.split(X_raw, y):
        X_train = X_raw.iloc[train, best_features_list[i]]
        y_train = data.iloc[train, data.shape[1] - 1]
        X_test = X_raw.iloc[test, best_features_list[i]]
        y_test = data.iloc[test, data.shape[1] - 1]
        model_list[i].fit(X_train, y_train)
        # make predictions
        expected = pd.concat([expected, y_test])
        predicted = np.concatenate([predicted, model_list[i].predict(X_test)])
    print(model_name_list[i] + str(precision_recall_fscore_support(expected, predicted, average='macro')))


# def my_validation(model, X_f, y_f):
#     score = np.array([])
#     for train, test in lpo.split(X_f, y_f):
#         n_min = 0
#         for test_i in test:
#             if test_i in min_list:
#                 n_min += 1
#         if n_min != 1 and n_min != 2:
#             continue
#         # print(test)
#         model.fit(X_f.iloc[train, :], y_f.iloc[train])
#         predicted = model.predict(X_f.iloc[test, :])
#         score = np.append(score, np.array([metrics.accuracy_score(y_f.iloc[test], predicted)]))
#     # print(np.array([metrics.accuracy_score(y_f.iloc[test], predicted)]))
#     # print(score)
#     return score
#
# from sklearn.model_selection import ShuffleSplit
# from sklearn.model_selection import StratifiedShuffleSplit
# from sklearn.model_selection import StratifiedKFold
# skf = StratifiedKFold(n_splits=4)
# ss = StratifiedShuffleSplit(n_splits=150, test_size=0.25, random_state=None)
# # for train_index, test_index in ss.split(X_raw, y):
# #     print("%s %s" % (train_index, test_index))
#
# rand_list = [88,82,81,46,87,50,45,91,93,49,83,85,90,75,77,60,40,79,84,102,86,100,73,74,96,64,53,95,57,11,24,89,10,21,92,
#              94,28,80,76,1,78,62,104,26,39,58,19,98,56,27,66,109,4,110,18,47,48,107,52,25,37,103,36,33,32,97,108,51,13,
#              99,12,55,101,65,59,63,2,30,29,111,67,54,71,8,16,5,105,3,31,68,17,41,7,6,9,38,44,14,70,61,22,15,23,34,20,35,
#              69,72,43,106,42,112]
# for r in range(len(rand_list)):
#     rand_list[r] -= 1
#
# data_frame = pd.DataFrame([])
# for l in range(1, 111):
#     print('!!!!!!!!!!!k = ' + str(l) + '!!!!!!!!!!!!!!!!')
#     # X_new = SelectKBest(mutual_info_classif, k=l).fit_transform(X_raw, y)
#     # print(X_raw.shape)
#     X_new = X_raw.iloc[:, rand_list[:l]]
#     # this part is just computing scores with different models using given features list
#     scores = []
#     for i in range(len(model_list)):
#         scores.append(cross_val_score(model_list[i], X_new, y, cv=ss))
#
#     s_array = []
#     for i in range(len(scores)):
#         print("%s Accuracy: %0.4f (+/- %0.4f)" % (model_name_list[i], scores[i].mean(), scores[i].std()))
#     s_array = [[scores[0].mean(), scores[1].mean(), scores[2].mean(), scores[3].mean()]]
#         # print scores
#         # print
#         # result_file.write(str(scores[i].mean()) + ' ' + str(scores[i].std()) + '\r\n')
#     s_df = pd.DataFrame(s_array)
#     if data_frame.empty:
#         data_frame = s_df
#     else:
#         frames = [data_frame, s_df]
#         data_frame = pd.concat(frames)
#     import numpy as np
#
#     for i in range(len(model_list)):
#         expected = pd.DataFrame()
#         predicted = np.array([])
#         for train, test in ss.split(X_new, y):
#             X_train = X_new.iloc[train, :]
#             y_train = y.iloc[train]
#             X_test = X_new.iloc[test, :]
#             y_test = y.iloc[test]
#             model_list[i].fit(X_train, y_train)
#             # make predictions
#             expected = pd.concat([expected, y_test])
#             predicted = np.concatenate([predicted, model_list[i].predict(X_test)])
#         print(model_name_list[i] + str(precision_recall_fscore_support(expected, predicted, average='macro')))
#     print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
# data_frame.to_csv(c + '_correlation_ss.csv')