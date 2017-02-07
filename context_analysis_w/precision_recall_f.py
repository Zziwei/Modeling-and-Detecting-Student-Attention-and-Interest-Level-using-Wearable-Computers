from sklearn import metrics
import pandas as pd
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
from sklearn.metrics import precision_recall_fscore_support

from imblearn.over_sampling import SMOTE


# # best features list for all -- knn
# best_features_list = [[28, 37, 47, 51, 80], [21, 22, 28, 47, 51, 80, 82, 87, 95, 108],
#                       [34, 41, 45, 49, 55, 81, 85, 87, 99, 101], [7, 21, 22, 41, 53, 61, 90, 111]]
best_features_list = [[25, 33, 49, 80, 98, 105], [3, 25, 31, 45, 80, 94, 95, 98], [3, 38, 43, 45, 49, 67, 80, 81, 86, 87, 98, 99, 107, 109],
               [45, 49, 53, 64, 65, 81, 87, 89, 90]]


# fit a CART model to the data
data = pd.read_csv('input_i_2_hrv_c.csv', header=None)
decisionTree = DecisionTreeClassifier()
knnClf = KNeighborsClassifier(n_neighbors=3)  # default:k = 5,defined by yourself:KNeighborsClassifier(n_neighbors=10)
svc = svm.SVC(kernel='linear', C=1)  # (kernel='linear', C=1)   #(kernel='rbf') #(kernel='poly', degree=5)
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

model_name_list = ['decision tree', 'knn', 'svm', 'naive bayes']  # , 'random forrest']
model_list = [decisionTree, knnClf, svc, naive_bayes]  # , rand_forrest]


# # ----------------------------------------------------------------------------------------------------------------------
# # this part is for selecting best features, every iteration select the feature combination with highest score
# feature_selection_file = open('knn_difficulty_ww_n.txt', 'a')
# for l in range(len(model_list)):
#     feature_list = []
#     highest_score = 0
#     while True:
#         tmp_highest_score = 0
#         tmp_best_feature_list = []
#         for i in range(X_raw.shape[1]):
#             if i in feature_list:
#                 continue
#             tmp_feature_list = list(feature_list)
#             tmp_feature_list.append(i)
#             print tmp_feature_list
#             X = X_raw.iloc[:, tmp_feature_list]
#             scores = []
#             scores.append(cross_val_score(model_list[l], X, y, cv=lpo).mean())
#             # scores.append(cross_val_score(knnClf, X, y, cv=lpo).mean())
#             # scores.append(cross_val_score(svc, X, y, cv=lpo).mean())
#             # scores.append(cross_val_score(naive_bayes, X, y, cv=lpo).mean())
#             # scores.append(cross_val_score(rand_forrest, X, y, cv=lpo).mean())
#             tmp_highest_score = max(tmp_highest_score, max(scores))
#             print scores
#             if tmp_highest_score == max(scores):
#                 tmp_best_feature_list = list(tmp_feature_list)
#                 print tmp_highest_score
#
#         if tmp_highest_score > highest_score:
#             highest_score = tmp_highest_score
#             feature_list = list(tmp_best_feature_list)
#             print 'highest score = ' + str(highest_score)
#             print feature_list
#         else:
#             feature_selection_file.write(model_name_list[l] + ' : ' + str(highest_score) + ' : ' + str(feature_list))
#             feature_selection_file.flush()
#             break
# # ----------------------------------------------------------------------------------------------------------------------


# # ---------------------------------------------------------------------------------------------------------------------
# # this part is to select best features, keep all features combination with increasing scores.
# feature_selection_file = open('knn_difficulty_ww_n.txt', 'a')
#
#
# def get_feature_candidate(f_list, model_index):
#     score = get_score(f_list, model_index)
#     candidate = FeaturesCandidate(f_list, score)
#     return candidate
#
#
# def get_score(f_list, model_index):
#     current_X = X_raw.iloc[:, f_list]
#     score = (cross_val_score(model_list[model_index], current_X, y, cv=lpo).mean())
#     return score
#
#
# for l in range(len(model_list)):
#     candidate_list = []
#     last_candidate_list = []
#     first_candidate_list = []
#     highest_score_in_iteration = 0
#     best_features_list_in_iteration = []
#     s_sum = 0
#     s_n = 0
#     combine_list = combine(range(X_raw.shape[1]), 1)
#     for tmp_feature_list in combine_list:
#         feature_candidate = get_feature_candidate(tmp_feature_list, l)
#         first_candidate_list.append(feature_candidate)
#         s_sum += feature_candidate.score
#         s_n += 1
#         print(str(tmp_feature_list) + ' :: ' + str(first_candidate_list[len(first_candidate_list) - 1].score))
#     s_mean = s_sum / s_n
#     print('mean = ' + str(s_mean))
#
#     for first_candidate_tmp in first_candidate_list:
#         if first_candidate_tmp.score >= s_mean:
#             last_candidate_list.append(first_candidate_tmp)
#             print('above mean :: ' + str(first_candidate_tmp.features_list) + ' :: ' + str(first_candidate_tmp.score))
#
#     for j in range(2, X_raw.shape[1] + 1):
#         combine_list = combine(range(X_raw.shape[1]), j)
#         s_sum = 0
#         s_n = 0
#         s_list = []
#         has_higher_score = False
#         # print str(combine_list)
#         for current_combine_feature in combine_list:
#             parents_list = []
#             for tmp_last_candidate in last_candidate_list:
#                 if tmp_last_candidate.is_parent(current_combine_feature):
#                     parents_list.append(tmp_last_candidate)
#
#             if len(parents_list) != 0:
#                 s = get_score(current_combine_feature, l)
#                 is_current_combine_feature_good = True
#                 for tmp_parent in parents_list:
#                     if tmp_parent.score > s - 0.008 or s < s_mean:
#                         is_current_combine_feature_good = False
#                         break
#                 if is_current_combine_feature_good:
#                     current_candidate = FeaturesCandidate(current_combine_feature, s)
#                     candidate_list.append(current_candidate)
#                     s_sum += s
#                     s_list.append(s)
#                     print(len(s_list))
#                     s_n += 1
#                     if s >= highest_score_in_iteration:
#                         highest_score_in_iteration = s
#                         best_features_list_in_iteration = current_combine_feature
#                         has_higher_score = True
#         if not has_higher_score:
#             feature_selection_file.write(
#                 model_name_list[l] + ' : ' + str(highest_score_in_iteration) + ' : ' + str(
#                     best_features_list_in_iteration) + '\r\n')
#             feature_selection_file.flush()
#             break
#
#         # if len(candidate_list) == 0:
#         #     highest_score = 0
#         #     best_features_list = []
#         #     for tmp_candidate in last_candidate_list:
#         #         if tmp_candidate.score > highest_score:
#         #             highest_score = tmp_candidate.score
#         #             best_features_list = copy.copy(tmp_candidate.features_list)
#         #     if highest_score_in_iteration >= highest_score:
#         #         highest_score = highest_score_in_iteration
#         #         best_features_list = copy.copy(best_features_list_in_iteration)
#         #     feature_selection_file.write(
#         #         model_name_list[l] + ' : ' + str(highest_score) + ' : ' + str(best_features_list) + '\r\n')
#         #     feature_selection_file.flush()
#         #     break
#         s_mean = s_sum / s_n
#         s_list.sort()
#         s_standard = s_mean
#         if s_n > 30:
#             n = 30
#             s_standard = s_list[len(s_list) - n]
#         sure_candidate_list = []
#         print('mean = ' + str(s_mean))
#         s_sum = 0
#         s_n = 0
#         for tmp_candidate in candidate_list:
#             if tmp_candidate.score > s_standard:
#                 sure_candidate_list.append(tmp_candidate)
#                 s_sum += tmp_candidate.score
#                 s_n += 1
#                 print(model_name_list[l] + ' score = ' + str(tmp_candidate.score) + ' ;; candidate list = ' + str(
#                     tmp_candidate.features_list))
#         s_mean = s_sum / s_n
#         print('new mean = ' + str(s_mean))
#         last_candidate_list = copy.copy(sure_candidate_list)
#         candidate_list = []
# feature_selection_file.close()
# # ----------------------------------------------------------------------------------------------------------------------


# this part is just computing scores with different models using given features list
scores = []
for i in range(len(model_list)):
    X = X_raw.iloc[:, best_features_list[i]]
    scores.append(cross_val_score(model_list[i], X, y, cv=lpo))

# scores.append(cross_val_score(rand_forrest, X, y, cv=lpo))
# print(scores)
# model_name_list = ['decision tree', 'knn', 'svm', 'naive bayes', 'random forrest']
# result_file = open('classify_result_3class_2experiments.txt', 'a')
for i in range(len(scores)):
    print("%s Accuracy: %0.4f (+/- %0.4f)" % (model_name_list[i], scores[i].mean(), scores[i].std()))
    # print scores
    print
    # result_file.write(str(scores[i].mean()) + ' ' + str(scores[i].std()) + '\r\n')


# for train, test in lpo.split(data):
#     model.fit(X, y)
#     # make predictions
#     expected = test.iloc[:, :21]
#     predicted = model.predict(test.iloc[:, :22])
#     # summarize the fit of the model
#     print(metrics.classification_report(expected, predicted))
#     print(metrics.confusion_matrix(expected, predicted))
#     print(model.feature_importances_)

# a = [7]
# b = range(14)
# c = []
# for i in b:
#     if i not in a:
#         c.append(i)
# model_list[0].fit(X_raw.iloc[c, best_features_list], y.iloc[c])
# print model_list[0].predict(X_raw.iloc[0, best_features_list])

import numpy as np

for i in range(len(model_list)):
    expected = pd.DataFrame()
    predicted = np.array([])
    for train, test in lpo.split(data):
        X_train = data.iloc[train, best_features_list[i]]
        y_train = data.iloc[train, data.shape[1] - 1]
        X_test = data.iloc[test, best_features_list[i]]
        y_test = data.iloc[test, data.shape[1] - 1]
        model_list[i].fit(X_train, y_train)
        # make predictions
        expected = pd.concat([expected, y_test])
        predicted = np.concatenate([predicted, model_list[i].predict(X_test)])
    print(model_name_list[i] + str(precision_recall_fscore_support(expected, predicted, average='macro')))