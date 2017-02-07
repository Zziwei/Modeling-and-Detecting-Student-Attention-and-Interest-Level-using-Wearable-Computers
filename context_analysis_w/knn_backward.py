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


class FeaturesCandidate:
    def __init__(self, features_list, score):
        self.features_list = features_list
        self.score = score

    def is_parent(self, candidate_features_list):
        is_all_in = True
        for tmp in self.features_list:
            if not (tmp in candidate_features_list):
                is_all_in = False
                break
        return is_all_in

    def display_feature_list(self):
        print("list : " + str(self.features_list))

    def display_score(self):
        print ("Name : ", self.name, ", Salary: ", self.salary)


def combine(l, n):
    answers = []
    one = [0] * n

    def next_c(li=0, ni=0):
        if ni == n:
            answers.append(copy.copy(one))
            return
        for lj in range(li, len(l)):
            one[ni] = l[lj]
            next_c(lj + 1, ni + 1)

    next_c()
    return answers


# best features list for all -- knn
# best_features_list = [8, 5, 17, 11, 22]
# best_features_list = [5, 9, 19, 23]
# knn Accuracy: 0.8168 (+/- 0.2279)


# fit a CART model to the data
data = pd.read_csv('input_difficulty_ww_n.csv', header=None)
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

model_name_list = ['knn']  # , 'random forrest']
model_list = [knnClf]  # , rand_forrest]

# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import mutual_info_classif
# for i in range(2, 76):
#     X_new = SelectKBest(mutual_info_classif, k=i).fit_transform(X_raw, y)
#     scores = []
#     scores.append(cross_val_score(decisionTree, X_new, y, cv=lpo).mean())
#     scores.append(cross_val_score(knnClf, X_new, y, cv=lpo).mean())
#     scores.append(cross_val_score(svc, X_new, y, cv=lpo).mean())
#     scores.append(cross_val_score(naive_bayes, X_new, y, cv=lpo).mean())
#     # scores.append(cross_val_score(rand_forrest, X, y, cv=lpo))
#     print('i = ' + str(i))
#     print(scores)
#     print


# import matplotlib.pyplot as plt
# from sklearn.svm import SVC
# from sklearn.model_selection import StratifiedKFold
# from sklearn.feature_selection import RFECV
#
# # The "accuracy" scoring is proportional to the number of correct
# # classifications
# rfecv = RFECV(estimator=svc, step=1, cv=lpo,
#               scoring='accuracy')
# rfecv.fit(X_raw, y)
#
# print("Optimal number of features : %d" % rfecv.n_features_)
#
# # Plot number of features VS. cross-validation scores
# plt.figure()
# plt.xlabel("Number of features selected")
# plt.ylabel("Cross validation score (nb of correct classifications)")
# plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
# plt.show()

# # ----------------------------------------------------------------------------------------------------------------------
# # this part is for selecting best features, every iteration select the feature combination with highest score
# # feature_selection_file = open('knn_difficulty_ww_n.txt', 'a')
# for l in range(len(model_list)):
#     feature_list = range(X_raw.shape[1])
#     highest_score = 0
#     while True:
#         tmp_highest_score = 0
#         tmp_best_feature_list = []
#         for i in range(len(feature_list)):
#             tmp_feature_list = copy.copy(feature_list)
#             tmp_feature_list.remove(tmp_feature_list[i])
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
#             # feature_selection_file.write(model_name_list[l] + ' : ' + str(highest_score) + ' : ' + str(feature_list))
#             # feature_selection_file.flush()
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
#                     if tmp_parent.score > s - 0.005:
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
#         if s_n > 50:
#             n = 50
#             s_standard = s_list[n - 1]
#         sure_candidate_list = []
#         print('mean = ' + str(s_mean))
#         for tmp_candidate in candidate_list:
#             if tmp_candidate.score > s_standard:
#                 sure_candidate_list.append(tmp_candidate)
#                 print(model_name_list[l] + ' score = ' + str(tmp_candidate.score) + ' ;; candidate list = ' + str(
#                     tmp_candidate.features_list))
#         last_candidate_list = copy.copy(sure_candidate_list)
#         candidate_list = []
# feature_selection_file.close()
# # ----------------------------------------------------------------------------------------------------------------------


# # this part is just computing scores with different models using given features list
# X = X_raw.iloc[:, best_features_list]
# scores = []
# scores.append(cross_val_score(decisionTree, X, y, cv=lpo))
# scores.append(cross_val_score(knnClf, X, y, cv=lpo))
# scores.append(cross_val_score(svc, X, y, cv=lpo))
# scores.append(cross_val_score(naive_bayes, X, y, cv=lpo))
# # scores.append(cross_val_score(rand_forrest, X, y, cv=lpo))
# # print(scores)
# model_name_list = ['decision tree', 'knn', 'svm', 'naive bayes', 'random forrest']
# # result_file = open('classify_result_3class_2experiments.txt', 'a')
# for i in range(len(scores)):
#     print("%s Accuracy: %0.4f (+/- %0.4f)" % (model_name_list[i], scores[i].mean(), scores[i].std()))
#     # print scores
#     print
#     # result_file.write(str(scores[i].mean()) + ' ' + str(scores[i].std()) + '\r\n')


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

# ----------------------------------------------------------------------------------------------------------------------
# this part is for selecting best features, every iteration select the feature combination with highest score
# feature_selection_file = open('knn_difficulty_ww_n.txt', 'a')
for l in range(len(model_list)):
    feature_list = []
    highest_score = 0
    while True:
        tmp_highest_score = 0
        tmp_best_feature_list = []
        for i in range(X_raw.shape[1]):
            if i in feature_list:
                continue
            tmp_feature_list = list(feature_list)
            tmp_feature_list.append(i)
            print tmp_feature_list
            X = X_raw.iloc[:, tmp_feature_list]
            scores = []
            scores.append(cross_val_score(model_list[l], X, y, cv=lpo).mean())
            # scores.append(cross_val_score(knnClf, X, y, cv=lpo).mean())
            # scores.append(cross_val_score(svc, X, y, cv=lpo).mean())
            # scores.append(cross_val_score(naive_bayes, X, y, cv=lpo).mean())
            # scores.append(cross_val_score(rand_forrest, X, y, cv=lpo).mean())
            tmp_highest_score = max(tmp_highest_score, max(scores))
            print scores
            if tmp_highest_score == max(scores):
                tmp_best_feature_list = list(tmp_feature_list)
                print tmp_highest_score

        if tmp_highest_score > highest_score:
            highest_score = tmp_highest_score
            feature_list = list(tmp_best_feature_list)
            print 'highest score = ' + str(highest_score)
            print feature_list
        else:
            print 'best features: ' + str(highest_score) + str(feature_list)
            # feature_selection_file.write(model_name_list[l] + ' : ' + str(highest_score) + ' : ' + str(feature_list))
            # feature_selection_file.flush()
            break
# ----------------------------------------------------------------------------------------------------------------------