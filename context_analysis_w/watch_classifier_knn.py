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
from sklearn.model_selection import KFold
import copy
import numpy as np

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedShuffleSplit
ss = StratifiedShuffleSplit(n_splits=150, test_size=0.2, random_state=None)


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


# fit a CART model to the data
m = 1
c = 'd'
min_list = []
if c == 'i':
    min_list = [0, 8, 9, 12]
else:
    min_list = [3, 4, 7, 10, 11, 13]
data = pd.read_csv('input_' + c + '_2_hrv_c.csv', header=None)
decisionTree = DecisionTreeClassifier()
knnClf = KNeighborsClassifier(n_neighbors=3)  # default:k = 5,defined by yourself:KNeighborsClassifier(n_neighbors=10)
svc = svm.SVC(kernel='linear', C=1)  # (kernel='linear', C=1)   #(kernel='rbf') #(kernel='poly', degree=5)
naive_bayes = GaussianNB()
rand_forrest = RandomForestClassifier(n_estimators=25)
lpo = LeavePOut(p=3)
kf = KFold(n_splits=5)
X_raw = data.iloc[:, :data.shape[1] - 1]
y = data.iloc[:, data.shape[1] - 1]


def my_validation(model, X_f, y_f):
    score = np.array([])
    if c == 'i':
        X_train = X_f.iloc[8:, :]
        y_train = y_f.iloc[8:]
        X_test = X_f.iloc[:8, :]
        y_test = y_f.iloc[:8]
    else:
        X_train = X_f.iloc[:8, :]
        y_train = y_f.iloc[:8]
        X_test = X_f.iloc[8:, :]
        y_test = y_f.iloc[8:]
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    lecture_s = metrics.accuracy_score(y_test, predicted)
    for train, test in lpo.split(X_f, y_f):
        n_min = 0
        for test_i in test:
            if test_i in min_list:
                n_min += 1
        if n_min != 1 and n_min != 2:
            continue
        # print(test)
        model.fit(X_f.iloc[train, :], y_f.iloc[train])
        predicted = model.predict(X_f.iloc[test, :])
        score = np.append(score, np.array([metrics.accuracy_score(y_f.iloc[test], predicted)]))
    # print(np.array([metrics.accuracy_score(y_f.iloc[test], predicted)]))
    # print(len(score))
    return score

model_name_list = ['decision_tree', 'knn', 'svm', 'nb']  # , 'random forrest']
model_list = [decisionTree, knnClf, svc, naive_bayes]  # , rand_forrest]


# ---------------------------------------------------------------------------------------------------------------------
# this part is to select best features, keep all features combination with increasing scores.
feature_selection_file = open(model_name_list[m] + '_' + c + '_2_hrv_c.txt', 'a')


def is_parent_list(parent_list, child):
    has_parent = False
    for parent in parent_list:
        if parent.is_parent(child):
            has_parent = True
            break
    return has_parent


def combine_1(l):
    n = 1
    answers = []
    one = [0] * n

    def next_c(li=0, ni=0):
        if ni == n:
            # if is_parent_list(cand_list, one):
            answers.append(copy.copy(one))
            return
        for lj in range(li, len(l)):
            one[ni] = l[lj]
            next_c(lj + 1, ni + 1)

    next_c()
    return answers


def combine(l, n, cand_list):
    answers = []
    for cand in cand_list:
        for ll in l:
            # print(cand.features_list)
            if ll in cand.features_list:
                continue
            else:
                one = copy.copy(cand.features_list)
                one.append(ll)
                one.sort()
                # print(one)
                if not one in answers:
                    answers.append(copy.copy(one))
                # print(answers)
    # print(answers)
    return answers


def get_feature_candidate(f_list, model_index):
    score = get_score(f_list, model_index)
    candidate = FeaturesCandidate(f_list, score)
    return candidate


def get_score(f_list, model_index):
    current_X = X_raw.iloc[:, f_list]
    score = (cross_val_score(model_list[model_index], current_X, y, cv=lpo).mean())
    # score = (my_validation(model_list[model_index], current_X, y).mean())
    return score


l = m
candidate_list = []
last_candidate_list = []
first_candidate_list = []
highest_score_in_iteration = 0
best_features_list_in_iteration = []
s_sum = 0
s_n = 0

# last_list = [[9, 44, 58, 59, 62]]
# last_score = [0.970695970696]
# s_mean = 0.970695970696

last_list = [[21, 22, 28, 47, 51, 80, 82, 87, 95, 108]]
last_score = [0.882783882784]
s_mean = 0.882783882784

# combine_list = combine_1(range(X_raw.shape[1]))
# for tmp_feature_list in combine_list:
#     feature_candidate = get_feature_candidate(tmp_feature_list, l)
#     first_candidate_list.append(feature_candidate)
#     s_sum += feature_candidate.score
#     s_n += 1
#     print(str(tmp_feature_list) + ' :: ' + str(first_candidate_list[len(first_candidate_list) - 1].score))
# s_mean = s_sum / s_n
print('last accuracy = ' + str(s_mean))

s_sum = 0
s_n = 0

for i in range(len(last_list)):
    feature_candidate = FeaturesCandidate(last_list[i], last_score[i])
    last_candidate_list.append(feature_candidate)

# for first_candidate_tmp in first_candidate_list:
#     if first_candidate_tmp.score >= s_mean:
#         s_sum += first_candidate_tmp.score
#         s_n += 1
#         last_candidate_list.append(first_candidate_tmp)
#         print('above mean :: ' + str(first_candidate_tmp.features_list) + ' :: ' + str(first_candidate_tmp.score))

# last_score = [0.975274725275, 0.972527472527, 0.973443223443, 0.971611721612, 0.974358974359,
#               0.973443223443, 0.974358974359, 0.971611721612, 0.972527472527, 0.971611721612]
# last_list = [[5, 15, 44, 45, 46], [9, 13, 15, 44, 48], [10, 11, 13, 44, 48], [11, 13, 35, 37, 44],
#              [11, 44, 45, 46, 53], [11, 44, 45, 48, 53], [11, 44, 46, 53, 72], [13, 15, 45, 46, 53],
#              [13, 31, 35, 37, 53], [44, 48, 53, 69, 73]]
# s_mean = 0.9730769230769
# for k in range(len(last_score)):
#     last_candidate_list.append(FeaturesCandidate(last_list[k], last_score[k]))

for j in range(2, X_raw.shape[1] + 1):
    combine_list = combine([32, 33, 34, 35], j, last_candidate_list)
    s_sum = 0
    s_n = 0
    s_list = []
    has_higher_score = False
    print str(combine_list)
    for current_combine_feature in combine_list:
        parents_list = []
        for tmp_last_candidate in last_candidate_list:
            if tmp_last_candidate.is_parent(current_combine_feature):
                parents_list.append(tmp_last_candidate)

        if len(parents_list) != 0:
            s = get_score(current_combine_feature, l)
            print(str(s) + ' : ' + str(current_combine_feature))
            is_current_combine_feature_good = True
            for tmp_parent in parents_list:
                if tmp_parent.score - 0.0000001 > s or s < s_mean - 0.000000001:
                    is_current_combine_feature_good = False
                    break
            if is_current_combine_feature_good:
                current_candidate = FeaturesCandidate(current_combine_feature, s)
                candidate_list.append(current_candidate)
                s_sum += s
                s_list.append(s)
                print(str(len(s_list)) + ' : ' + str(s) + ' : ' + str(current_combine_feature))
                s_n += 1
                if s >= highest_score_in_iteration - 0.000000001:
                    highest_score_in_iteration = s
                    best_features_list_in_iteration = current_combine_feature
                    has_higher_score = True
    if not has_higher_score:
        feature_selection_file.write(
            model_name_list[l] + ' : ' + str(highest_score_in_iteration) + ' : ' + str(
                best_features_list_in_iteration) + '\r\n')
        feature_selection_file.flush()
        break
    s_mean = s_sum / s_n
    s_list.sort()
    s_standard = s_mean
    if s_n > 30:
        n = 30
        s_standard = s_list[len(s_list) - n]
    sure_candidate_list = []
    print('mean = ' + str(s_mean))
    s_sum = 0
    s_n = 0
    for tmp_candidate in candidate_list:
        if tmp_candidate.score >= s_standard - 0.000000001:
            sure_candidate_list.append(tmp_candidate)
            s_sum += tmp_candidate.score
            s_n += 1
            print(model_name_list[l] + ' score = ' + str(tmp_candidate.score) + ' ;; candidate list = ' + str(
                tmp_candidate.features_list))
    s_mean = s_sum / s_n
    print('new mean = ' + str(s_mean))
    last_candidate_list = copy.copy(sure_candidate_list)
    candidate_list = []
feature_selection_file.close()
