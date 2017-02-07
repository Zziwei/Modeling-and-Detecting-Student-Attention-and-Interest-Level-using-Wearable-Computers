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

from imblearn.over_sampling import SMOTE


# fit a CART model to the data
data = pd.read_csv('input_d_2_hrv.csv', header=None)
decisionTree = DecisionTreeClassifier()
knnClf = KNeighborsClassifier(n_neighbors=4)  # default:k = 5,defined by yourself:KNeighborsClassifier(n_neighbors=10)
svc = svm.SVC(kernel='linear', C=1)  # (kernel='linear', C=1)   #(kernel='rbf') #(kernel='poly', degree=5)
naive_bayes = GaussianNB()
rand_forrest = RandomForestClassifier(n_estimators=25)
lpo = LeavePOut(p=3)
kf = KFold(n_splits=5)
X_raw = data.iloc[:, :data.shape[1] - 1]
y = data.iloc[:, data.shape[1] - 1]
# X = X_raw.iloc[:, best_features_list]

model_name_list = ['DTree', 'knn', 'svm', 'nb']  # , 'random forrest']
model_list = [decisionTree, knnClf, svc, naive_bayes]  # , rad_forrest]


best_features_list = [[28, 37, 47, 51, 80], [21, 22, 28, 47, 51, 80, 82, 87, 95, 108],
               [33, 41, 45, 49, 55, 81, 85, 87], [1, 21, 22, 34, 41, 53, 61, 64, 90, 92, 111]]
# best_features_list = [[25, 49, 80, 98, 105], [3, 25, 31, 45, 80, 94, 95, 98], [3, 38, 43, 45, 49, 67, 80, 81, 86, 87, 98, 99, 107, 109],
#                [45, 49, 53, 64, 65, 81, 87, 89, 90]]
# best_features_list = [11, 44, 45, 46, 53, 51]
for i in range(len(model_list)):
    print(model_name_list[i])
    model = model_list[i]
    X_train = X_raw.iloc[7:, best_features_list[i]]
    y_train = y.iloc[7:]
    X_test = X_raw.iloc[:7, best_features_list[i]]
    y_test = y.iloc[:7]
    model.fit(X_train, y_train)
    predict = model.predict(X_test)
    print(metrics.classification_report(y_test, predict))
    print(metrics.confusion_matrix(y_test, predict))
    print()