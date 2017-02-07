import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans

data = pd.read_csv('watch_features_4classes.csv')
print('data({0[0]},{0[1]})'.format(data.shape))
kmeans = KMeans(init='k-means++', n_clusters=3)
kmeans.fit(data)
print(kmeans.labels_)
# print(kmeans.cluster_centers_)
# print(kmeans.precompute_distances)
# 4 2 2 1 2 0 2 3

# 2 1 1 0 0 1 1 0