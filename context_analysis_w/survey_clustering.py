import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans

data = pd.read_csv('clustering_survey_d.csv', header=None)
print('data({0[0]},{0[1]})'.format(data.shape))
kmeans = KMeans(init='k-means++', n_clusters=2)
kmeans.fit(data)
print(kmeans.labels_)
# print(kmeans.cluster_centers_)
# print(kmeans.precompute_distances)