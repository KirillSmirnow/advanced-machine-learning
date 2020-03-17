import numpy as np
from sklearn.cluster import KMeans

X = [
    [12, 11],
    [5, 12],
    [14, 15],
    [3, 3],
    [9, 1],
    [11, 11],
    [15, 2],
    [6, 4],
    [17, 11],
    [13, 11],
    [18, 11],
    [9, 15],
    [15, 20],
    [9, 18],
    [14, 5]
]

kmeans = KMeans(n_clusters=3, init=np.array([[10.33, 8.5], [10.0, 7.0], [12.57, 12.14]]), max_iter=100, n_init=1)
kmeans.fit(X)
predictions = kmeans.predict(X)
print("Predictions: " + str(predictions))

cluster0 = []
for i in range(len(predictions)):
    if predictions[i] == 0:
        cluster0.append(X[i])

distances_sum = 0
for a in range(len(cluster0)):
    for b in range(len(cluster0)):
        if a < b:
            distances_sum += (cluster0[a][0] - cluster0[b][0]) ** 2 + (cluster0[a][1] - cluster0[b][1]) ** 2
print("Average intra-cluster distance for the cluster 0 is " + str(distances_sum / len(cluster0)))
