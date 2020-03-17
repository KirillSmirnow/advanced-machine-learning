from sklearn.neighbors import KNeighborsClassifier
import csv

train_x = []
train_y = []
with open("normalized-stars-ru.csv") as file:
    reader = csv.DictReader(file)
    for row in reader:
        predictors = []
        for predictor in row.keys():
            if predictor != "TARGET":
                predictors.append(float(row[predictor]))
        train_x.append(predictors)
        train_y.append(int(row["TARGET"]))

test_x = [0.772, 0.204, 0.137, 0.55, 0.316, 0.221, 0.19, 0.927]

distances = []
for e in train_x:
    ss = 0
    for i in range(len(test_x)):
        ss += (e[i] - test_x[i]) ** 2
    distances.append(ss ** 0.5)
distances.sort()
print(distances[0])

knn = KNeighborsClassifier()
knn.fit(train_x, train_y)
print(knn.predict([test_x]))
