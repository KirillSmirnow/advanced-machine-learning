import csv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score

FEATURES = "chocolate, fruity, caramel, peanutyalmondy, nougat, crispedricewafer, " \
           "hard, bar, pluribus, sugarpercent, pricepercent".split(", ")
EXCLUDE_ELEMENTS = "Milky Way Simply Caramel, Payday, Runts".split(", ")

train_x = []
train_y = []
with open("candy-data.csv") as test_file:
    reader = csv.DictReader(test_file)
    for row in reader:
        if row["\ufeffcompetitorname"] not in EXCLUDE_ELEMENTS:
            train_x.append([float(row[feature]) for feature in FEATURES])
            train_y.append(int(row["Y"]))

test_x = []
test_y = []
test_names = []
with open("candy-test.csv") as test_file:
    reader = csv.DictReader(test_file)
    for row in reader:
        test_x.append([float(row[feature]) for feature in FEATURES])
        test_y.append(int(row["Y"]))
        test_names.append(row["\ufeffcompetitorname"])

lr = LogisticRegression(random_state=2019)
lr.fit(train_x, train_y)

predictions = lr.predict_proba(test_x)
predicted_labels = tuple(map(lambda prediction: 0 if prediction[0] > 0.5 else 1, predictions))
for i in range(len(test_names)):
    if test_names[i] in ("Trolli Sour Bites", "Twizzlers"):
        print("%s -> %s, prob1=%.2f" % (test_names[i], predicted_labels[i], predictions[i][1]))

print("Precision = %.3f" % precision_score(test_y, predicted_labels))
print("Recall = %.3f" % recall_score(test_y, predicted_labels))
print("AUC = %.3f" % roc_auc_score(test_y, predicted_labels))
