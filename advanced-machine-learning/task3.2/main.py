import csv
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

TRAIN_TEST_COUNT = 650
TRAIN_PARTITION_FACTOR = 0.8

# Get data
all_predictors = []
all_responses = []
with open("diabetes.csv") as file:
    objects = csv.DictReader(file)
    for object in objects:
        all_predictors.append([float(object[predictor]) for predictor in object if predictor != "Outcome"])
        all_responses.append(int(object["Outcome"]))

predictors = all_predictors[:TRAIN_TEST_COUNT]
responses = all_responses[:TRAIN_TEST_COUNT]

train_predictors = predictors[:int(len(predictors) * TRAIN_PARTITION_FACTOR)]
test_predictors = predictors[int(len(predictors) * TRAIN_PARTITION_FACTOR):]
train_responses = responses[:int(len(responses) * TRAIN_PARTITION_FACTOR)]
test_responses = responses[int(len(responses) * TRAIN_PARTITION_FACTOR):]

print(f"Number of objects labeled '0' = {responses.count(0)}")

# Train
classifier = DecisionTreeClassifier(criterion="entropy", max_leaf_nodes=30, min_samples_leaf=20, random_state=2020)
classifier.fit(train_predictors, train_responses)
print(f"Tree depth = {classifier.tree_.max_depth}")

# Score
test_predictions = classifier.predict(test_predictors)
print(f"Accuracy = {accuracy_score(test_responses, test_predictions)}")
print(f"F1 score = {f1_score(test_responses, test_predictions, average='macro')}")

# Predict
for id in (743, 715, 740, 741):
    prediction = classifier.predict([all_predictors[id]])
    print(f"Prediction for id={id}: {prediction}")
