from scipy import stats

import numpy as np

from classification import DecisionTreeClassifier
from data.Dataset import Dataset
from eval import Evaluator

if __name__ == "__main__":
    print("Loading the training dataset...")
    dataset = Dataset()
    dataset.readData("data/train_full.txt")
    x = dataset.features
    y = dataset.labels

    print("Training the decision tree...")
    classifier = DecisionTreeClassifier()
    classifier = classifier.train(x, y)

    classifier.print()

    print("\n")

    print("Tree visualisation graphically")
    print("\n")
    print("\n")
    print("\n")
    classifier.printImageTree()
    print("\n")
    print("\n")
    print("\n")
    print("\n")
    print("\n")
    print("\n")
    print("\n")

    print("Loading the test set...")
    testSet = Dataset()
    testSet.readData("data/test.txt")
    validationSet = Dataset()
    validationSet.readData("data/validation.txt")
    xTest = testSet.features
    yTest = testSet.labels
    xValidation = validationSet.features
    yValidation = validationSet.labels

    predictions = classifier.predict(xTest)
    print("Predictions: {}".format(predictions))

    classes = ["A", "C", "E", "G", "O", "Q"]

    print("Evaluating test predictions...")
    evaluator = Evaluator()
    confusion = evaluator.confusion_matrix(predictions, yTest)

    print("Confusion matrix:")
    print(confusion)

    accuracy = evaluator.accuracy(confusion)
    print()
    print("Accuracy: {}".format(accuracy))

    (p, macro_p) = evaluator.precision(confusion)
    (r, macro_r) = evaluator.recall(confusion)
    (f, macro_f) = evaluator.f1_score(confusion)

    print()
    print("Class: Precision, Recall, F1")
    for (i, (p1, r1, f1)) in enumerate(zip(p, r, f)):
        print("{}: {:.2f}, {:.2f}, {:.2f}".format(classes[i], p1, r1, f1));

    print()
    print("Macro-averaged Precision: {:.2f}".format(macro_p))
    print("Macro-averaged Recall: {:.2f}".format(macro_r))
    print("Macro-averaged F1: {:.2f}".format(macro_f))