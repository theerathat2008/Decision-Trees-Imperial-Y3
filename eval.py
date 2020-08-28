##############################################################################
# CO395: Introduction to Machine Learning
# Coursework 1 Skeleton code
# Prepared by: Josiah Wang
#
# Your tasks: 
# Complete the following methods of Evaluator:
# - confusion_matrix()
# - accuracy()
# - precision()
# - recall()
# - f1_score()
##############################################################################
import copy
import itertools
import random
import statistics
from functools import reduce

import numpy as np

from classification import DecisionTreeClassifier


class Evaluator(object):
    """ Class to perform evaluation
    """
    
    def confusion_matrix(self, prediction, annotation, class_labels=None):
        """ Computes the confusion matrix.
        
        Parameters
        ----------
        prediction : np.array
            an N dimensional numpy array containing the predicted
            class labels
        annotation : np.array
            an N dimensional numpy array containing the ground truth
            class labels
        class_labels : np.array
            a C dimensional numpy array containing the ordered set of class
            labels. If not provided, defaults to all unique values in
            annotation.
        
        Returns
        -------
        np.array
            a C by C matrix, where C is the number of classes.
            Classes should be ordered by class_labels.
            Rows are ground truth per class, columns are predictions.
        """
        
        if not class_labels:
            class_labels = np.unique(annotation)
        
        confusion = np.zeros((len(class_labels), len(class_labels)), dtype=np.int)

        indexDictionary = dict()

        for i in range(len(class_labels)):
            label = class_labels[i]
            indexDictionary[label] = i

        confusionDictionary = { (i, j) : 0 for i in range(len(class_labels)) for j in range(len(class_labels))}

        for i in range(len(annotation)):
            annotatedLabel = annotation[i]
            predictedLabel = prediction[i]
            confusionIndex = (indexDictionary[annotatedLabel], indexDictionary[predictedLabel])
            confusionDictionary[confusionIndex] += 1

        for i in range(len(confusion)):
            for j in range(len(confusion)):
                confusion[i][j] = confusionDictionary[(i, j)]

        return confusion
    
    
    def accuracy(self, confusion):
        """ Computes the accuracy given a confusion matrix.
        
        Parameters
        ----------
        confusion : np.array
            The confusion matrix (C by C, where C is the number of classes).
            Rows are ground truth per class, columns are predictions
        
        Returns
        -------
        float
            The accuracy (between 0.0 to 1.0 inclusive)
        """
        
        # feel free to remove this
        accuracy = 0.0

        diagonalSum = confusion.trace()
        totalSum = confusion.sum()
        accuracy = diagonalSum / totalSum
        
        return accuracy
        
    
    def precision(self, confusion):
        """ Computes the precision score per class given a confusion matrix.
        
        Also returns the macro-averaged precision across classes.
        
        Parameters
        ----------
        confusion : np.array
            The confusion matrix (C by C, where C is the number of classes).
            Rows are ground truth per class, columns are predictions.
        
        Returns
        -------
        np.array
            A C-dimensional numpy array, with the precision score for each
            class in the same order as given in the confusion matrix.
        float
            The macro-averaged precision score across C classes.   
        """
        
        # Initialise array to store precision for C classes
        p = np.zeros((len(confusion), ))

        for i in range(len(confusion)):
            column = confusion[:, i]
            columnSum = column.sum()

            if (columnSum > 0):
                precision = confusion[i, i] / columnSum
            else:
                precision = 0

            p[i] = precision

        # You will also need to change this        
        macro_p = np.mean(p)

        return (p, macro_p)
    
    
    def recall(self, confusion):
        """ Computes the recall score per class given a confusion matrix.
        
        Also returns the macro-averaged recall across classes.
        
        Parameters
        ----------
        confusion : np.array
            The confusion matrix (C by C, where C is the number of classes).
            Rows are ground truth per class, columns are predictions.
        
        Returns
        -------
        np.array
            A C-dimensional numpy array, with the recall score for each
            class in the same order as given in the confusion matrix.
        
        float
            The macro-averaged recall score across C classes.   
        """
        
        # Initialise array to store recall for C classes
        r = np.zeros((len(confusion), ))

        for i in range(len(confusion)):
            row = confusion[i, :]
            rowSum = row.sum()

            if (rowSum > 0):
                recall = confusion[i, i] / rowSum
            else:
                recall = 0

            r[i] = recall
        
        # You will also need to change this        
        macro_r = np.mean(r)
        
        return (r, macro_r)
    
    
    def f1_score(self, confusion):
        """ Computes the f1 score per class given a confusion matrix.
        
        Also returns the macro-averaged f1-score across classes.
        
        Parameters
        ----------
        confusion : np.array
            The confusion matrix (C by C, where C is the number of classes).
            Rows are ground truth per class, columns are predictions.
        
        Returns
        -------
        np.array
            A C-dimensional numpy array, with the f1 score for each
            class in the same order as given in the confusion matrix.
        
        float
            The macro-averaged f1 score across C classes.   
        """
        
        # Initialise array to store recall for C classes
        f = np.zeros((len(confusion), ))

        precisionList = self.precision(confusion)[0]
        recallList = self.recall(confusion)[0]

        for i in range(len(f)):
            precision = precisionList[i]
            recall = recallList[i]

            if (precision == 0 and recall == 0):
                f[i] = 0
            else:
                f[i] = 2 * ((precision * recall) / (precision + recall))
        
        # You will also need to change this        
        macro_f = np.mean(f)
        
        return (f, macro_f)

    def __splitDataset(self, dataset, k):
        subsetSize = int(len(dataset.features) / k)
        featureLeft = len(dataset.features)
        subsets = []

        # Split dataset into k equal folds
        for i in range(k):
            features = np.empty([subsetSize, len(dataset.features[0])])
            labels = np.full(subsetSize, "")
            for j in range(subsetSize):
                # Draw a random number to select the next sample to be added
                selectedSample = int(random.uniform(0, featureLeft))

                # Add that sample to the subset
                features[j] = dataset.features[selectedSample]
                labels[j] = dataset.labels[selectedSample]

                # Remove the selected sample from the dataset
                dataset.features = np.delete(dataset.features, selectedSample, 0)
                dataset.labels = np.delete(dataset.labels, selectedSample, 0)

                featureLeft = featureLeft - 1

            subsets.append((features, labels))

        return subsets

    def kFoldCrossValidation(self, dataset, k):
        # Split the dataset into k folds
        subsets = self.__splitDataset(dataset, k)

        models = []

        # Perform cross validation
        for i in range(k):
            # Separate one fold for testing
            testSet = subsets[i]

            # Run an internal cross-validation over the remaining k - 1 folds to find the optimal parameters
            classifier = self.__crossValidation(subsets[:i] + subsets[(i + 1) : ])

            # Test the classifier on the test set
            testFeatures = testSet[0]
            testLabels = testSet[1]
            predictions = classifier.predict(testFeatures)
            evaluator = Evaluator()
            confusion = evaluator.confusion_matrix(predictions, testLabels)
            accuracy = evaluator.accuracy(confusion)
            macroP = evaluator.precision(confusion)[1]
            macroR = evaluator.recall(confusion)[1]
            macroF = evaluator.f1_score(confusion)[1]

            # Add the model to the candidate models
            models.append((classifier, [accuracy, macroP, macroR, macroF]))

        accuracies = ([model[1][0] for model in models])
        averageAccuracy = statistics.mean(accuracies)
        standardDeviation = statistics.stdev(accuracies)

        return (models, averageAccuracy, standardDeviation)

    def __crossValidation(self, subsets):
        models = []

        for i in range(len(subsets)):
            # Separate one fold for validation
            validationSet = subsets[i]

            # Use the remaining folds for training
            features, labels = list(zip(*(subsets[:i] + subsets[(i + 1) : ])))
            features = reduce(lambda x, y : np.append(x, y, axis=0), features)
            labels = reduce(lambda x, y : np.append(x, y, axis=0), labels)
            classifier = DecisionTreeClassifier()
            classifier.train(features, labels)

            # Evaluate the classifier on the validationSet
            validationFeatures = validationSet[0]
            validationLabels = validationSet[1]
            predictions = classifier.predict(validationFeatures)
            evaluator = Evaluator()
            confusion = evaluator.confusion_matrix(predictions, validationLabels)
            accuracy = evaluator.accuracy(confusion)
            macroP = evaluator.precision(confusion)[1]
            macroR = evaluator.recall(confusion)[1]
            macroF = evaluator.f1_score(confusion)[1]

            # Add the model to the candidate models
            models.append((classifier, [accuracy, macroP, macroR, macroF]))

        # Return the best model
        return self.getBestModel(models)

    def getBestModel(self, models):
        bestModel = (None, [0, 0, 0, 0])

        for model in models:
            accuracy = model[1][0]
            macroP = model[1][1]
            macroR = model[1][2]
            macroF = model[1][3]

            if (accuracy > bestModel[1][0] and macroP > bestModel[1][1] and macroR > bestModel[1][2] and macroF >
                    bestModel[1][3]):
                bestModel = model
        return bestModel[0]

    def prune(self, classifier, validationSet):
        return self.__pruneTree(classifier, classifier.root, classifier.root, validationSet)

    def __pruneTree(self, classifier, parentNode, childNode, validationSet):
        if (childNode.isLeaf()):
            return

        if (childNode.leftChild.isLeaf() and childNode.rightChild.isLeaf()):
            oldParentNode = copy.deepcopy(parentNode)
            evaluator = Evaluator()

            # Compute the accuracy of the old tree on the validation set
            oldPredictions = classifier.predict(validationSet.features)
            oldConfusion = evaluator.confusion_matrix(oldPredictions, validationSet.labels)
            oldAccuracy = evaluator.accuracy(oldConfusion)

            # Convert the childNode into a leaf node with class label set by majority vote
            self.__pruneNode(childNode)

            # Prune the tree
            if (childNode == parentNode.leftChild):
                parentNode.leftChild = childNode
            else:
                parentNode.rightChild = childNode

            # Compute the accuracy of the pruned tree on the validation set
            prunedPredictions = classifier.predict(validationSet.features)
            prunedConfusion = evaluator.confusion_matrix(prunedPredictions, validationSet.labels)
            prunedAccuracy = evaluator.accuracy(prunedConfusion)

            if (prunedAccuracy > oldAccuracy):
                # Return pruned tree
                return
            else:
                # Restore the old tree
                parentNode.leftChild = oldParentNode.leftChild
                parentNode.rightChild = oldParentNode.rightChild

        else:
            self.__pruneTree(classifier, childNode, childNode.leftChild, validationSet)
            self.__pruneTree(classifier, childNode, childNode.rightChild, validationSet)

            if (childNode.leftChild.isLeaf() and childNode.rightChild.isLeaf()):
                self.__pruneTree(classifier, parentNode, childNode, validationSet)

            return childNode

    def __pruneNode(self, node):
        dataset = node.dataset
        labels = dataset[1]

        label = max(labels, key=labels.count)
        node.leftChild = None
        node.rightChild = None
        node.label = label
        node.condition = None