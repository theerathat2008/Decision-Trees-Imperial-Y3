##############################################################################
# CO395: Introduction to Machine Learning
# Coursework 1 Skeleton code
# Prepared by: Josiah Wang
#
# Your tasks: Complete the train() and predict() methods of the 
# DecisionTreeClassifier 
##############################################################################
import collections
import math
import pickle
import sys

import numpy as np

class DecisionTreeClassifier(object):
    """
    A decision tree classifier

    Attributes
    ----------
    is_trained : bool
        Keeps track of whether the classifier has been trained

    Methods
    -------
    train(X, y)
        Constructs a decision tree from data X and label y
    predict(X)
        Predicts the class label of samples X

    """

    def __init__(self):
        self.is_trained = False
        self.root = self.IntNode()
        self.ATTRIBUTE_LIST = ["x-box", "y-box", "width", "high", "onpix", "x-bar", "y-bar", "x2bar", "y2bar", "xybar",
                               "x2ybr", "xy2br", "x-ege", "xegvy", "y-ege", "yegvx"]
        self.maxAttributeRange = 0
        self.treeArray = np.full((4, 32), " ", dtype=object)

    def train(self, x, y):
        """ Constructs a decision tree classifier from data

        Parameters
        ----------
        x : numpy.array
            An N by K numpy array (N is the number of instances, K is the
            number of attributes)
        y : numpy.array
            An N-dimensional numpy array

        Returns
        -------
        DecisionTreeClassifier
            A copy of the DecisionTreeClassifier instance

        """

        # Make sure that x and y have the same number of instances
        assert x.shape[0] == len(y), \
            "Training failed. x and y must have the same number of instances."

        self.maxAttributeRange = len(x[0])
        self.root = self.__induceDecisionTree((x, y))

        # set a flag so that we know that the classifier has been trained
        self.is_trained = True

        return self

    def __induceDecisionTree(self, dataset):
        features = dataset[0]
        labels = dataset[1]

        if (all(label == labels[0] for label in labels) or not self.__canSplit(dataset)):
            # Return a leaf node with the majority label
            label = max(labels, key=labels.count)
            return self.IntNode(None, None, label, None)

        else:
            # Search for an optimal rule to split the data upon
            condition = self.__findBestNode(dataset)
            nodeLabel = self.ATTRIBUTE_LIST[condition[0]] + " <= " + str(condition[1])

            # Split the dataset according to the splitting rule
            leftSet, rightSet = self.__splitDataset(condition, dataset)

            # Induce a decision tree on the children
            leftChild = self.__induceDecisionTree(leftSet)
            rightChild = self.__induceDecisionTree(rightSet)


            # Return the parent
            return self.IntNode(leftChild, rightChild, nodeLabel, condition, dataset)

    # Checks whether the dataset can be split any further
    def __canSplit(self, dataset):
        if (len(dataset[0]) == 0 or len(dataset[1]) == 0):
            return False

        EPSILON = 2
        features = dataset[0]

        # Check whether all the data points are close to each other
        for attribute in range(0, self.maxAttributeRange):
            min = sys.maxsize
            max = 1 - min

            for sample in features:
                value = sample[attribute]

                if (value < min):
                    min = value

                if (value > max):
                    max = value

                if ((max - min) > EPSILON):
                    return True

        return False

    def __findBestNode(self, dataset):
        potentialNodes = self.__computePotentialNodes(dataset)
        bestGain = 0
        bestNode = None

        # Selects the best node by using Information Gain
        for node in potentialNodes:
            leftSet, rightSet = self.__splitDataset(node, dataset)
            entropyParent = self.__computeEntropy(dataset)
            entropyChildren = 0
            entropyChildren += (len(leftSet[0]) / len(dataset[0])) * self.__computeEntropy(leftSet)
            entropyChildren += (len(rightSet[0]) / len(dataset[0])) * self.__computeEntropy(rightSet)
            gain = entropyParent - entropyChildren

            if (gain > bestGain):
                bestGain = gain
                bestNode = node

        return bestNode

    # Finds good split points by sorting the values of the attributes and considering only split points
    # that are between two examples in sorted order
    def __computePotentialNodes(self, dataset):
        features = dataset[0]
        labels = dataset[1]

        # Sorts the attributes in ascending order
        sortedAttributes = np.argsort(features, axis=0).T
        potentialNodes = []

        # Searches for split points that are between two examples in sorted order
        for i in range(len(sortedAttributes)):
            attribute = sortedAttributes[i]
            sample = attribute[0]
            previousLabel = labels[sample]

            for j in range(len(attribute)):
                sample = attribute[j]
                currentLabel = labels[sample]

                if (currentLabel != previousLabel):
                    previousSample = attribute[j - 1]
                    value = features[previousSample][i]
                    potentialNodes.append((i, value))

                previousLabel = currentLabel

        return list(dict.fromkeys(potentialNodes))


    def __computeEntropy(self, dataset):
        labelsDistribution = self.__computeLabelsDistribution(dataset)
        labels = list(labelsDistribution)
        entropy = -sum([labelsDistribution[label] * math.log(labelsDistribution[label], 2) for label in labels])

        return entropy

    def __computeLabelsDistribution(self, dataset):
        labels = dataset[1]
        dictLabels = collections.Counter(labels)

        for label in dictLabels.keys():
            value = dictLabels[label]
            dictLabels[label] = value / len(labels)

        return dictLabels

    def __splitDataset(self, condition, dataset):
        attribute = condition[0]
        value = condition[1]
        features = dataset[0]
        labels = dataset[1]
        leftSetFeatures = []
        leftSetLabels = []
        rightSetFeatures = []
        rightSetLabels = []

        for i in range(len(features)):
            sample = features[i]
            label = labels[i]
            if (sample[attribute] <= value):
                leftSetFeatures.append(sample)
                leftSetLabels.append(label)
            else:
                rightSetFeatures.append(sample)
                rightSetLabels.append(label)

        return ((leftSetFeatures, leftSetLabels), (rightSetFeatures, rightSetLabels))

    def predict(self, x):
        """ Predicts a set of samples using the trained DecisionTreeClassifier.

        Assumes that the DecisionTreeClassifier has already been trained.

        Parameters
        ----------
        x : numpy.array
            An N by K numpy array (N is the number of samples, K is the
            number of attributes)

        Returns
        -------
        numpy.array
            An N-dimensional numpy array containing the predicted class label
            for each instance in x
        """

        # make sure that classifier has been trained before predicting
        if not self.is_trained:
            raise Exception("Decision Tree classifier has not yet been trained.")

        # set up empty N-dimensional vector to store predicted labels
        # feel free to change this if needed
        predictions = np.zeros((x.shape[0],), dtype=np.object)

        for i in range(len(predictions)):
            predictions[i] = self.__predictLabel(x[i], self.root)

        # remember to change this if you rename the variable
        return predictions

    def __predictLabel(self, sample, node):
        if (node.isLeaf()):
            return node.label
        else:
            attribute = node.condition[0]
            value = node.condition[1]

            if (sample[attribute] <= value):
                return self.__predictLabel(sample, node.leftChild)
            else:
                return self.__predictLabel(sample, node.rightChild)

    def print(self):
        self.__traverseTree(self.root)

    def getMaximumDepth(self):
        return self.__computeMaximumDepth(self.root)

    # Computes the maximum depth of the tree
    def __computeMaximumDepth(self, node):
        if (node.isLeaf()):
            return 1
        return max(self.__computeMaximumDepth(node.leftChild), self.__computeMaximumDepth(node.rightChild)) + 1

    def __traverseTree(self, node, level=0):
        if (node.isLeaf()):
            print('\t' * level + "+---Leaf " + node.label)
        else:
            print('\t' * level + "+---IntNode " + node.label)
            self.__traverseTree(node.leftChild, level + 1)
            self.__traverseTree(node.rightChild, level + 1)

    # Iterate through first 4 levels of tree to create an array used to visualise the tree
    def __createArrayPrintTree(self, node, skip, level=0):
        if (node.isLeaf()) :
            self.treeArray[int(level)][int(skip)] = str("Leaf " + node.label)

        else:
            self.treeArray[int(level)][int(skip)] = str("IntNode " + node.label)
            if (level < 3):
                self.__createArrayPrintTree(node.leftChild, skip - (math.pow(2, 3 - level)), level + 1)
                self.__createArrayPrintTree(node.rightChild, skip + (math.pow(2, 3 - level)), level + 1)

    def printImageTree(self):
        array = self.__createArrayPrintTree(self.root, 16, 0)
        np.set_printoptions(linewidth=1000)
        array = self.treeArray
        for i in range(3):
            for j in range(i + 1):
                array[i][j * 13] = " " * (3 - i) * 10

            print(array[i])
            print("\n")
        array[3][0] = "      "
        array[3][15] = "      "
        print(array[3])

    # Saves the model to a file
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.root, f, pickle.HIGHEST_PROTOCOL)

    # Loads the model from a file
    def load(self, path):
        with open(path, "rb") as f:
            self.root = pickle.load(f)
            self.is_trained = True

    # A node in the decision tree classifier
    class IntNode:

        def __init__(self, leftChild=None, rightChild=None, label=None, condition=None, dataset = None):
            self.leftChild = leftChild
            self.rightChild = rightChild
            self.label = label
            self.condition = condition
            self.dataset = dataset

        def isLeaf(self):
            return self.leftChild == None and self.rightChild == None