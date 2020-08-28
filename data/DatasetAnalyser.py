import numpy as np

from Dataset import Dataset


class DatasetAnalyser:

    def __init__(self):
        self.datasets = {}

    def add(self, dataset, name):
        self.datasets[name] = dataset

    def analyse(self):
        for key in self.datasets.keys():
            dataset = self.datasets[key]
            numOfFeatures = len(dataset.features)
            dictLabels = dict.fromkeys(dataset.labels, 0)
            labels = list(dictLabels)
            distributionA, distributionC, distributionE, distributionG, distributionO, distributionQ = self.__computeDistribution(dictLabels, dataset)

            print(key)
            print("Number of features: " + str(numOfFeatures))
            print("Unique labels")
            print("Number of labels: " + str(len(labels)))
            print("Labels: " + str(np.sort(labels)))
            print("Distribution of letters")
            print("A: " + str(distributionA) + "%")
            print("C: " + str(distributionC) + "%")
            print("E: " + str(distributionE) + "%")
            print("G: " + str(distributionG) + "%")
            print("O: " + str(distributionO) + "%")
            print("Q: " + str(distributionQ) + "%")
            print()

    def __computeDistribution(self, dictLabels, dataset):
        for label in dataset.labels:
            dictLabels[label] += 1

        return list(map(lambda x : x / len(dataset.features) * 100, dictLabels.values()))

    def checkDifference(self, dataset1, dataset2):

        sourceDataset = self.datasets[dataset1]
        targetDataset = self.datasets[dataset2]

        featureList = [",".join(list(map(lambda x: str(x), feature))) for feature in sourceDataset.features.tolist()]
        dictObservation = dict()

        for i in range(len(sourceDataset.labels)):
            dictObservation[featureList[i]] = sourceDataset.labels[i]

        dictLabels = { "A" : 0, "C" : 0, "E" : 0, "G" : 0, "O" : 0, "Q" : 0 }

        j = 0

        for feature in targetDataset.features:
            feature = ",".join(list(map(lambda x : str(x), feature)))
            label = targetDataset.labels[j]

            if (label != dictObservation[feature]):
                dictLabels[label] += 1

            j = j + 1

        print("Proportion of labels in " + dataset1 + " different than those in " + dataset2)
        print(str((sum(dictLabels.values()) / len(sourceDataset.features)) * 100) + "%")

fullData = Dataset()
subData = Dataset()
noisyData = Dataset()

fullData.readData("train_full.txt")
subData.readData("train_sub.txt")
noisyData.readData("train_noisy.txt")

datasetAnalyser = DatasetAnalyser()
datasetAnalyser.add(fullData, "fullData")
datasetAnalyser.add(subData, "subData")
datasetAnalyser.add(noisyData, "noisyData")

datasetAnalyser.analyse()
datasetAnalyser.checkDifference("fullData", "noisyData")