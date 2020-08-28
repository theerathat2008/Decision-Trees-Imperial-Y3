import numpy as np

class Dataset(object):

    def __init__(self):
        self.features = None
        self.labels = None

    # Constructs the dataset by reading data from a file
    def readData(self, path):
        # Open a file for reading
        with open(path) as f:

            features = []
            labels = []

            # Parse the data
            for line in f:
                data = line.rstrip().split(',')
                features.append(list(map(lambda x : int(x), data[:-1])))
                labels.append(data[-1])

            # Create the dataset
            self.features = np.array(features)
            self.labels = np.array(labels)

    # Prints the dataset
    def print(self):
        print("Features:")
        print(self.features)
        print("Labels:")
        print(self.labels)