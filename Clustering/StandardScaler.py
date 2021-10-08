import numpy as np


class StandardScaler(object):
    def __init__(self):
        self.std = None
        self.mean = None

    def fit(self, data):
        data = np.array(data)
        self.std = np.std(np.hstack(data))
        self.mean = np.mean(np.hstack(data))

    def transform(self, data):
        new_data = []
        for d in data:
            # check if d is list or variable
            if hasattr(d, "__len__"):
                new_data.append([(x - self.mean) / self.std for x in d])
            else:
                new_data.append((d-self.mean)/self.std)
        return new_data
