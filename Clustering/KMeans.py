from tslearn.clustering import TimeSeriesKMeans
from tslearn.clustering import silhouette_score as sct
from sklearn.cluster import KMeans as kmeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score as sc
import pickle


class KMeans(object):

    def __init__(self, cluster_amount, time_series=False, metric='softdtw'):
        self.cluster_amount = cluster_amount
        self.time_series = time_series
        self.data = None
        if metric in ['dtw', 'softdtw', 'euclidean']:
            self.metric = metric
        else:
            raise ValueError('Metric must be dtw, softdtw or euclidean, not {}!'.format(metric))
        if time_series:
            self.cluster_algo = TimeSeriesKMeans(n_clusters=cluster_amount, n_init=10, metric=metric, verbose=False)
        else:
            self.cluster_algo = kmeans(n_clusters=cluster_amount)

    def fit(self, data):
        self.data = data
        self.cluster_algo.fit(data)

    def predict(self, data):
        return self.cluster_algo.predict(data)

    def labels_(self):
        return self.cluster_algo.labels_

    def plot(self):
        labels = self.labels_()
        colors = {1: (0, 0, 1), 2: (1, 0, 0), 0: (0, 1, 0), 3: (0.5, 0.5, 0), 4: (0, 0.5, 0.5), 5: (0.5, 0, 0.5)}
        for j in range(self.cluster_amount):
            for i, x in enumerate(self.data):
                if labels[i] == j:
                    plt.plot(x, c=colors[j])
        plt.show()
        for j in range(self.cluster_amount):
            for i, x in enumerate(self.data):
                if labels[i] == j:
                    plt.plot(x)
            # plt.savefig('{}.svg'.format(j))
            plt.show()

    def score(self):
        if self.time_series:
            return sct(self.data, self.labels_(), self.metric, verbose=False)
        else:
            return sc(self.data, self.labels_())

    def to_pickle(self, location):
        if self.time_series:
            self.cluster_algo.to_pickle(location)
        else:
            s = pickle.dumps(self.cluster_algo)
            with open(location, "wb") as f:
                f.write(s)
