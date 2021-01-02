#implementation of kmeans
import numpy as np


class KMeans():
    def __init__(self, number_of_clusters, centroids):

        self.number_of_clusters=number_of_clusters
        self.centroids=centroids

    def fit(self, x)

