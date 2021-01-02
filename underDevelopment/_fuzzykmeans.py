#implementation of fuzzy kmeans
import numpy as np


class FuzzyKMeans():
    def __init__(self, number_of_clusters , x):

        self.number_of_clusters=number_of_clusters
        self.x=x
