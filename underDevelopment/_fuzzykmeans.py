#implementation of fuzzy kmeans
import numpy as np


class FuzzyKMeans():
    def __init__(self, number_of_clusters, centroids):

        self.number_of_clusters=number_of_clusters
        self.centroids=centroids

    def fit(self, data_matrix):
        self.data_matrix=data_matrix
        self.features=data_matrix.shape[1]
        self.datapoints=data_matrix.shape[0]

        new_centroids=self.centroids
        previous_centroids=np.zeros(new_centroids.shape)
        cluster_assignment_list=np.zeros(self.datapoints)
        membership_matrix=np.zeros([self.number_of_clusters,self.datapoints])
        distance_matrix=np.zeros([self.number_of_clusters,self.datapoints])

        while (previous_centroids==new_centroids).all()==False :
            distance_matrix=calculate_distance_matrix(new_centroids)
            membership_matrix=calculate_membership_matrix(distance_matrix)
            previous_centroids=new_centroids
            new_centroids=calculate_centroids(membership_matrix)

        self.centroids=new_centroids
        self.cluster_assignment_list=cluster_assignment_list

        return self

    def calculate_distance_matrix(self, new_centroids):
        pass
        return distance_matrix

    def calculate_membership_matrix(self, distance_matrix):
        pass
        return membership_matrix

    def calculate_centroids(self, membership_matrix):
        pass
        return new_centroids

    def find_distance(self, point_a, point_b):
         
         return np.linalg.norm(point_a - point_b)
