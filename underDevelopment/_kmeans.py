#implementation of kmeans
import numpy as np


class KMeans:
    def __init__(self, number_of_clusters, centroids):

        self.number_of_clusters=number_of_clusters
        self.centroids=centroids

    def fit(self, data_matrix):
        self.data_matrix=data_matrix
        self.features=data_matrix.shape[1]
        self.datapoints=data_matrix.shape[0]
        self.cluster_assignment_list=np.zeros(self.datapoints).astype('int32')
        self.no_of_iteration=0

        new_centroids=self.centroids
        previous_centroids=np.zeros(new_centroids.shape)

        while (previous_centroids==new_centroids).all()==False :

            self.no_of_iteration=self.no_of_iteration+1
            datapoint_no=0
            while datapoint_no < self.datapoints :
                distance=self.find_distance(self.data_matrix[datapoint_no], new_centroids[0])
                assigned_cluster_no=0

                centroid_no=1
                while centroid_no < new_centroids.shape[0] :
                    tmp_distance=self.find_distance(self.data_matrix[datapoint_no], new_centroids[centroid_no])
                    
                    if tmp_distance < distance :
                        distance=tmp_distance
                        assigned_cluster_no=centroid_no

                    centroid_no=centroid_no+1

                self.cluster_assignment_list[datapoint_no]=assigned_cluster_no
                datapoint_no=datapoint_no+1

            previous_centroids=new_centroids
            new_centroids=self.calculate_centroids()

        self.centroids=new_centroids

        return self
    def find_distance(self, point_a, point_b): 
        return np.linalg.norm(point_a - point_b)
    
    def calculate_centroids(self):
        new_centroids=np.zeros(self.centroids.shape)
         
        datapoint_no=0
        while datapoint_no < self.datapoints :
            new_centroids[self.cluster_assignment_list[datapoint_no]]=new_centroids[self.cluster_assignment_list[datapoint_no]] + self.data_matrix[datapoint_no]
            datapoint_no=datapoint_no + 1

            centroid_no=0
            while centroid_no < self.number_of_clusters :
                temp=np.count_nonzero(self.cluster_assignment_list==centroid_no)
                if temp==0 :
                    temp=1
                new_centroids[centroid_no]=new_centroids[centroid_no]/temp
                centroid_no=centroid_no+1

            return new_centroids












