#implementation of fuzzy kmeans
import numpy as np


class FuzzyKMeans():
    def __init__(self, number_of_clusters, centroids, fuzziness, threshold):

        self.number_of_clusters=number_of_clusters
        self.centroids=centroids
        self.fuzziness=fuzziness
        self.threshold=threshold

    def fit(self, data_matrix):
        self.data_matrix=data_matrix
        self.features=data_matrix.shape[1]
        self.datapoints=data_matrix.shape[0]

        new_centroids=self.centroids
        previous_centroids=np.zeros(new_centroids.shape)
        cluster_assignment_list=np.zeros(self.datapoints)
        membership_matrix=np.zeros([self.number_of_clusters,self.datapoints])
        distance_matrix=np.zeros([self.number_of_clusters,self.datapoints])

        while within_threshold(previous_centroids, new_centroids)==False :
            distance_matrix=calculate_distance_matrix(new_centroids)
            membership_matrix=calculate_membership_matrix(distance_matrix)
            previous_centroids=new_centroids
            new_centroids=calculate_centroids(membership_matrix)

        self.membership_matrix=membership_matrix
        self.centroids=new_centroids
        self.cluster_assignment_list=cluster_assignment()

        return self

    def calculate_distance_matrix(self, new_centroids):
        distance_matrix=np.zeros([self.number_of_clusters,self.datapoints])
        
        for i in range(self.number_of_clusters):

            for j in range(self.datapoints):
                distance_matrix[i][j]=find_distance(new_centroids[i], self.data_matrix[j])

        return distance_matrix

    def calculate_membership_matrix(self, distance_matrix):
        membership_matrix=np.zeros([self.number_of_clusters,self.datapoints])
        
        for i in range(self.datapoints):
            
            for j in range(self.number_of_clusters):
                temp=0.0

                for k in range(self.number_of_clusters):
                    temp=(distance_matrix[j][j]**2)/(distance_matrix[j][k]**2)+temp

                membership_matrix[j][i]=(temp**(1.0/(self.fuzziness-1)))**(-1)
        
        return membership_matrix

    def calculate_centroids(self, membership_matrix):
        new_centroids=np.zeros(self.centroids.shape)
        for i in range(self.number_of_clusters):

            for j in range(self.features):
                temp_numerator=0.0
                temp_denominator=0.0
                
                for k in range(self.datapoints):
                    temp_numerator=((membership_matrix[i][k]**self.fuzziness)*self.data_matrix[k][j])+temp_numerator
                    temp_denominator=(membership_matrix[i][k]**self.fuzziness)+temp_denominator
                
                new_centroids[i][j]=(temp_numerator/temp_denominator)
        
        return new_centroids

    def find_distance(self, point_a, point_b):
         
         return np.linalg.norm(point_a - point_b)
    
    def cluster_assignment(self):
        cluster_assignment_list=np.zeros(self.datapoints)

        for i in range(self.datapoints):
            cluster_no=0
            membership=self.membership_matrix[0][i]

            for j in range(1,self.number_of_clusters):

                if membership < self.membership_matrix[j][i] :
                    cluster_no=j
                    membership=self.membership_matrix[j][i]
            
            cluster_assignment_list[i]=cluster_no

        return cluster_assignment_list

    def within_threshold(self, previous_centroids, new_centroids):
        
        for i in range(self.number_of_clusters):
            distance=find_distance(previous_centroids[i], new_centroids[i])

            if distance > self.threshold :
                return False

        return True











