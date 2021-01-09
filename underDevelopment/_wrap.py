import numpy as np
from sklearn.cluster import KMeans
import skfuzzy as sf

class Wrap:
    def kmeans(data_matrix,number_of_clusters):
        kmeans = KMeans(n_clusters=number_of_clusters,algorithm='full').fit(data_matrix)
        return kmeans.cluster_centers_,kmeans.labels_

    def fmeans(data_matrix,number_of_clusters,fuzzy_parameter,error,maximun_iterations):
        fmeans=sf.cluster.cmeans(data_matrix.T,number_of_clusters,fuzzy_parameter,error,maximun_iterations)
        centroids=fmeans[0]
        datapoint_no=0
        cluster_assignment_list=np.zeros(data_matrix.shape[0]).astype('int32')
        while (datapoint_no < data_matrix.shape[0]):
            distance=np.linalg.norm(data_matrix[datapoint_no]-centroids[0])
            assigned_cluster_no=0
            centroid_no=1
            while centroid_no < number_of_clusters :
                tmp_distance=np.linalg.norm(data_matrix[datapoint_no]-centroids[centroid_no])
                if tmp_distance < distance :
                    distance=tmp_distance
                    assigned_cluster_no=centroid_no
                centroid_no=centroid_no+1
            cluster_assignment_list[datapoint_no]=assigned_cluster_no
            datapoint_no=datapoint_no+1

        return centroids,cluster_assignment_list


