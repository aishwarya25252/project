import numpy as np
from sklearn.cluster import KMeans
from skfuzzy.cluster import cmeans
from sklearn.metrics import davies_bouldin_score

class Wrap:
    def kmeans(data_matrix,number_of_clusters,fuzzy_parameter=None,error=None,maximun_iterations=None):
        kmeans = KMeans(n_clusters=number_of_clusters,algorithm='full').fit(data_matrix)
        return kmeans.cluster_centers_,kmeans.labels_,davies_bouldin_score(data_matrix,kmeans.labels_)

    def fmeans(data_matrix,number_of_clusters,fuzzy_parameter,error,maximun_iterations):    #alter
        fmeans=cmeans(data_matrix.T,number_of_clusters,fuzzy_parameter,error,maximun_iterations)
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

    def fkmeans(data_matrix,number_of_clusters,fuzzy_parameter,error,maximun_iterations):
        fmeans=cmeans(data_matrix.T,number_of_clusters,fuzzy_parameter,error,maximun_iterations)
        centroids=fmeans[0]
        kmeans = KMeans(n_clusters=number_of_clusters,algorithm='full').fit(centroids)

        return kmeans.cluster_centers_,kmeans.predict(data_matrix),davies_bouldin_score(data_matrix,kmeans.predict(data_matrix))

    def consencous_kmeans(dataframe,number_of_clusters,iterations,sample_count,hold):   #alter
        consencous_matrix=np.zeros([dataframe.index.size,dataframe.index.size])
        adjusted_matrix=np.zeros([dataframe.index.size,dataframe.index.size])
        for i in range(iterations):
            data=dataframe.sample(sample_count)
            index_list=data.index
            data_matrix=data.values
            kmeans = KMeans(n_clusters=number_of_clusters,algorithm='full').fit(data_matrix)
            for j in range(sample_count):
                for k in range(sample_count):
                    if (kmeans.labels_[j]==kmeans.labels_[k]):
                        consencous_matrix[index_list[j]][index_list[k]]=consencous_matrix[index_list[j]][index_list[k]]+1.0

        for i in range(dataframe.index.size):
            for j in range(dataframe.index.size):
                if (consencous_matrix[i][j] > hold):
                    adjusted_matrix[i][j]=1
                else :
                    adjusted_matrix[i][j]=0

        return consencous_matrix, adjusted_matrix

    def alternate_consencous_kmeans(dataframe,number_of_clusters,iterations,sample_count):
        bestscore=100
        bestcentroids=None
        bestlabels=None
        for i in range(iterations):
            data=dataframe.sample(sample_count)
            kmeans = KMeans(n_clusters=number_of_clusters,algorithm='full').fit(data)
            labels = kmeans.predict(dataframe)
            centroids = kmeans.cluster_centers_
            score=davies_bouldin_score(dataframe,labels)
            print(score)
            if (score < bestscore) :
                bestcentroids=centroids
                bestlabels=labels
                bestscore=score

        return bestcentroids,bestlabels,bestscore







