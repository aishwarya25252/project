#implementation of certain useful functions
import numpy as np
import pandas as pd

class PreProcessing :
    def generate_normalised_centroids(shape):

        return np.random.rand(shape[0],shape[1])

    def normalise_dataframe(dataframe):

        return (dataframe.T/dataframe.max(0).values.reshape(dataframe.shape[1],1)).T

    def import_dataframe(name,delimiter):

        return pd.read_csv(name,sep=delimiter,index_col=0).T

    def sample_sataframe(dataframe,fraction):

        return d.sample(frac=fraction)

    def feature_selection(dataframe,feature_count):
        tmp=dataframe.std()
        labels=dataframe.sort_values(ascending=False).index[:feature_count]

        return dataframe[labels]

    def data_generate(centroids,cluster_size):
        centroids=centroids*100
        data_matrix=(np.random.rand(cluster_size,centroids.shape[1])*10)+centroids[0]
        new_centroids=np.average(data_matrix,0)
        new_centroids=new_centroids.reshape(1,new_centroids.shape[0])

        for i in centroids[1:]:
            tmp=np.random.rand(cluster_size,centroids.shape[1])*10
            tmp=tmp+i
            new_centroids=np.concatenate((new_centroids,np.average(tmp,0).reshape(1,tmp.shape[1])))
            data_matrix=np.concatenate((data_matrix,tmp))
        np.random.shuffle(data_matrix)

        return data_matrix, new_centroids

    def find_average(matrix):

        return np.average(matrix,0)
