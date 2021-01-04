#implementation of certain useful functions
import numpy as np
import pandas as pd

def random_normalised_centroids(shape):

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
    for i in centroids[1:]:
        tmp=np.random.rand(cluster_size,centroids.shape[1])*10
        tmp=tmp+i
        data_matrix=np.concatenate((data_matrix,tmp))
    np.random.shuffle(data_matrix)

    return data_matrix

