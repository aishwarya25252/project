#implementation of certain useful functions
import numpy as np
import pandas as pd

def random_normalised_centroids(shape):

    return np.random.rand(shape[0],shape[1])

def normalise_dataframe(dataframe):
    
    return (dataframe.T/dataframe.max(0).values.reshape(dataframe.shape[1],1)).T

def import_dataframe(name,delimiter):

    return pd.read_csv(name,sep=delimiter,index_col=0).T
