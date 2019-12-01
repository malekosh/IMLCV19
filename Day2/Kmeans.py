import numpy as np
from IPython.display import HTML
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from sklearn.cluster import KMeans
from sklearn import datasets

# A manual implementation of k-means from the example at: 
# https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_iris.html

def run_kmeans(X, k=2, n=5, np_seed=50):
    
    # Set numpy seed
    np.random.seed(np_seed)

    # Initialize means with features similar to data:

    # Loop n times to find best cluster centroids:
    
    # Plot data before and after clustering in two separate plots:
    
    return centroids