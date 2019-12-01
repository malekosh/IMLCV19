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
    centroids = np.zeros((k, X.shape[-1]))
    for dim in range(X.shape[1]):
        centroids[:,dim] = np.random.choice(X[:,dim], k)

    # Loop n times to find best cluster centroids
    for i in range(n):
        X_dists = np.expand_dims(X, axis=1)-np.expand_dims(centroids, axis=0)
        X_dists_euc = np.sqrt(np.sum(X_dists**2, axis=-1))
        samples_assignments = np.argmin(X_dists_euc, axis=-1)
        for ki in range(k):
            centroids[ki,:] = np.mean(X[samples_assignments==ki,:], axis=0)
    
    # Plot data before and after clustering in two separate plots:
    plt.figure(figsize=[11,5])
    plt.subplot(121)
    plt.title('Data before clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.scatter(X[:, 0], X[:, 1])
    plt.subplot(122)
    plt.title('Data after clustering')
    plt.xlabel('Feature 1')
    plt.scatter(X[:, 0], X[:, 1], c=samples_assignments.astype(np.float))
    plt.scatter(centroids[:, 0], centroids[:, 1],
                c=np.arange(centroids.shape[0]).astype(np.float), edgecolor='k', marker='X', s=100)
    plt.show()
    
    return centroids