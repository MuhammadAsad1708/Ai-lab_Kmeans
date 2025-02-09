import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data(filename):
    df = pd.read_csv(filename)
    return df.values  # Convert dataframe to NumPy array

def normalize_data(data):
    return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))

def initialize_centroids(data, k):
    return data[np.random.choice(data.shape[0], k, replace=False)]

def compute_distances(data, centroids):
    distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
    return distances

def assign_clusters(distances):
    return np.argmin(distances, axis=1)

def update_centroids(data, labels, k):
    new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
    return new_centroids

def k_means_clustering(data, k, max_iters=100, tol=1e-4):
    centroids = initialize_centroids(data, k)
    for _ in range(max_iters):
        distances = compute_distances(data, centroids)
        labels = assign_clusters(distances)
        new_centroids = update_centroids(data, labels, k)
        if np.linalg.norm(new_centroids - centroids) < tol:
            break
        centroids = new_centroids
    return labels, centroids

def plot_clusters(data, labels, k):
    for i in range(k):
        plt.scatter(data[labels == i, 0], data[labels == i, 1], label=f'Cluster {i}')
    plt.legend()
    plt.show()

# Load and preprocess dataset
data = load_data("/Users/muhammadasad/Desktop/ai lab/6/kmeans - kmeans_blobs.csv")

data = normalize_data(data)

# Run K-means for k=2 and k=3
for k in [2, 3]:
    labels, _ = k_means_clustering(data, k)
    print(f"Plot for k={k}")
    plot_clusters(data, labels, k)
