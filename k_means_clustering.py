import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class SimpleKMeans:
    def __init__(self, k=3, max_iters=100, random_state=42):
        """
        Initialize K-Means clustering algorithm
        
        Parameters:
        k: number of clusters
        max_iters: maximum number of iterations
        random_state: random seed for reproducibility
        """
        self.k = k
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        
    def initialize_centroids(self, X):
        """Randomly initialize centroids from the data points"""
        np.random.seed(self.random_state)
        random_idx = np.random.permutation(len(X))
        centroids = X[random_idx[:self.k]]
        return centroids
    
    def compute_distance(self, X, centroids):
        """Compute Euclidean distance between each point and each centroid"""
        distances = np.zeros((len(X), self.k))
        for i, centroid in enumerate(centroids):
            # Euclidean distance: sqrt(sum((x - centroid)^2))
            distances[:, i] = np.sqrt(np.sum((X - centroid) ** 2, axis=1))
        return distances
    
    def find_closest_centroid(self, distances):
        """Assign each point to the closest centroid"""
        return np.argmin(distances, axis=1)
    
    def compute_centroids(self, X, labels):
        """Compute new centroids as the mean of points in each cluster"""
        centroids = np.zeros((self.k, X.shape[1]))
        for i in range(self.k):
            # Get all points assigned to cluster i
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                centroids[i] = np.mean(cluster_points, axis=0)
        return centroids
    
    def fit(self, X):
        """Fit K-Means to the data X"""
        # Step 1: Initialize centroids randomly
        self.centroids = self.initialize_centroids(X)
        
        for iteration in range(self.max_iters):
            # Step 2: Calculate distances and assign points to clusters
            distances = self.compute_distance(X, self.centroids)
            self.labels = self.find_closest_centroid(distances)
            
            # Step 3: Calculate new centroids
            new_centroids = self.compute_centroids(X, self.labels)
            
            # Step 4: Check for convergence (if centroids don't change)
            if np.allclose(self.centroids, new_centroids):
                print(f"Converged after {iteration + 1} iterations")
                break
                
            self.centroids = new_centroids
            
        return self
    
    def predict(self, X):
        """Predict cluster labels for new data"""
        distances = self.compute_distance(X, self.centroids)
        return self.find_closest_centroid(distances)

# Generate sample data
np.random.seed(42)
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Apply K-Means clustering
kmeans = SimpleKMeans(k=4, max_iters=100)
kmeans.fit(X)
labels = kmeans.labels
centroids = kmeans.centroids

# Visualize the results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot original data with true labels
ax1.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', s=50, alpha=0.7)
ax1.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], 
            c='red', marker='X', s=200, label='Centroids')
ax1.set_title("Original Data with True Labels")
ax1.set_xlabel("Feature 1")
ax1.set_ylabel("Feature 2")
ax1.legend()

# Plot clustered data
ax2.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
ax2.scatter(centroids[:, 0], centroids[:, 1], 
            c='red', marker='X', s=200, label='Centroids')
ax2.set_title("K-Means Clustering Results (k=4)")
ax2.set_xlabel("Feature 1")
ax2.set_ylabel("Feature 2")
ax2.legend()

plt.tight_layout()
# Add this line after plt.show() in your original code
plt.savefig('kmeans_clustering.png')


# Print cluster information
print("\nCluster Centroids:")
for i, centroid in enumerate(centroids):
    print(f"Cluster {i}: {centroid}")

print(f"\nNumber of points in each cluster:")
for i in range(kmeans.k):
    print(f"Cluster {i}: {np.sum(labels == i)} points")