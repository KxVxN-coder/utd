import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA

# -------------------------------
# Load the digits dataset
# -------------------------------
digits = load_digits()
X = digits.data
y = digits.target
k = 10  # Number of clusters

# -------------------------------
# K-Means from Scratch
# -------------------------------
def k_means(X, k, max_iters=100):
    # Initialize centroids randomly
    np.random.seed(42)
    centroids = X[np.random.choice(len(X), k, replace=False)]
    
    for _ in range(max_iters):
        # Assign clusters
        distances = pairwise_distances(X, centroids)
        labels = np.argmin(distances, axis=1)

        # Update centroids
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    # Compute total loss (sum of squared distances)
    loss = np.sum([np.linalg.norm(X[i] - centroids[labels[i]])**2 for i in range(len(X))])
    return labels, centroids, loss

# -------------------------------
# K-Medoids from Scratch
# -------------------------------
def k_medoids(X, k, max_iters=100):
    np.random.seed(42)
    medoid_indices = np.random.choice(len(X), k, replace=False)
    medoids = X[medoid_indices]

    for _ in range(max_iters):
        distances = pairwise_distances(X, medoids)
        labels = np.argmin(distances, axis=1)

        new_medoids = []
        for i in range(k):
            cluster_points = X[labels == i]
            if len(cluster_points) == 0:
                continue
            medoid_index = np.argmin(pairwise_distances(cluster_points, cluster_points).sum(axis=1))
            new_medoids.append(cluster_points[medoid_index])
        
        new_medoids = np.array(new_medoids)
        if np.allclose(medoids, new_medoids):
            break
        medoids = new_medoids

    # Compute total loss (sum of distances)
    loss = np.sum([np.linalg.norm(X[i] - medoids[labels[i]]) for i in range(len(X))])
    return labels, medoids, loss

# -------------------------------
# Run Both Algorithms
# -------------------------------
labels_kmeans, centers_kmeans, loss_kmeans = k_means(X, k)
labels_kmedoids, medoids_kmedoids, loss_kmedoids = k_medoids(X, k)

print(f"K-Means Loss: {loss_kmeans:.2f}")
print(f"K-Medoids Loss: {loss_kmedoids:.2f}")

# -------------------------------
# Visualize Cluster Centers
# -------------------------------
fig, axs = plt.subplots(2, k, figsize=(12, 3))
for i in range(k):
    axs[0, i].imshow(centers_kmeans[i].reshape(8, 8), cmap='gray')
    axs[0, i].axis('off')
    axs[0, i].set_title(f"KM {i}")

    axs[1, i].imshow(medoids_kmedoids[i].reshape(8, 8), cmap='gray')
    axs[1, i].axis('off')
    axs[1, i].set_title(f"KMed {i}")
    
plt.suptitle("Top: K-Means Centers | Bottom: K-Medoids Medoids")
plt.tight_layout()
plt.show()

# -------------------------------
# Optional: PCA Visualization
# -------------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_kmeans, cmap='tab10', s=10)
plt.title("K-Means Clusters (PCA)")

plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_kmedoids, cmap='tab10', s=10)
plt.title("K-Medoids Clusters (PCA)")
plt.show()
