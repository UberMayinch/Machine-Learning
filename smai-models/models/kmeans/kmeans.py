import numpy as np

class Kmeans:
    def __init__(self, n_clusters=3, max_iter=500, tol=0.0001):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.n_features = None
        self.cluster_centers = None
        self.labels = None

    def getCost(self, X, centers):
        distances = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        return distances

    def kmeans_plusplus(self, X):
        # Select the first cluster center randomly
        centers = [X[np.random.choice(X.shape[0])]]
        
        # Select the remaining cluster centers
        for _ in range(1, self.n_clusters):
            # Compute the distance of each point to the nearest cluster center
            distances = np.min(self.getCost(X, np.array(centers)), axis=1)
            # Square the distances
            squared_distances = distances ** 2
            # Select the next center with probability proportional to the squared distance
            probabilities = squared_distances / np.sum(squared_distances)
            cumulative_probabilities = np.cumsum(probabilities)
            r = np.random.rand()
            next_center = X[np.searchsorted(cumulative_probabilities, r)]
            centers.append(next_center)
        
        return np.array(centers)

    def fit(self, X):
        # Ensure X is 2D
        try:
            self.n_features = np.shape(X)[1]
        except:
            X = X.reshape(np.shape(X)[0], 1)
            self.n_features = 1

        # Initialize cluster centers using KMeans++
        centers = self.kmeans_plusplus(X)

        for _ in range(self.max_iter):
            # Calculate distances to each center and assign to closest center
            distances = self.getCost(X, centers)  # Should calculate pairwise distances
            cluster_membership = np.argmin(distances, axis=1)

            # Update cluster centers
            new_centers = np.zeros_like(centers)
            for i in range(self.n_clusters):
                members = X[cluster_membership == i]
                if len(members) > 0:
                    new_centers[i] = np.mean(members, axis=0)
                else:
                    # Handle empty clusters: reinitialize the center
                    new_centers[i] = X[np.random.choice(X.shape[0])]

            # Check for convergence
            if np.linalg.norm(centers - new_centers) < self.tol:
                break

            centers = new_centers

        self.cluster_centers = centers

    def predict(self, X):
        if self.cluster_centers is None:
            raise ValueError("Call the fit method first")
        else:
            distances = np.linalg.norm(X[:, np.newaxis, :] - self.cluster_centers[np.newaxis, :, :], axis=2)
            labels = np.argmin(distances, axis=1)
            self.labels = labels

        return labels
