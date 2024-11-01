import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cupy as cp  

class KNN_model:
    def __init__(self, data, k=5, distance_metric="euclidean", use_gpu=False):
        """
        Initialize the KNN_model class.

        Parameters:
        - data: numpy.ndarray or cupy.ndarray
            The input data array containing both features and labels.
        - k: int, optional (default=5)
            The number of nearest neighbors to consider.
        - distance_metric: str, optional (default="euclidean")
            The distance metric to use for calculating distances between data points.
            Supported options: "euclidean", "cosine", "manhattan".
        - use_gpu: bool, optional (default=False)
            Whether to use GPU for computations.

        Returns:
        None
        """
        self.use_gpu = use_gpu
        
        if self.use_gpu:
            self.data = cp.array(data[:, :-1].astype(cp.float32))  # Features on GPU
            self.labels = data[:, -1]  # Labels on GPU
        else:
            self.data = data[:, :-1]  # Features on CPU
            self.labels = data[:, -1]  # Labels on CPU

        self.neighbors = k
        self.distance_metric = self.euclidean_distance if distance_metric == "euclidean" else self.cosine_distance if distance_metric == "cosine" else self.manhattan_distance
        
    def euclidean_distance(self, X1, X2):
        """
        Calculate the Euclidean distance between two sets of data points.

        Parameters:
        - X1: numpy.ndarray or cupy.ndarray
            The first set of data points.
        - X2: numpy.ndarray or cupy.ndarray
            The second set of data points.

        Returns:
        - numpy.ndarray or cupy.ndarray
            The pairwise Euclidean distances between the data points.
        """
        # Efficient euclidean distance calculation using broadcasting and matrix multiplication
        if self.use_gpu:
            X1_square = cp.sum(X1**2, axis=1).reshape(-1, 1)
            X2_square = cp.sum(X2**2, axis=1).reshape(1, -1)
            return cp.sqrt(X1_square + X2_square - 2 * cp.dot(X1, X2.T))
        else:
            X1 = X1.astype(np.float32)  
            X2 = X2.astype(np.float32)
            X1_square = np.sum(X1**2, axis=1).reshape(-1, 1)
            X2_square = np.sum(X2**2, axis=1).reshape(1, -1)
            return np.sqrt(X1_square + X2_square - 2 * np.dot(X1, X2.T))
        
    def cosine_distance(self, X1, X2):
        """
        Calculate the Cosine distance between two sets of data points.

        Parameters:
        - X1: numpy.ndarray or cupy.ndarray
            The first set of data points.
        - X2: numpy.ndarray or cupy.ndarray
            The second set of data points.

        Returns:
        - numpy.ndarray or cupy.ndarray
            The pairwise Cosine distances between the data points.
        """
        # Efficient cosine distance calculation using broadcasting and matrix multiplication
        if self.use_gpu:
            X1_norm = cp.linalg.norm(X1, axis=1).reshape(-1, 1)
            X2_norm = cp.linalg.norm(X2, axis=1).reshape(1, -1)
            dot_product = cp.dot(X1, X2.T)
            return 1 - dot_product / (X1_norm * X2_norm)
        else:
            X1 = X1.astype(np.float32)  
            X2 = X2.astype(np.float32)  
            X1_norm = np.linalg.norm(X1, axis=1).reshape(-1, 1)
            X2_norm = np.linalg.norm(X2, axis=1).reshape(1, -1)
            dot_product = np.dot(X1, X2.T)
            return 1 - dot_product / (X1_norm * X2_norm)
        
    def manhattan_distance(self, X1, X2):
        """
        Calculate the Manhattan distance between two sets of data points.

        Parameters:
        - X1: numpy.ndarray or cupy.ndarray
            The first set of data points.
        - X2: numpy.ndarray or cupy.ndarray
            The second set of data points.

        Returns:
        - numpy.ndarray or cupy.ndarray
            The pairwise Manhattan distances between the data points.
        """
        # Efficient manhattan distance calculation using broadcasting and matrix subtraction
        if self.use_gpu:
            return cp.sum(cp.abs(X1[:, None] - X2), axis=-1)
        else:
            return np.sum(np.abs(X1[:, None] - X2), axis=-1)
        
    def inference(self, test):
        """
        Perform inference on the test data.

        Parameters:
        - test: numpy.ndarray or cupy.ndarray
            The test data array containing both features and labels.

        Returns:
        - tuple
            A tuple containing the predicted labels and the true labels.
        """
        test_labels = test[:, -1]
        test_features = test[:, :-1]
        
        if self.use_gpu:
            test_features = cp.array(test_features.astype(cp.float32))
        
        distances = self.distance_metric(self.data, test_features)
        distances = distances.T
        
        nearest_indices = np.argsort(distances, axis=1)[:, :self.neighbors] if self.use_gpu else np.argsort(distances, axis=1)[:, :self.neighbors]
        
        nearest_labels = self.labels[nearest_indices.get()] if self.use_gpu else self.labels[nearest_indices]
        
        predicted_labels = []
        
        for row in nearest_labels:
            label_counts = {}
            for label in row:
                if label in label_counts:
                    label_counts[label] += 1
                else:
                    label_counts[label] = 1
            max_count = max(label_counts.values())
            max_labels = [label for label, count in label_counts.items() if count == max_count]
            for label in row:
                if label in max_labels:
                    predicted_label = label
                    break
            predicted_labels.append(predicted_label)
        
        arr1 = np.array(predicted_labels)
        arr2 = np.array(test_labels)
        matches = arr1 == arr2
        accuracy = np.count_nonzero(matches)/len(arr1) * 100
        return predicted_labels, test_labels
    
    def access_params(self):
        """
        Print the current parameter values.

        Returns:
        None
        """
        print(self.data.shape)
        print(self.data[0].shape)
        print(f"No of neighbors: {self.neighbors}")
        print(f"Using GPU: {self.use_gpu}")
        return
    
    def modify_params(self, k=None, distance_metric=None):
        """
        Modify the parameter values.

        Parameters:
        - k: int, optional
            The number of neighbors.
        - distance_metric: str, optional
            The distance metric.

        Returns:
        None
        """
        k = int(input("Enter the number of neighbors: "))
        distance_metric = input("Enter the distance metric: ")
        if k is not None:
            self.neighbors = k
        if distance_metric is not None:
            self.distance_metric = self.euclidean_distance if distance_metric == "euclidean" else None
        return
