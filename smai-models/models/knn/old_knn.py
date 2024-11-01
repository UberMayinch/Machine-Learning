import numpy as np

class KNN_model:
    def __init__(self, data, k=5, distance_metric="euclidean"):
        self.data = data  # Convert DataFrame to NumPy array
        self.neighbors = k
        self.distance_metric = DistMetrics(distance_metric)
        
    def access_params(self):
        print(f"Data shape: {self.data.shape}")
        print(f"First data point shape: {self.data[0][:-1].shape}")
        print(f"No of neighbors: {self.neighbors}")
        print(f"Distance metric: {self.distance_metric}")
        return

    def modify_params(self, k=None, distance_metric=None):
        k = int(input("Enter the number of neighbors: "))
        distance_metric = input("Enter the distance metric: ")
        if k is not None:
            self.neighbors = k
        if distance_metric is not None:
            self.distance_metric = DistMetrics(distance_metric)
        return
    
    def inference(self, test):
        distances = [] 
        for datapoint in self.data:
            # Compute distance between the test point and training points
            distances.append([self.distance_metric(test[:-1], datapoint[:-1]), datapoint[-1]])
        neighbors = sorted(distances)[:self.neighbors]

        # print(neighbors)  # Debugging print

        # Count the occurrences of each label
        label_counts = {}
        for neighbor in neighbors:
            label = neighbor[1]
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1
        
        # Find the label with the maximum count
        max_count = max(label_counts.values())
        max_labels = [label for label, count in label_counts.items() if count == max_count]
        
        # If there is a tie, select the label with the minimum sum of distances
        min_sum_distance = float('inf')
        selected_label = None
        for label in max_labels:
            sum_distance = sum([distance for distance, neighbor_label in neighbors if neighbor_label == label])
            if sum_distance < min_sum_distance:
                min_sum_distance = sum_distance
                selected_label = label
        
        return selected_label

class DistMetrics:
    def __init__(self, method_name):
        self.methods = {
            "euclidean": self.euclidean,
            "cosine": self.cosine,
            "manhattan": self.manhattan,
        }

        # Select the function based on the method_name provided
        if method_name in self.methods:
            self.selected_method = self.methods[method_name]
        else:
            raise ValueError(f"Method '{method_name}' not found.")

    def euclidean(self, a, b):
        return np.linalg.norm(a - b)

    def cosine(self, a, b):
        S = (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        return 1 - S

    def manhattan(self, a, b):
        return np.sum(np.abs(a - b))

    def __call__(self, a, b):
        # This allows the instance to be called like a function
        return self.selected_method(a, b)


