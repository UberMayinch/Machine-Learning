from sklearn.neighbors import KNeighborsClassifier
from knn_optimized import KNN_model
from old_knn import KNN_model as OldKNN_model
from performance import performanceMetrics
from a1 import preprocess, split_data
import time
import numpy as np
import matplotlib.pyplot as plt
import gc

print("hi")
# Preprocess data and split into train, validation, and test sets
data = preprocess()
train_data, val_data, test_data = split_data(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)

# Define different training sizes to test
train_sizes = [100,200, 500, 1000, 2000, 5000]

# Initialize lists to store inference times
sk_times = []
my_times = []
old_times = []

# Loop over each training size
for size in train_sizes:
    print(f"\nTraining Size: {size}")
    
    # Sample a subset of the training data
    sampled_train_data = train_data[:size]

    # Sklearn KNN
    sk = KNeighborsClassifier(n_neighbors=15, metric='manhattan')
    sk_start = time.time()
    sk.fit(sampled_train_data[:, :-1], sampled_train_data[:, -1])
    sk_y_pred = sk.predict(test_data[:, :-1])
    sk_end = time.time()
    sk_time_per_inference = (sk_end - sk_start) / len(test_data)
    sk_times.append(sk_time_per_inference)
    skperf = performanceMetrics(sk_y_pred, test_data[:, -1])
    print(f"Sklearn KNN Time per Inference: {sk_time_per_inference:.6f}s")
    print(f"Sklearn KNN Accuracy: {skperf.accuracy():.4f}")
    
    # Optimized KNN
    my = KNN_model(sampled_train_data, k=15, distance_metric='manhattan', use_gpu=True)
    my_start = time.time()
    my_y_pred, my_y = my.inference(test_data)
    my_end = time.time()
    my_time_per_inference = (my_end - my_start) / len(test_data)
    my_times.append(my_time_per_inference)
    myperf = performanceMetrics(my_y_pred, my_y)
    print(f"Optimized KNN Time per Inference: {my_time_per_inference:.6f}s")
    print(f"Optimized KNN Accuracy: {myperf.accuracy():.4f}")
    gc.collect()
    # Old KNN
    if size < 2001:
        old = OldKNN_model(sampled_train_data, k=15, distance_metric='manhattan')
        old_y_pred = []
        old_y = my_y
        old_start = time.time()
        for test in test_data[:50]:  # Only test with the first 50 instances for old KNN
            old_y_pred.append(old.inference(test))
        old_end = time.time()
        old_time_per_inference = (old_end - old_start) / 50
        old_times.append(old_time_per_inference)
        oldperf = performanceMetrics(old_y_pred, old_y[:50])
        print(f"Old KNN Time per Inference: {old_time_per_inference:.6f}s")
        print(f"Old KNN Accuracy: {oldperf.accuracy():.4f}")
    else:
        old_times.append(0)
        print("Old KNN Time per Inference: N/A")
        print("Old KNN Accuracy: N/A")

# Plot Inference Time vs Test Size for each model
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, sk_times, marker='o', label='Sklearn KNN')
plt.plot(train_sizes, my_times, marker='o', label='Optimized KNN')
plt.plot(train_sizes, old_times, marker='o', label='Old KNN')
plt.xlabel('Training Size')
plt.ylabel('Time per Inference (s)')
plt.title('Inference Time vs Training Size')
plt.legend()
plt.grid(True)
plt.show()

# Plot Inference Time vs Model for the whole test set
plt.figure(figsize=(8, 5))
models = ['Sklearn KNN', 'Optimized KNN', 'Old KNN']
times = [sk_times[-1], my_times[-1], old_times[-1]]
plt.bar(models, times, color=['blue', 'orange', 'green'])
plt.ylabel('Time per Inference (s)')
plt.title('Inference Time vs Model for Whole Test Set')
plt.grid(True)
plt.show()
