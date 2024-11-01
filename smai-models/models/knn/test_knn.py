import numpy as np
from performance import performanceMetrics
import numpy as np
import pytest
from models.knn.knn_optimized import KNN_model

def test_confusion_matrix():
    y = np.array([1, 2, 3, 4, 1, 2, 3, 4])
    y_pred = np.array([1, 2, 3, 4, 2, 1, 4, 3])
    expected_matrix = np.array([[1, 1, 0, 0],
                                [1, 1, 0, 0],
                                [0, 0, 1, 1],
                                [0, 0, 1, 1]])

    metrics = performanceMetrics(y, y_pred)
    calculated_matrix = metrics.compute_confusion_matrix()

    assert np.array_equal(calculated_matrix, expected_matrix), "The generated confusion matrix does not match the expected matrix"
    
def test_label_data():
        y = np.array([1, 2, 3, 4, 1, 2, 3, 4])
        y_pred = np.array([1, 2, 3, 4, 2, 1, 4, 3])
        label = 2
        expected_tp = 1
        expected_fp = 1
        expected_fn = 2
        expected_tn = 4

        metrics = performanceMetrics(y, y_pred)
        calculated_tp, calculated_fp, calculated_fn, calculated_tn = metrics.label_data(label)

        assert calculated_tp == expected_tp, f"Expected true positives: {expected_tp}, but got: {calculated_tp}"
        assert calculated_fp == expected_fp, f"Expected false positives: {expected_fp}, but got: {calculated_fp}"
        assert calculated_fn == expected_fn, f"Expected false negatives: {expected_fn}, but got: {calculated_fn}"
        assert calculated_tn == expected_tn, f"Expected true negatives: {expected_tn}, but got: {calculated_tn}"

def test_inference():
    # Create a KNN_model instance
    data = np.array([[1, 2, 3, 0],
                        [4, 5, 6, 1],
                    [7, 8, 9, 0],
                        [10, 11, 12, 1]])
    knn_model = KNN_model(data, k=1, distance_metric="euclidean")

            # Test inference with a test data point
    test_data = np.array([5, 4, 7])
    predicted_label = knn_model.inference(test_data)

            # Check if the predicted label matches the expected label
    expected_label = 1
    assert predicted_label == expected_label, f"Expected label: {expected_label}, but got: {predicted_label}"

    
test_inference()
        
# test_label_data()

# test_confusion_matrix()