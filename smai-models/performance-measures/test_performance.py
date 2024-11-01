import numpy as np
from performance import performanceMetrics

def test_compute_confusion_matrix():
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
    label = 1  # Use 0-indexed labels for correct confusion matrix indexing
    expected_tp = 1
    expected_fp = 1
    expected_fn = 1
    expected_tn = 5

    metrics = performanceMetrics(y, y_pred)
    calculated_tp, calculated_fp, calculated_fn, calculated_tn = metrics.label_data(label)

    assert calculated_tp == expected_tp, f"Expected true positives: {expected_tp}, but got: {calculated_tp}"
    assert calculated_fp == expected_fp, f"Expected false positives: {expected_fp}, but got: {calculated_fp}"
    assert calculated_fn == expected_fn, f"Expected false negatives: {expected_fn}, but got: {calculated_fn}"
    assert calculated_tn == expected_tn, f"Expected true negatives: {expected_tn}, but got: {calculated_tn}"

def test_micro_precision():
    y = np.array([1, 2, 3, 4, 1, 2, 3, 4])
    y_pred = np.array([1, 2, 3, 4, 2, 1, 4, 3])
    expected_precision = 0.5  # Calculated manually for this case

    metrics = performanceMetrics(y, y_pred)
    calculated_precision = metrics.micro_precision()

    assert np.isclose(calculated_precision, expected_precision), f"Expected micro precision: {expected_precision}, but got: {calculated_precision}"

def test_micro_recall():
    y = np.array([1, 2, 3, 4, 1, 2, 3, 4])
    y_pred = np.array([1, 2, 3, 4, 2, 1, 4, 3])
    expected_recall = 0.5  # Calculated manually for this case

    metrics = performanceMetrics(y, y_pred)
    calculated_recall = metrics.micro_recall()

    assert np.isclose(calculated_recall, expected_recall), f"Expected micro recall: {expected_recall}, but got: {calculated_recall}"

def test_micro_f1_score():
    y = np.array([1, 2, 3, 4, 1, 2, 3, 4])
    y_pred = np.array([1, 2, 3, 4, 2, 1, 4, 3])
    expected_f1_score = 0.5  # Calculated manually for this case

    metrics = performanceMetrics(y, y_pred)
    calculated_f1_score = metrics.micro_f1_score()

    assert np.isclose(calculated_f1_score, expected_f1_score), f"Expected micro F1 score: {expected_f1_score}, but got: {calculated_f1_score}"

def test_macro_precision():
    y = np.array([1, 2, 3, 4, 1, 2, 3, 4])
    y_pred = np.array([1, 2, 3, 4, 2, 1, 4, 3])
    expected_precision = 0.5  # Calculated manually for this case

    metrics = performanceMetrics(y, y_pred)
    calculated_precision = metrics.macro_precision()

    assert np.isclose(calculated_precision, expected_precision), f"Expected macro precision: {expected_precision}, but got: {calculated_precision}"

def test_macro_recall():
    y = np.array([1, 2, 3, 4, 1, 2, 3, 4])
    y_pred = np.array([1, 2, 3, 4, 2, 1, 4, 3])
    expected_recall = 0.5  # Calculated manually for this case

    metrics = performanceMetrics(y, y_pred)
    calculated_recall = metrics.macro_recall()

    assert np.isclose(calculated_recall, expected_recall), f"Expected macro recall: {expected_recall}, but got: {calculated_recall}"

def test_macro_f1_score():
    y = np.array([1, 2, 3, 4, 1, 2, 3, 4])
    y_pred = np.array([1, 2, 3, 4, 2, 1, 4, 3])
    expected_f1_score = 0.5  # Calculated manually for this case

    metrics = performanceMetrics(y, y_pred)
    calculated_f1_score = metrics.macro_f1_score()

    assert np.isclose(calculated_f1_score, expected_f1_score), f"Expected macro F1 score: {expected_f1_score}, but got: {calculated_f1_score}"

# Run the test cases
test_compute_confusion_matrix()
test_label_data()
test_micro_precision()
test_micro_recall()
test_micro_f1_score()
test_macro_precision()
test_macro_recall()
test_macro_f1_score()
