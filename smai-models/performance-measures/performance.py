import numpy as np
import matplotlib.pyplot as plt

class performanceMetrics:
    def __init__(self, y, y_pred):
        self.y = y
        self.y_pred = y_pred
        self.confmat = self.compute_confusion_matrix()

    def accuracy(self):
        arr1 = np.array(self.y)
        arr2 = np.array(self.y_pred)
        matches = arr1 == arr2
        accuracy = np.count_nonzero(matches) / len(arr1) * 100
        return accuracy
    
    def compute_confusion_matrix(self):
        labels = np.unique(np.concatenate((self.y, self.y_pred)))
        label_to_index = {label: idx for idx, label in enumerate(labels)}
        mat = np.zeros((len(labels), len(labels)))

        for true, pred in zip(self.y, self.y_pred):
            mat[label_to_index[true]][label_to_index[pred]] += 1
        
        # Add a small regularization term to avoid division by zero
        mat += 1e-10
        
        return mat

    def plot_confusion_matrix(self):
        mat = self.confmat
        labels = np.unique(np.concatenate((self.y, self.y_pred)))
        plt.imshow(mat, cmap='hot', interpolation='nearest')
        plt.title("Confusion Matrix")
        plt.colorbar()
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()

    def label_data(self, label):
        label_to_index = {label: idx for idx, label in enumerate(np.unique(np.concatenate((self.y, self.y_pred))))}
        idx = label_to_index[label]
        tp = self.confmat[idx, idx]
        fp = np.sum(self.confmat[:, idx]) - tp
        fn = np.sum(self.confmat[idx, :]) - tp
        tn = np.sum(self.confmat) - (tp + fp + fn)

        return tp, fp, fn, tn

    def micro_precision(self):
        tp_sum = np.sum(np.diag(self.confmat))
        fp_sum = np.sum(self.confmat) - tp_sum
        if tp_sum + fp_sum == 0:
            return 0.0
        precision = tp_sum / (tp_sum + fp_sum)
        return precision

    def micro_recall(self):
        tp_sum = np.sum(np.diag(self.confmat))
        fn_sum = np.sum(np.sum(self.confmat, axis=1) - np.diag(self.confmat))
        if tp_sum + fn_sum == 0:
            return 0.0
        recall = tp_sum / (tp_sum + fn_sum)
        return recall

    def micro_f1_score(self):
        precision = self.micro_precision()
        recall = self.micro_recall()
        if precision + recall == 0:
            return 0.0
        f1_score = 2 * (precision * recall) / (precision + recall)
        return f1_score

    def macro_precision(self):
        precisions = np.diag(self.confmat) / np.sum(self.confmat, axis=0)
        precisions = np.nan_to_num(precisions)  # Handle NaNs due to division by zero
        precision = np.mean(precisions)
        return precision

    def macro_recall(self):
        recalls = np.diag(self.confmat) / np.sum(self.confmat, axis=1)
        recalls = np.nan_to_num(recalls)  # Handle NaNs due to division by zero
        recall = np.mean(recalls)
        return recall

    def macro_f1_score(self):
        precision = self.macro_precision()
        recall = self.macro_recall()
        if precision + recall == 0:
            return 0.0
        f1_score = 2 * (precision * recall) / (precision + recall)
        return f1_score

    def printMetrics(self):
        print(f"Accuracy: {self.accuracy()}")
        print(f"Micro Precision: {self.micro_precision()}")
        print(f"Micro Recall: {self.micro_recall()}")
        print(f"Micro F1 Score: {self.micro_f1_score()}")
        print(f"Macro Precision: {self.macro_precision()}")
        print(f"Macro Recall: {self.macro_recall()}")
        print(f"Macro F1 Score: {self.macro_f1_score()}")

