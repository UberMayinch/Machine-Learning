import numpy as np

class MultiLabelMLP:
    def __init__(self, input_size, num_classes, hidden_layer_sizes, hidden_activations, learning_rate=0.01, n_iter=1000, batch_size=None, tol=0.0001):
        self.input_size = input_size
        self.num_classes = num_classes
        self.hidden_layer_sizes = hidden_layer_sizes
        self.hidden_activations = hidden_activations
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.tol = tol

        self.layer_sizes = [self.input_size] + self.hidden_layer_sizes + [self.num_classes]
        self.params = {}
        for i in range(len(self.layer_sizes)-1):
            self.params[f'W{i+1}'] = np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1]) / np.sqrt(self.layer_sizes[i])
            self.params[f'b{i+1}'] = np.zeros((1, self.layer_sizes[i+1]))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -709, 709)))
    
    def sigmoid_derivative(self, z):
        sig = self.sigmoid(z)
        return sig * (1 - sig)
    
    def relu(self, z):
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        return np.where(z > 0, 1, 0)
    
    def tanh(self, z):
        return np.tanh(z)
    
    def tanh_derivative(self, z):
        return 1 - np.tanh(z) ** 2

    def apply_activation(self, z, activation):
        if activation == 'sigmoid':
            return self.sigmoid(z)
        elif activation == 'relu':
            return self.relu(z)
        elif activation == 'tanh':
            return self.tanh(z)
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

    def apply_activation_derivative(self, z, activation):
        if activation == 'sigmoid':
            return self.sigmoid_derivative(z)
        elif activation == 'relu':
            return self.relu_derivative(z)
        elif activation == 'tanh':
            return self.tanh_derivative(z)
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

    def forward_pass(self, X):
        activations = {'A0': X}
        Z = {}

        for i in range(1, len(self.layer_sizes)):
            Z[f'Z{i}'] = np.dot(activations[f'A{i-1}'], self.params[f'W{i}']) + self.params[f'b{i}']
            if i == len(self.layer_sizes) - 1:
                activations[f'A{i}'] = self.sigmoid(Z[f'Z{i}'])  # Output layer uses sigmoid for multilabel
            else:
                activations[f'A{i}'] = self.apply_activation(Z[f'Z{i}'], self.hidden_activations[i-1])

        return activations, Z

    def backward_pass(self, activations, Z, y_true):
        grads = {}
        m = y_true.shape[0]
        L = len(self.layer_sizes) - 1
        y_pred = activations[f'A{L}']
        y_true = y_true.reshape(y_pred.shape)
        
        # Gradient for the output layer
        dZ = (y_pred - y_true) * self.sigmoid_derivative(Z[f'Z{L}'])
        
        for i in range(L, 0, -1):
            grads[f'dW{i}'] = np.dot(activations[f'A{i-1}'].T, dZ) / m
            grads[f'db{i}'] = np.sum(dZ, axis=0, keepdims=True) / m
            
            if i > 1:
                dZ = np.dot(dZ, self.params[f'W{i}'].T) * self.apply_activation_derivative(Z[f'Z{i-1}'], self.hidden_activations[i-2])

        return grads

   
    def hamming_loss(self, y_true, y_pred):
        y_pred_binary = (y_pred >= 0.5).astype(int)
        return np.mean(np.sum(y_true != y_pred_binary, axis=1) / y_true.shape[1])

    def update_params(self, grads):
        for i in range(1, len(self.layer_sizes)):
            self.params[f'W{i}'] -= self.learning_rate * grads[f'dW{i}']
            self.params[f'b{i}'] -= self.learning_rate * grads[f'db{i}']

    def fit(self, X, y):
        n_samples = X.shape[0]
        prev_loss = float('inf')
        
        for i in range(self.n_iter):
            if self.batch_size is None:
                activations, Z = self.forward_pass(X)
                grads = self.backward_pass(activations, Z, y)
                self.update_params(grads)
            else:
                indices = np.random.permutation(n_samples)
                for start in range(0, n_samples, self.batch_size):
                    end = start + self.batch_size
                    X_batch = X[indices[start:end]]
                    y_batch = y[indices[start:end]]
                    
                    activations, Z = self.forward_pass(X_batch)
                    grads = self.backward_pass(activations, Z, y_batch)
                    self.update_params(grads)
            
            if i % 50 == 0 or i == self.n_iter - 1:
                activations, _ = self.forward_pass(X)
                loss = self.hamming_loss(y, activations[f'A{len(self.layer_sizes) - 1}'])
                print(f'Iteration {i} - Hamming Loss: {loss}')
                
                if abs(prev_loss - loss) < self.tol:
                    print(f'Early stopping at iteration {i} with Hamming Loss: {loss}')
                    break
                prev_loss = loss

    def predict(self, X, threshold=0.5):
        activations, _ = self.forward_pass(X)
        y_pred = activations[f'A{len(self.layer_sizes) - 1}']
        return (y_pred >= threshold).astype(int)

    def predict_proba(self, X):
        activations, _ = self.forward_pass(X)
        return activations[f'A{len(self.layer_sizes) - 1}']



##BCE Implementatoin

# class MultiLabelMLP:
#     def __init__(self, input_size, num_classes, hidden_layer_sizes, hidden_activations, learning_rate=0.01, n_iter=1000, batch_size=None, tol=0.0001):
#         self.input_size = input_size
#         self.num_classes = num_classes
#         self.hidden_layer_sizes = hidden_layer_sizes
#         self.hidden_activations = hidden_activations
#         self.learning_rate = learning_rate
#         self.n_iter = n_iter
#         self.batch_size = batch_size
#         self.tol = tol

#         self.layer_sizes = [self.input_size] + self.hidden_layer_sizes + [self.num_classes]
#         self.params = {}
#         for i in range(len(self.layer_sizes)-1):
#             self.params[f'W{i+1}'] = np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1]) / np.sqrt(self.layer_sizes[i])
#             self.params[f'b{i+1}'] = np.zeros((1, self.layer_sizes[i+1]))

#     def sigmoid(self, z):
#         return 1 / (1 + np.exp(-np.clip(z, -709, 709)))
    
#     def relu(self, z):
#         return np.maximum(0, z)
    
#     def relu_derivative(self, z):
#         return np.where(z > 0, 1, 0)
    
#     def apply_activation(self, z, activation):
#         if activation == 'sigmoid':
#             return self.sigmoid(z)
#         elif activation == 'relu':
#             return self.relu(z)
#         else:
#             raise ValueError(f"Unsupported activation function: {activation}")

#     def apply_activation_derivative(self, z, activation):
#         if activation == 'sigmoid':
#             return self.sigmoid_derivative(z)
#         elif activation == 'relu':
#             return self.relu_derivative(z)
#         else:
#             raise ValueError(f"Unsupported activation function: {activation}")

#     def forward_pass(self, X):
#         activations = {'A0': X}
#         Z = {}

#         for i in range(1, len(self.layer_sizes)):
#             Z[f'Z{i}'] = np.dot(activations[f'A{i-1}'], self.params[f'W{i}']) + self.params[f'b{i}']
#             if i == len(self.layer_sizes) - 1:
#                 activations[f'A{i}'] = self.sigmoid(Z[f'Z{i}'])  # Output layer uses sigmoid for multilabel
#             else:
#                 activations[f'A{i}'] = self.apply_activation(Z[f'Z{i}'], self.hidden_activations[i-1])

#         return activations, Z

#     def backward_pass(self, activations, Z, y_true):
#         grads = {}
#         m = y_true.shape[0]
#         L = len(self.layer_sizes) - 1
#         y_pred = activations[f'A{L}']
#         y_true = y_true.reshape(y_pred.shape)
        
#         # Gradient for the output layer (Binary Cross-Entropy)
#         dZ = (y_pred - y_true)
        
#         for i in range(L, 0, -1):
#             grads[f'dW{i}'] = np.dot(activations[f'A{i-1}'].T, dZ) / m
#             grads[f'db{i}'] = np.sum(dZ, axis=0, keepdims=True) / m
            
#             if i > 1:
#                 dZ = np.dot(dZ, self.params[f'W{i}'].T) * self.apply_activation_derivative(Z[f'Z{i-1}'], self.hidden_activations[i-2])

#         return grads

#     def binary_cross_entropy_loss(self, y_true, y_pred):
#         return -np.mean(y_true * np.log(y_pred + 1e-15) + (1 - y_true) * np.log(1 - y_pred + 1e-15))

#     def update_params(self, grads):
#         for i in range(1, len(self.layer_sizes)):
#             self.params[f'W{i}'] -= self.learning_rate * grads[f'dW{i}']
#             self.params[f'b{i}'] -= self.learning_rate * grads[f'db{i}']

#     def fit(self, X, y):
#         n_samples = X.shape[0]
#         prev_loss = float('inf')
        
#         for i in range(self.n_iter):
#             if self.batch_size is None:
#                 activations, Z = self.forward_pass(X)
#                 grads = self.backward_pass(activations, Z, y)
#                 self.update_params(grads)
#             else:
#                 indices = np.random.permutation(n_samples)
#                 for start in range(0, n_samples, self.batch_size):
#                     end = start + self.batch_size
#                     X_batch = X[indices[start:end]]
#                     y_batch = y[indices[start:end]]
                    
#                     activations, Z = self.forward_pass(X_batch)
#                     grads = self.backward_pass(activations, Z, y_batch)
#                     self.update_params(grads)
            
#             if i % 50 == 0 or i == self.n_iter - 1:
#                 activations, _ = self.forward_pass(X)
#                 loss = self.binary_cross_entropy_loss(y, activations[f'A{len(self.layer_sizes) - 1}'])
#                 print(f'Iteration {i} - Binary Cross-Entropy Loss: {loss}')
                
#                 if abs(prev_loss - loss) < self.tol:
#                     print(f'Early stopping at iteration {i} with Binary Cross-Entropy Loss: {loss}')
#                     break
#                 prev_loss = loss

#     def predict(self, X, threshold=0.5):
#         activations, _ = self.forward_pass(X)
#         y_pred = activations[f'A{len(self.layer_sizes) - 1}']
#         return (y_pred >= threshold).astype(int)

#     def predict_proba(self, X):
#         activations, _ = self.forward_pass(X)
#         return activations[f'A{len(self.layer_sizes) - 1}']
