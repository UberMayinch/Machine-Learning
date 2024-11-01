import numpy as np

class MLP:
    def __init__(self, input_size, output_size, hidden_layer_sizes, activations, task='classification', threshold=0.5, learning_rate=0.01, n_iter=1000, batch_size=None, tol=0.0001):
        self.input_size = input_size  # Size of the input layer
        self.output_size = output_size  # Size of the output layer
        self.hidden_layer_sizes = hidden_layer_sizes  # List of hidden layer sizes
        self.activations = activations  # List of activation functions for hidden layers
        self.task = task  
        self.threshold = threshold  # Threshold for multilabel classification
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.tol = tol

        # Initialize parameters (weights and biases)
        self.layer_sizes = [self.input_size] + self.hidden_layer_sizes + [self.output_size]
        self.params = {}
        for i in range(len(self.layer_sizes) - 1):
            self.params[f'W{i+1}'] = np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1]) * 0.1
            self.params[f'b{i+1}'] = np.zeros((1, self.layer_sizes[i+1]))


    def get_params(self):
        return {
                'input_size': self.input_size,
                'output_size': self.output_size,
                'hidden_layer_sizes': self.hidden_layer_sizes,
                'activations': self.activations,
                'task': self.task,
                'threshold': self.threshold,
                'learning_rate': self.learning_rate,
                'n_iter': self.n_iter,
                'batch_size': self.batch_size
            }

    def set_params(self, **params):
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # Reinitialize parameters if layer sizes or activations are changed
        if 'input_size' in params or 'output_size' in params or 'hidden_layer_sizes' in params:
            self.layer_sizes = [self.input_size] + self.hidden_layer_sizes + [self.output_size]
            self.params = {}
            for i in range(len(self.layer_sizes) - 1):
                self.params[f'W{i+1}'] = np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1]) * 0.1
                self.params[f'b{i+1}'] = np.zeros((1, self.layer_sizes[i+1]))

    # Activation functions
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
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
    
    def linear(self, z):
        return z
    
    def linear_derivative(self, z):
        return np.ones_like(z)
    
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def cross_entropy_loss(self, y_true, y_pred):
        n_samples = y_true.shape[0]
        logp = -np.log(y_pred[range(n_samples), y_true])
        loss = np.sum(logp) / n_samples
        return loss
    
    def rmse_loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    # Forward pass
    def forward_pass(self, X):
        activations = {'A0': X}
        Z = {}  # To store linear transformations (needed for backward pass)

        for i in range(1, len(self.layer_sizes)):
            Z[f'Z{i}'] = np.dot(activations[f'A{i-1}'], self.params[f'W{i}']) + self.params[f'b{i}']
            if i == len(self.layer_sizes) - 1:
                # Output layer based on task
                if self.task == 'classification':
                    activations[f'A{i}'] = self.softmax(Z[f'Z{i}'])
                elif self.task == 'multilabel_classification':
                    activations[f'A{i}'] = self.sigmoid(Z[f'Z{i}'])
                elif self.task == 'regression':
                    activations[f'A{i}'] = self.linear(Z[f'Z{i}'])  # No activation for regression
            else:
                # Apply the chosen activation function for hidden layers
                if self.activations[i-1] == 'sigmoid':
                    activations[f'A{i}'] = self.sigmoid(Z[f'Z{i}'])
                elif self.activations[i-1] == 'relu':
                    activations[f'A{i}'] = self.relu(Z[f'Z{i}'])
                elif self.activations[i-1] == 'tanh':
                    activations[f'A{i}'] = self.tanh(Z[f'Z{i}'])
                elif self.activations[i-1] == 'linear':
                    activations[f'A{i}'] = self.linear(Z[f'Z{i}'])

        return activations, Z

    # Backward pass
    def backward_pass(self, activations, Z, y_true):
        grads = {}
        n_samples = y_true.shape[0]
        y_pred = activations[f'A{len(self.layer_sizes) - 1}']
        y_true = y_true.reshape(len(y_true), 1)

        # Compute gradient based on task
        if self.task == 'classification':
            dz = y_pred
            dz[range(n_samples), y_true] -= 1
            dz /= n_samples
        elif self.task == 'regression':
            dz = (y_pred - y_true)# For regression, we use RMSE (MSE in gradient)
            dz /= n_samples



        for i in range(len(self.layer_sizes) - 1, 0, -1):
            grads[f'dW{i}'] = np.dot(activations[f'A{i-1}'].T, dz)
            grads[f'db{i}'] = np.sum(dz, axis=0, keepdims=True)

            
            if i > 1:
                if self.activations[i-2] == 'sigmoid':
                    dz = np.dot(dz, self.params[f'W{i}'].T) * self.sigmoid_derivative(Z[f'Z{i-1}'])
                elif self.activations[i-2] == 'relu':
                    dz = np.dot(dz, self.params[f'W{i}'].T) * self.relu_derivative(Z[f'Z{i-1}'])
                elif self.activations[i-2] == 'tanh':
                    dz = np.dot(dz, self.params[f'W{i}'].T) * self.tanh_derivative(Z[f'Z{i-1}'])
                elif self.activations[i-2] == 'linear':
                    dz = np.dot(dz, self.params[f'W{i}'].T) * self.linear_derivative(Z[f'Z{i-1}'])

        return grads
    
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
                X_shuffled = X[indices]
                y_shuffled = y[indices]
                
                for start in range(0, n_samples, self.batch_size):
                    end = start + self.batch_size
                    X_batch = X_shuffled[start:end]
                    y_batch = y_shuffled[start:end]
                    
                    activations, Z = self.forward_pass(X_batch)
                    grads = self.backward_pass(activations, Z, y_batch)
                    self.update_params(grads)
            
            # Logging the loss
            if i % 100 == 0 or i == self.n_iter - 1:
                activations, _ = self.forward_pass(X)
                if self.task == 'classification' or self.task == 'multilabel_classification':
                    loss = self.cross_entropy_loss(y, activations[f'A{len(self.layer_sizes) - 1}'])
                elif self.task == 'regression':
                    loss = self.rmse_loss(y, activations[f'A{len(self.layer_sizes) - 1}'])
                print(f'Iteration {i} - Loss: {loss}')
                
                # Early stopping condition
                if abs(prev_loss - loss) < self.tol:
                    print(f'Early stopping at iteration {i} with loss: {loss}')
                    break
                prev_loss = loss
    
    def predict(self, X):
        activations, _ = self.forward_pass(X)
        if self.task == 'classification':
            return np.argmax(activations[f'A{len(self.layer_sizes) - 1}'], axis=1)
        elif self.task == 'multilabel_classification':
            return (activations[f'A{len(self.layer_sizes) - 1}'] >= self.threshold).astype(int)
        elif self.task == 'regression':
            return activations[f'A{len(self.layer_sizes) - 1}']


    def gradient_check(self, X, y, epsilon=1e-5):
        # Perform forward pass
        activations, Z = self.forward_pass(X)
        # Perform backward pass to get analytical gradients
        grads = self.backward_pass(activations, Z, y)
        
        # Numerical gradient checking
        for key in self.params:
            param_shape = self.params[key].shape
            param_grad = np.zeros(param_shape)
            it = np.nditer(self.params[key], flags=['multi_index'], op_flags=['readwrite'])
            
            while not it.finished:
                idx = it.multi_index
                original_value = self.params[key][idx]
                
                # Compute f(theta + epsilon)
                self.params[key][idx] = original_value + epsilon
                activations_plus, _ = self.forward_pass(X)
                if self.task == 'classification' or self.task == 'multilabel_classification':
                    loss_plus = self.cross_entropy_loss(y, activations_plus[f'A{len(self.layer_sizes) - 1}'])
                elif self.task == 'regression':
                    loss_plus = self.rmse_loss(y, activations_plus[f'A{len(self.layer_sizes) - 1}'])
                
                # Compute f(theta - epsilon)
                self.params[key][idx] = original_value - epsilon
                activations_minus, _ = self.forward_pass(X)
                if self.task == 'classification' or self.task == 'multilabel_classification':
                    loss_minus = self.cross_entropy_loss(y, activations_minus[f'A{len(self.layer_sizes) - 1}'])
                elif self.task == 'regression':
                    loss_minus = self.rmse_loss(y, activations_minus[f'A{len(self.layer_sizes) - 1}'])
                
                # Compute numerical gradient
                param_grad[idx] = (loss_plus - loss_minus) / (2 * epsilon)
                
                # Restore original value
                self.params[key][idx] = original_value
                it.iternext()
            
            # Compare numerical gradient with analytical gradient
            if not np.allclose(param_grad, grads[f'd{key}'], atol=1e-5):
                print(f'Gradient check failed for {key}')
                print(f'Numerical gradient: {param_grad}')
                print(f'Analytical gradient: {grads[f"d{key}"]}')
                return False
        
        print('Gradient check passed!')
        return True