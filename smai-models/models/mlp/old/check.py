import numpy as np

class MLP:
    def __init__(self, input_size, output_size, hidden_layer_sizes, hidden_activations, output_activation, loss_function, learning_rate=0.01, n_iter=1000, batch_size=None, tol=0.0001):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.hidden_activations = hidden_activations
        self.output_activation = output_activation
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.tol = tol

        self.layer_sizes = [self.input_size] + self.hidden_layer_sizes + [self.output_size]
        self.params = {}
        for i in range(len(self.layer_sizes)-1):
            self.params[f'W{i+1}'] = np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1]) / np.sqrt(self.layer_sizes[i])
            self.params[f'b{i+1}'] = np.zeros((1, self.layer_sizes[i+1]))

    def get_params(self):
        return {
            'input_size': self.input_size,
            'output_size': self.output_size,
            'hidden_layer_sizes': self.hidden_layer_sizes,
            'hidden_activations': self.hidden_activations,
            'output_activation': self.output_activation,
            'loss_function': self.loss_function,
            'learning_rate': self.learning_rate,
            'n_iter': self.n_iter,
            'batch_size': self.batch_size,
            'tol': self.tol
        }

    def set_params(self, **params):
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        if 'input_size' in params or 'output_size' in params or 'hidden_layer_sizes' in params:
            self.layer_sizes = [self.input_size] + self.hidden_layer_sizes + [self.output_size]
            self.params = {}
            for i in range(len(self.layer_sizes) - 1):
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
    
    def linear(self, z):
        return z
    
    def linear_derivative(self, z):
        return np.ones_like(z)
    
    def mse_loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def forward_pass(self, X):
        activations = {'A0': X}
        Z = {}

        for i in range(1, len(self.layer_sizes)):
            Z[f'Z{i}'] = np.dot(activations[f'A{i-1}'], self.params[f'W{i}']) + self.params[f'b{i}']
            if i == len(self.layer_sizes) - 1:
                activations[f'A{i}'] = self.apply_activation(Z[f'Z{i}'], self.output_activation)
            else:
                activations[f'A{i}'] = self.apply_activation(Z[f'Z{i}'], self.hidden_activations[i-1])

        return activations, Z

    def apply_activation(self, z, activation):
        if activation == 'sigmoid':
            return self.sigmoid(z)
        elif activation == 'relu':
            return self.relu(z)
        elif activation == 'tanh':
            return self.tanh(z)
        elif activation == 'linear':
            return self.linear(z)
        elif activation == 'softmax':
            return self.softmax(z)
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

    def apply_activation_derivative(self, z, activation):
        if activation == 'sigmoid':
            return self.sigmoid_derivative(z)
        elif activation == 'relu':
            return self.relu_derivative(z)
        elif activation == 'tanh':
            return self.tanh_derivative(z)
        elif activation == 'linear':
            return self.linear_derivative(z)
        elif activation == 'softmax':
            return 1  # For softmax, we handle this separately in backward_pass
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
    def backward_pass(self, activations, Z, y_true):
        grads = {}
        m = y_true.shape[0]
        L = len(self.layer_sizes) - 1
        y_pred = activations[f'A{L}']
    
    # Compute initial dZ for the output layer
        if self.loss_function == 'cross_entropy':
            if self.output_activation == 'softmax':
                dZ = y_pred.copy()
                dZ[np.arange(m), y_true] -= 1
                dZ /= m
            else:
                dZ = (y_pred - y_true) * self.apply_activation_derivative(Z[f'Z{L}'], self.output_activation)
        elif self.loss_function == 'mse':
            dZ = (y_pred - y_true) * self.apply_activation_derivative(Z[f'Z{L}'], self.output_activation) / m
            dZ /= m
            dz *= 2

        # Loop backward through layers
        for i in range(L, 0, -1):
            grads[f'dW{i}'] = np.dot(activations[f'A{i-1}'].T, dZ) / m
            grads[f'db{i}'] = np.sum(dZ, axis=0, keepdims=True) / m
        
            if i > 1:
                dZ = np.dot(dZ, self.params[f'W{i}'].T) * self.apply_activation_derivative(Z[f'Z{i-1}'], self.hidden_activations[i-2])

        return grads

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def mse_loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def cross_entropy_loss(self, y_true, y_pred):
        n_samples = y_true.shape[0]
        return -np.sum(np.log(y_pred[np.arange(n_samples), y_true] + 1e-10)) / n_samples

    def compute_loss(self, y_true, y_pred):
        if self.loss_function == 'cross_entropy':
            if y_true.ndim == 1:  # If y_true is a vector of class labels
                return self.cross_entropy_loss(y_true, y_pred)
            else:  # If y_true is one-hot encoded
                return -np.mean(np.sum(y_true * np.log(y_pred + 1e-10), axis=1))
        elif self.loss_function == 'mse':
            return self.mse_loss(y_true, y_pred)
        else:
            raise ValueError(f"Unsupported loss function: {self.loss_function}")

    def predict(self, X):
        activations, _ = self.forward_pass(X)
        output = activations[f'A{len(self.layer_sizes) - 1}']
        if self.output_activation == 'softmax':
            return np.argmax(output, axis=1)
        else:
            return output


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
                
                if i % 100 == 0 or i == self.n_iter - 1:
                    activations, _ = self.forward_pass(X)
                    loss = self.compute_loss(y, activations[f'A{len(self.layer_sizes) - 1}'])
                    print(f'Iteration {i} - Loss: {loss}')
                    
                    if abs(prev_loss - loss) < self.tol:
                        print(f'Early stopping at iteration {i} with loss: {loss}')
                        break
                    prev_loss = loss

    def gradient_check(self, X, y, epsilon=1e-7):
        activations, Z = self.forward_pass(X)
        grads = self.backward_pass(activations, Z, y)

        for key in self.params:
            param_grad = np.zeros_like(self.params[key])
            it = np.nditer(self.params[key], flags=['multi_index'], op_flags=['readwrite'])

            while not it.finished:
                idx = it.multi_index
                old_value = self.params[key][idx]

                # Calculate loss with increased parameter
                self.params[key][idx] = old_value + epsilon
                activations_plus, _ = self.forward_pass(X)
                loss_plus = self.compute_loss(y, activations_plus[f'A{len(self.layer_sizes) - 1}'])

                # Calculate loss with decreased parameter
                self.params[key][idx] = old_value - epsilon
                activations_minus, _ = self.forward_pass(X)
                loss_minus = self.compute_loss(y, activations_minus[f'A{len(self.layer_sizes) - 1}'])

                param_grad[idx] = (loss_plus - loss_minus) / (2 * epsilon)

                # Restore the parameter to its original value
                self.params[key][idx] = old_value
                it.iternext()

            # Determine correct key for gradients
            if 'W' in key:
                layer_index = key[-1]  # Assumes keys end with layer index (e.g., W1, W2, ...)
                grad_key = f'dW{layer_index}'
            else:
                layer_index = key[-1]  # Assumes keys end with layer index (e.g., b1, b2, ...)
                grad_key = f'db{layer_index}'

            # Compare numerical and analytical gradients
            if not np.allclose(param_grad, grads[grad_key], rtol=1e-5, atol=1e-5):
                print(f'Gradient check failed for {key}')
                print(f'Numerical gradient: {param_grad}')
                print(f'Analytical gradient: {grads[grad_key]}')
                return False

        print('Gradient check passed!')
        return True
