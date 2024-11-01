import numpy as np
from mlp import MLP

class Autoencoder:
    def __init__(self, input_size, hidden_layer_sizes, hidden_activations, output_activation, 
                 loss_function, learning_rate, n_iter, batch_size):
        self.input_size = input_size
        self.output_size = input_size  # For autoencoder, output size is same as input size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.hidden_activations = hidden_activations
        self.output_activation = output_activation
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.batch_size = batch_size
        
        self.model = MLP(
            input_size=self.input_size,
            output_size=self.output_size,
            hidden_layer_sizes=self.hidden_layer_sizes,
            hidden_activations=self.hidden_activations,
            output_activation=self.output_activation,
            loss_function=self.loss_function,
            learning_rate=self.learning_rate,
            n_iter=self.n_iter,
            batch_size=self.batch_size,
            task='regression'
        )
    
    def fit(self, X_train):
        self.model.fit(X_train, X_train)
    
    def get_latent_vector(self, X):
        activations, _ = self.model.forward_pass(X)
        latent_vectors = activations['A3']  # Assuming the bottleneck is at layer 3
        return latent_vectors
    
    def reconstruct(self, X):
        return self.model.predict(X)

# Example usage
if __name__ == "__main__":
    X_train = np.random.rand(73000, 12)  # Replace with your actual dataset
    print(np.shape(X_train))

    # Define the Autoencoder parameters
    input_size = 12
    hidden_layer_sizes = [10, 7, 10]
    hidden_activations = ['relu','relu', 'relu']
    output_activation = 'linear'
    loss_function = 'mse'
    learning_rate = 0.001
    n_iter = 5000
    batch_size = 50

    # Create an instance of the Autoencoder class
    autoencoder = Autoencoder(
        input_size=input_size,
        hidden_layer_sizes=hidden_layer_sizes,
        hidden_activations=hidden_activations,
        output_activation=output_activation,
        loss_function=loss_function,
        learning_rate=learning_rate,
        n_iter=n_iter,
        batch_size=batch_size
    )

    # Train the autoencoder
    autoencoder.fit(X_train)

    # Get some latent vectors
    latent_vectors = autoencoder.get_latent_vector(X_train[:10])
    print("Latent Vectors:")
    print(latent_vectors)

    # Reconstruct the first 10 inputs
    reconstructed = autoencoder.reconstruct(X_train[:10])
    print("Reconstructed Samples:")
    print(reconstructed)
