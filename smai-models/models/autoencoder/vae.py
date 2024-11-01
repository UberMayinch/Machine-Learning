import numpy as np
from mlp import MLP

class VariationalAutoencoder:
    def __init__(self, input_size, hidden_layer_sizes, hidden_activations, output_activation, 
                 loss_function, learning_rate, n_iter, batch_size, latent_dim):
        self.input_size = input_size
        self.output_size = input_size  # For autoencoder, output size is same as input size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.hidden_activations = hidden_activations
        self.output_activation = output_activation
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = MLP(
            input_size=self.input_size,
            output_size=2*self.latent_dim,  # Mean and log variance
            hidden_layer_sizes=self.hidden_layer_sizes[:-1],
            hidden_activations=self.hidden_activations[:-1],
            output_activation='linear',
            loss_function='mse',
            learning_rate=self.learning_rate,
            n_iter=self.n_iter,
            batch_size=self.batch_size,
            task='regression'
        )
        
        # Decoder
        self.decoder = MLP(
            input_size=self.latent_dim,
            output_size=self.output_size,
            hidden_layer_sizes=self.hidden_layer_sizes[::-1],
            hidden_activations=self.hidden_activations[::-1],
            output_activation=self.output_activation,
            loss_function=self.loss_function,
            learning_rate=self.learning_rate,
            n_iter=self.n_iter,
            batch_size=self.batch_size,
            task='regression'
        )
    
    def reparameterize(self, mean, logvar):
        std = np.exp(0.5 * logvar)
        eps = np.random.randn(*mean.shape)
        return mean + eps * std
    
    def fit(self, X_train):
        for _ in range(self.n_iter):
            # Mini-batch training
            indices = np.random.choice(X_train.shape[0], self.batch_size, replace=False)
            X_batch = X_train[indices]
            
            # Forward pass
            mean, logvar = np.split(self.encoder.predict(X_batch), 2, axis=1)
            z = self.reparameterize(mean, logvar)
            X_recon = self.decoder.predict(z)
            
            # Compute losses
            reconstruction_loss = np.mean((X_batch - X_recon)**2)
            kl_loss = -0.5 * np.sum(1 + logvar - np.square(mean) - np.exp(logvar), axis=1)
            kl_loss = np.mean(kl_loss)
            
            total_loss = reconstruction_loss + kl_loss
            
            # Backpropagation (this is simplified, you'd need to implement proper backprop)
            self.encoder.backpropagation(X_batch, np.concatenate([mean, logvar], axis=1))
            self.decoder.backpropagation(z, X_batch)
    
    def get_latent_vector(self, X):
        mean, logvar = np.split(self.encoder.predict(X), 2, axis=1)
        return self.reparameterize(mean, logvar)
    
    def reconstruct(self, X):
        latent = self.get_latent_vector(X)
        return self.decoder.predict(latent)

# Example usage
if __name__ == "__main__":
    X_train = np.random.rand(73000, 12)  # Replace with your actual dataset
    print(np.shape(X_train))
    
    # Define the VAE parameters
    input_size = 12
    hidden_layer_sizes = [10, 7, 10]
    hidden_activations = ['relu', 'relu', 'relu']
    output_activation = 'linear'
    loss_function = 'mse'
    learning_rate = 0.001
    n_iter = 5000
    batch_size = 50
    latent_dim = 7  # Dimension of the latent space
    
    # Create an instance of the VariationalAutoencoder class
    vae = VariationalAutoencoder(
        input_size=input_size,
        hidden_layer_sizes=hidden_layer_sizes,
        hidden_activations=hidden_activations,
        output_activation=output_activation,
        loss_function=loss_function,
        learning_rate=learning_rate,
        n_iter=n_iter,
        batch_size=batch_size,
        latent_dim=latent_dim
    )
    
    # Train the VAE
    vae.fit(X_train)
    
    # Get some latent vectors
    latent_vectors = vae.get_latent_vector(X_train[:10])
    print("Latent Vectors:")
    print(latent_vectors)
    
    # Reconstruct the first 10 inputs
    reconstructed = vae.reconstruct(X_train[:10])
    print("Reconstructed Samples:")
    print(reconstructed)