import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance_ratio = None

    def fit(self, X):
        # Center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # Compute the covariance matrix
        cov_matrix = np.cov(X_centered, rowvar=False)

        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort eigenvectors by decreasing eigenvalues
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Store the first n_components eigenvectors
        self.components = eigenvectors[:, :self.n_components]

        # Compute explained variance ratio
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio = eigenvalues[:self.n_components] / total_variance

        return self

    def transform(self, X):
        # Center the data
        X_centered = X - self.mean

        # Project the data onto the principal components
        X_transformed = np.dot(X_centered, self.components)

        return X_transformed
    def checkPCA(self, X, tolerance=0.50):
    # Fit the data
        self.fit(X)

    # Transform the data
        X_transformed = self.transform(X)

    # Check if the shape is correct
        if X_transformed.shape[1] != self.n_components:
            return False

    # Check if the variance is preserved (within numerical precision and tolerance)
        original_var = np.var(X, axis=0).sum()
        transformed_var = np.var(X_transformed, axis=0).sum()
        var_preservation = transformed_var / original_var

        if not np.isclose(var_preservation, 1.0, rtol=1.0 - tolerance):
            return False

    # Check if the components are orthogonal
        dot_product = np.dot(self.components.T, self.components)
        if not np.allclose(dot_product, np.eye(self.n_components), atol=1e-6):
            return False

        return True


    def inverse_transform(self, X_transformed):
        # Project back to the original space
        X_reconstructed = np.dot(X_transformed, self.components.T)

        # Add the mean back
        X_reconstructed += self.mean

        return X_reconstructed

    def get_explained_variance_ratio(self):
        return self.explained_variance_ratio