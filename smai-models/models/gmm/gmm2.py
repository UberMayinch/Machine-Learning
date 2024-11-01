
import numpy as np

class GMM:
    def __init__(self, n_components):
        self.n_components = n_components
        self.means = None
        self.covariances = None
        self.weights = None
    
    def fit(self, X, max_iter=100, tol=1e-4):
        # Initialize parameters
        n_samples, n_features = X.shape
        self.means = np.random.randn(self.n_components, n_features)
        self.covariances = np.array([np.eye(n_features)] * self.n_components)
        self.weights = np.ones(self.n_components) / self.n_components
        self._prev_log_likelihood = -np.inf  # Initialize for convergence check

        # Perform expectation maximization
        for _ in range(max_iter):
            # Expectation step
            responsibilities = self._expectation(X)

            # Maximization step
            self._maximization(X, responsibilities)

            # Check convergence
            current_log_likelihood = self._log_likelihood(X)
            if np.abs(current_log_likelihood - self._prev_log_likelihood) < tol:
                break
            self._prev_log_likelihood = current_log_likelihood

        return self

    def getParams(self):
        return self.means, self.covariances, self.weights

    def getMembership(self, X):
        return self._expectation(X)

    def getLikelihood(self, X):
        return self._log_likelihood(X)

    def _expectation(self, X):
        # Compute responsibilities
        weighted_likelihoods = self._compute_weighted_likelihoods(X)
        responsibilities = weighted_likelihoods / np.sum(weighted_likelihoods, axis=1, keepdims=True)
        return responsibilities

    # def _maximization_old(self, X, responsibilities):
    # # Update means
    #     self.means = np.dot(responsibilities.T, X) / np.sum(responsibilities, axis=0)[:, np.newaxis]

    # # Update covariances
    #     for k in range(self.n_components):
    #         diff = X - self.means[k]
    #         self.covariances[k] = np.dot((responsibilities[:, k][:, np.newaxis] * diff).T, diff) / np.sum(responsibilities[:, k])

    # # Update weights
    #     self.weights = np.sum(responsibilities, axis=0) / X.shape[0]

    def _maximization(self, X, responsibilities, reg_covar=1e-6):
        # Update means
        self.means = np.dot(responsibilities.T, X) / np.sum(responsibilities, axis=0)[:, np.newaxis]

        # Update covariances
        for k in range(self.n_components):
            diff = X - self.means[k]
            cov_k = np.dot((responsibilities[:, k][:, np.newaxis] * diff).T, diff) / np.sum(responsibilities[:, k])
        
            # Add regularization term to the covariance matrix
            self.covariances[k] = cov_k + reg_covar * np.eye(X.shape[1])

        # Update weights
        self.weights = np.sum(responsibilities, axis=0) / X.shape[0]
    def _compute_weighted_likelihoods(self, X):
        n_samples, n_features = X.shape
        weighted_likelihoods = np.zeros((n_samples, self.n_components))
    
        for k in range(self.n_components):
            diff = X - self.means[k]
            cov_inv = np.linalg.inv(self.covariances[k])
            exponent = -0.5 * np.sum(np.dot(diff, cov_inv) * diff, axis=1)
        
        # Calculate the normalization factor
            normalization = np.sqrt(np.linalg.det(self.covariances[k]) * (2 * np.pi) ** n_features)
        
            weighted_likelihoods[:, k] = self.weights[k] * np.exp(exponent) / normalization
        
        return weighted_likelihoods


    def _log_likelihood(self, X):
        weighted_likelihoods = self._compute_weighted_likelihoods(X)
        return np.sum(np.log(np.sum(weighted_likelihoods, axis=1)))

    def aic(self, X):
        n_samples = X.shape[0]
        log_likelihood = self._log_likelihood(X)
        n_parameters = self.n_components * (X.shape[1] + X.shape[1] * (X.shape[1] + 1) / 2 + 1)
        aic_value = -2 * log_likelihood + 2 * n_parameters
        return aic_value

    def bic(self, X):
        n_samples = X.shape[0]
        log_likelihood = self._log_likelihood(X)
        n_parameters = self.n_components * (X.shape[1] + X.shape[1] * (X.shape[1] + 1) / 2 + 1)
        bic_value = -2 * log_likelihood + n_parameters * np.log(n_samples)
        return bic_value
