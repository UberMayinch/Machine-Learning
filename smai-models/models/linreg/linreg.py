import numpy as np

class PolyRegression:
    def __init__(self, x, y, degree=1):
        self.x = np.array(x, dtype=np.float64)
        self.y = np.array(y, dtype=np.float64)
        self.degree = degree
        self.coeffs = np.random.rand(degree+1).astype(np.float64)
        
        # Scale features
        self.x_mean = np.mean(self.x)
        self.x_std = np.std(self.x)
        self.x = (self.x - self.x_mean) / self.x_std

    def MSE(self, x, y, coeffs, l1=0, l2=0):
        x_poly = np.vander(x, self.degree+1, increasing=True)
        return np.sum((y - np.dot(x_poly, coeffs)) ** 2) + l1 * np.sum(np.abs(coeffs)) + l2 * np.sum(coeffs**2)

    def sd(self, x, y, coeffs):
        x_poly = np.vander(x, self.degree+1, increasing=True)
        return np.std(y - np.dot(x_poly, coeffs))

    def variance(self, x, y, coeffs):
        x_poly = np.vander(x, self.degree+1, increasing=True)
        return np.var(y - np.dot(x_poly, coeffs))
    
    def grad_descent(self, x, y, coeffs, max_iter=1000, learning_rate=0.001, tolerance=0.0001, l1=0, l2=0):
        N = len(x)
        x_poly = np.vander(x, self.degree+1, increasing=True)
        _ = 0
        coeffs = np.array(coeffs, dtype=np.float64)
        ## only for part 3
        coeff_list = [coeffs]
        loss = self.MSE(x, y, coeffs, l1, l2)
        while _ < max_iter and loss > tolerance:
            grad = -2/N * np.dot(x_poly.T, (y - np.dot(x_poly, coeffs))) + l1 * np.sign(coeffs) + 2 * l2 * coeffs
            
            # print(coeffs)
         # Gradient clipping
            max_grad = 1.0
            grad_norm = np.linalg.norm(grad)
            if grad_norm > max_grad:
                grad = grad * max_grad / grad_norm

            coeffs -= learning_rate * grad
            loss = self.MSE(x, y, coeffs, l1, l2)
            _ += 1

            # For Part 3 only
            if _ % 10 == 0:
                coeffs_to_add = np.copy(coeffs)
                coeff_list.append(coeffs_to_add)
            
        return coeff_list
        return coeffs

    def fit(self, max_iter=1000, learning_rate=0.001, tolerance=0.0001, l1=0, l2=0):
        self.coeffs = self.grad_descent(self.x, self.y, self.coeffs, max_iter, learning_rate, tolerance, l1, l2)
        return self.coeffs

    def predict(self, x):
        x = np.array(x, dtype=np.float64)
        x_scaled = (x - self.x_mean) / self.x_std  # Scale the features
        x_poly = np.vander(x_scaled, self.degree+1, increasing=True)
        return np.dot(x_poly, self.coeffs)
