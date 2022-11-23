import numpy as np
from scipy.optimize import minimize, Bounds

class Dual_SVM:
    def __init__(self, kernel_type, gamma = 0.0, C=1.0):
        self.C = C # C is the regularization parameter
        self.lambdas = None
        self.w = None # w is the weight vector
        self.b = None # b is the bias
        self.X = None 
        self.y = None
        self.gamma = gamma
        self.support_vectors = None
        self.overlapping_support_vectors = None
        if kernel_type == "linear":
            self.kernel = self.linear_kernel # linear kernel
        elif kernel_type == "gaussian":
            self.kernel = self.gaussian_kernel # gaussian kernel

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        self.X = X
        self.y = y
        # Initialize lambdas
        self.lambdas = np.zeros(n_samples) # lambdas are the dual variables of the primal problem (the weights of the primal problem)

        # Define the objective function
        def objective_function(lambdas):
            out = -np.sum(lambdas) + 0.5 * np.dot(self.lambdas, np.dot(self.lambdas.T, (self.y@self.y.T) * self.kernel(self.X, self.X)))
            return out

        # Define the constraints
        constraints = ({'type': 'eq', 'fun': self.constraints})
        bounds = Bounds(0, self.C) # 0 <= lambdas <= C

        # Define the initial guess
        initial_guess = np.zeros(n_samples) # lambdas = 0 (initial guess is 0)

        # Run the optimization
        solution = minimize(fun=objective_function, x0=initial_guess, bounds=bounds, method='SLSQP', constraints=constraints) # minimize the objective function using Sequential Least SQuares Programming (SLSQP) method
        self.lambdas = solution.x # lambdas are the solution of the optimization
        # count the support vectors
        self.support_vectors = np.where(self.lambdas > 1e-5)[0] # support vectors are the lambdas that are greater than 1e-5
        # count the overlapping support vectors
        self.overlapping_support_vectors = np.where((self.lambdas > 1e-5) & (self.lambdas < self.C))[0] # overlapping support vectors are the lambdas that are greater than 1e-5 and less than C
        # Calculate w and b
        self.w = np.dot(self.lambdas * self.y, self.X) # w = sum(lambdas * y * X) (dot product of lambdas, y and X) 
        self.b = np.dot(self.lambdas, self.y)

    def constraints(self, lambdas):
        # lambdas * y = 0 (dot product of lambdas and y is 0)
        return np.dot(lambdas.T, self.y)

    def predict(self, X: np.ndarray):
        # predict using kernel trick
        prediction_res = []

        for i in range(len(X)):
            prediction = np.sign(sum(self.lambdas[self.support_vectors] * self.y[self.support_vectors] * self.kernel(self.X[self.support_vectors], X[i])))
            if prediction > 0:
                prediction_res.append(1)
            else:
                prediction_res.append(-1)

        return np.array(prediction_res)

    def evaluate(self, X: np.ndarray, y: np.ndarray):
        return np.mean(self.predict(X) != y)

    def linear_kernel(self, x1, x2):
        return np.dot(x1, x2.T)

    def gaussian_kernel(self, x1: np.ndarray, x2: np.ndarray):
        return np.exp(-np.linalg.norm(x1-x2)**2 / self.gamma)
