import numpy as np
import matplotlib.pyplot as plt
from math import floor, log10



class SimpleLinearRegression:
    def __init__(self, lr: float, tolerance: float, n_iterations: float, metric: str, method:str) -> None:
        '''
        '''
        # Initialization
        self.tolerance = tolerance
        self.max_iterations = n_iterations
        self.total_iterations = 0
        self.lr = lr
        self.metric = metric
        self.method = method
        # Parameters of Regression
        self.w = None
        self.b = None
        self.weights = list()
        self.costs = list()

    def __f(self, X: np.array, w: float, b: float) -> float:
        return w * X + b

    def __loss_function(self, X: np.array, y: np.array, w: float, b: float, metric: str) -> float:
        # Mean Squared Error (MSE)
        if metric.strip().lower() == "mse":
            return (1/(2 * X.shape[0])) * np.sum( (self.__f(X, w, b) - y)**2 )

    def __update_coefficients(self, X: np.array, y: np.array, w: float, b: float, lr: float, metric: str, method: str) -> float:
    # Gradient Descent (GD)
        if method.strip().lower() == "gd":
            def d_loss_function(X: np.array, y: np.array, w: float, b: float, metric: str) -> float:
                # Derivatives of Mean Squared Error
                if metric.strip().lower() == "mse":
                    dw = (1/X.shape[0]) * np.sum( (self.__f(X, w, b) - y) * X )
                    db = (1/X.shape[0]) * np.sum( self.__f(X, w, b) - y )
                    return dw, db
        
            dw, db = d_loss_function(X, y, w, b, metric)
            aux_w = w - lr * dw
            aux_b = b - lr * db
            return aux_w, aux_b

    def __gradient_descent(self, X: np.array, y: np.array):
        # Generating initial values for w and b
        self.w, self.b = 0, 0
        # Initial cost
        cost = self.__loss_function(X, y, self.w, self.b, self.metric)
        self.costs.append(cost)
        while (cost > self.tolerance) and (self.total_iterations < self.max_iterations):
            self.w, self.b = self.__update_coefficients(X, y, self.w, self.b, self.lr, self.metric, method="gd")    # Update the coefficients of f
            cost = self.__loss_function(X, y, self.w, self.b, self.metric)                           # The total cost related to new coefficients
            # Saving the history
            self.weights.append((self.w, self.b))
            self.costs.append(cost)
            self.total_iterations += 1

    def fit(self, X: np.array, y: np.array) -> None:
        if self.method.strip().lower() == "gd":
            return self.__gradient_descent(X, y)

    def predict(self, X: np.array) -> np.array:
        return self.__f(X=X, w=self.w, b=self.b)

    def evaluate(self, X: np.array, y: np.array) -> float:
        return self.__loss_function(X, y, self.w, self.b, self.metric)

    @property
    def info(self) -> dict:
        infos = {"tolerance": self.tolerance, "user_limit_iterations": self.max_iterations, "model_real_iterations_executed": self.total_iterations,
                "learning_rate": self.lr, "metric": self.metric, "method": self.method, "weights": (self.w, self.b),
                "weights_history": self.weights, "costs_history": self.costs}
        return infos
    
    @property
    def learning_curve(self) -> None:
        x_len = len(self.costs)
        step = int(floor(log10(abs(self.max_iterations))))
        plt.plot(np.arange(x_len), self.costs, label="Loss")
        plt.title(label="Learning Curve")
        plt.ylabel(ylabel=f"Loss - {self.info["metric"]}")
        plt.xlabel(xlabel="Iterations")
        plt.xticks(ticks=np.arange(0, x_len+1, step), rotation=45)
        plt.legend()