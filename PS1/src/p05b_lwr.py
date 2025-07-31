import matplotlib.pyplot as plt
import numpy as np
import util

from linear_model import LinearModel


def main(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path)

    # *** START CODE HERE ***
    # Fit a LWR model
    clf = LocallyWeightedLinearRegression(tau=0.5)
    clf.fit(x_train, y_train)
    # Get MSE value on the validation set
    x_eval, y_eval = util.load_dataset(eval_path)
    predictions = clf.predict(x_eval)
    mse = ((predictions - y_eval)**2) / len(y_eval)
    # Plot validation predictions on top of training set
    plt.plot(np.arange(len(predictions)), predictions)
    plt.plot(np.arange(len(predictions)), y_eval)
    plt.savefig("output/lwr.png")
    # No need to save predictions
    # Plot data
    # *** END CODE HERE ***


class LocallyWeightedLinearRegression(LinearModel):
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        # *** START CODE HERE ***
        self.x = x
        self.y = y
        # *** END CODE HERE ***

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        predictions = []
        for xt in x:
            n = len(self.x)
            W = np.zeros((n, n))
            for i in range(n):
                W[i][i] = np.exp((-(xt - self.x[i])**2) / (2 * self.tau**2))

            theta = np.linalg.inv(self.x.T @ W @ self.x) @ (self.x.T @ W @ self.y)
            predictions.append(xt @ theta)
        return np.array(predictions).flatten()
        # *** END CODE HERE ***
