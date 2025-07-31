import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=False)

    # *** START CODE HERE ***
    clf = GDA()
    clf.fit(x_train, y_train)
    clf.predict(x_eval)
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        m, n = x.shape
        phi = np.mean(y)

        mu0 = x[y == 0].mean(axis=0)
        mu1 = x[y == 1].mean(axis=0)
        self.mu = [mu0, mu1]

        # Shared covariance
        diffs = x - np.where(y[:, np.newaxis] == 1, mu1, mu0)
        sigma = (diffs.T @ diffs) / m
        self.sigma = sigma

        # Inverse
        sigma_inv = np.linalg.inv(sigma)

        # Compute theta_1 and theta_0
        theta_1 = sigma_inv @ (mu1 - mu0)
        theta_0 = (
            0.5 * (mu0 @ sigma_inv @ mu0 - mu1 @ sigma_inv @ mu1)
            - np.log((1 - phi) / phi)
        )

        # Final theta with intercept
        self.theta = np.zeros(n + 1)
        self.theta[0] = theta_0
        self.theta[1:] = theta_1
        # *** END CODE HERE ***

    def predict(self, x, method='theta'):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).
            method: theta | naive

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        def px_y(x, sigma, mu):
            """
            x: shape (m, n)
            mu: shape (n,)
            sigma: shape (n, n)
            returns: shape (m,)
            """
            m, n = x.shape
            inv_sigma = np.linalg.inv(sigma)
            det_sigma = np.linalg.det(sigma)
            norm_const = 1.0 / ((2 * np.pi) ** (n / 2) * np.sqrt(det_sigma))

            # Center x
            centered = x - mu  # shape (m, n)
            
            # Compute Mahalanobis distance for each row: (x - mu)^T @ Sigma^-1 @ (x - mu)
            # Using einsum for efficient row-wise dot product
            exponent = -0.5 * np.einsum("ij,jk,ik->i", centered, inv_sigma, centered)

            return norm_const * np.exp(exponent)

        if method == "theta":
            x = util.add_intercept(x)
            return 1 / (1 + np.exp(-x @ self.theta))
        elif method == "naive":
            px_y1 = px_y(x, self.sigma, self.mu[1])
            px_y0 = px_y(x, self.sigma, self.mu[0])
            p_y1 = 0.5  # (y_train == 1).sum() / m
            p_y0 = 0.5  # (y_train == 0).sum() / m
            prediction = (px_y1 * p_y1) / (px_y1*p_y1 + px_y0*p_y0)
            return prediction
        # *** END CODE HERE
