import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)

    # *** START CODE HERE ***
    clf = LogisticRegression()
    clf.fit(x_train, y_train, method='newton')
    results = clf.predict(x_eval)
    print(f"The mean difference between predictions and ground truth is {(results - y_eval).mean()}")
    # *** END CODE HERE ***


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def fit(self, x, y, method='gradient'):
        """Fit model using either Gradient Ascent or Newton's Method.

        Args:
            x: Training inputs (m, n).
            y: Training labels (m,).
            method: Optimization method ('gradient' or 'newton').
        """
        # *** START CODE HERE ***
        m, n = x.shape
        self.theta = np.zeros(n) if self.theta is None else self.theta

        if method == 'gradient':
            for _ in range(self.max_iter):
                h = sigmoid(x @ self.theta)
                gradient = x.T @ (y - h) / m
                self.theta += self.step_size * gradient

        elif method == 'newton':
            for i in range(self.max_iter):
                h = sigmoid(x @ self.theta)
                gradient = x.T @ (y - h) / m
                S = np.diag(h * (1 - h))  # (m, m)
                H = (x.T @ S @ x) / m     # (n, n)

                try:
                    delta = np.linalg.solve(H, gradient)
                except np.linalg.LinAlgError:
                    print("Warning: Hessian not invertible.")
                    break

                self.theta += delta

                # convergence check: stop if delta is small
                if np.linalg.norm(delta, ord=1) < self.eps:
                    if self.verbose:
                        print(f"Converged at iteration {i}")
                    break
        else:
            raise ValueError("Method must be 'gradient' or 'newton'.")
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return (sigmoid(x @ self.theta) > 0.5).astype(int)
        # *** END CODE HERE ***
