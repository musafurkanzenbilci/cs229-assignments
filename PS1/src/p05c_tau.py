import matplotlib.pyplot as plt
import numpy as np
import util

from p05b_lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path)
    x_test, y_test = util.load_dataset(test_path)

    # *** START CODE HERE ***
    # Search tau_values for the best tau (lowest MSE on the validation set)
    tau_values = np.arange(0.5, 25, 5)
    # Fit a LWR model with the best tau value
    for tau in tau_values:
        clf = LocallyWeightedLinearRegression(tau)
        clf.fit(x_train, y_train)
    # Run on the test set to get the MSE value
        predictions = clf.predict(x_test)
    # Save predictions to pred_path
        np.savetxt(pred_path, predictions)
    # Plot data
        plt.figure(tau)
        plt.plot(np.arange(len(predictions)), y_test - predictions)
        # plt.plot(np.arange(len(predictions)), y_test)
        plt.savefig(f"output/lwr_{tau}.png")
        plt.close()
    # *** END CODE HERE ***
