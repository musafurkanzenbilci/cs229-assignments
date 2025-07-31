import numpy as np
import util

from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    # *** START CODE HERE ***
    x_train_t, y_train_t = util.load_dataset(train_path, label_col='t')
    x_train_y, y_train_y = util.load_dataset(train_path, label_col='y')

    x_test_t, y_test_t = util.load_dataset(test_path, label_col='t')
    x_test_y, y_test_y = util.load_dataset(test_path, label_col='y')


    x_valid_t, y_valid_t = util.load_dataset(valid_path, label_col='t')
    x_valid_y, y_valid_y = util.load_dataset(valid_path, label_col='y')

    # Part (c): Train and test on true labels
    # Make sure to save outputs to pred_path_c
    
    # Use newton method with t labels
    clf_t = LogisticRegression()
    clf_t.fit(x=x_train_t, y=y_train_t, method="newton")
    clf_t.predict(x=x_test_t)

    # Part (d): Train on y-labels and test on true labels
    # Make sure to save outputs to pred_path_d
    
    # Use newton method with y labels
    clf_y = LogisticRegression()
    clf_y.fit(x=x_train_y, y=y_train_y, method="newton")
    predictions_t = clf_y.predict(x=x_test_t)
    
    # Part (e): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs to pred_path_e
    scores = clf_y.predict(x=x_valid_t)
    correction = scores.sum() / len(scores)
    scaled_prediction = predictions_t / correction
    print(scaled_prediction)
    # *** END CODER HERE

