import numpy as np

def accuracy_matrix(y_test: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
    """computes the accuracy given a matrix of predictions
    
    Args:

        y_test, columnwise label
        y_hat, matrix of predictions where each column is a model prediction
    """

    return (y_test == y_hat).sum(0) / y_test.shape[0]

def r2(y_test: np.ndarray, y_hat: np.ndarray) -> float:
    err = ((y_test - y_hat) ** 2).sum()
    bench = ((y_test - y_test.mean()) ** 2).sum()
    return 1 - (err / bench)


def mse_matrix(y_test: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
    """y_hat is a matrix of predictions"""
    samples = y_test.shape[0]
    return (((y_test - y_hat) ** 2) / samples).sum(0)


def r2_matrix(y_test: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
    """y_hat is a matrix of predictions"""
    return 1 - ((y_test - y_hat) ** 2).sum(0) / ((y_test - y_test.mean()) ** 2).sum(0)


def columnwise_mse_regression(y_test: np.ndarray, y_hat: np.ndarray) -> float:
    return (((y_test - y_hat) ** 2) / y_test.shape[0]).sum()


def columnwise_root_mse_regression(y_test: np.ndarray, y_hat: np.ndarray) -> float:
    return np.sqrt(columnwise_mse_regression(y_test, y_hat))
