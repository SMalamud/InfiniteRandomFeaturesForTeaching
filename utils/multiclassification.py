import numpy as np


def get_one_against_all_matrix(labels: np.ndarray) -> np.ndarray:
    """creates the one against all matrix"""
    unique_labels = np.unique(labels)

    one_against_all_labels = np.concatenate(
        [(labels == label).astype(int) for label in unique_labels], axis=1
    )
    return one_against_all_labels
