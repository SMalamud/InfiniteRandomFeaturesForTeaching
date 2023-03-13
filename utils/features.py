import numpy as np


import numpy as np


def create_features_grid(
    sample_size: int, max_multiplier: int, min_dividend: int, step: int
) -> np.ndarray:
    """returns a n_features_grid with:

    1. max_size = sample_size * max_multiplier
    2. min_size = sample_size / min_dividend
    3. the features grid granularity is defined by
    """
    features = np.arange(
        int(sample_size / min_dividend), int(sample_size * max_multiplier), step=step
    )

    features = np.concatenate([features, np.array([sample_size])])
    features.sort()
    return features
