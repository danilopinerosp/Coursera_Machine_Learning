def feature_scaling(x):
    """
    Returns a vector with the same size of x but get every value into
    approximately a [-1,1] range.

    It divides every value by the max value of the vector

    Parameters:
    ----------
    x -> numpy.array

    Returns:
    -------
    scaled -> numpy.array

    Examples:
    --------
    >> x = np.array([1, 4, 6, 7, 8])
    >> feature_scaling(x)
    [0.125 0.5   0.75  0.875 1.   ]

    """
    import numpy as np

    scaled = x / np.max(x)
    return scaled

def mean_normalization(x):
    """
    Returns an array with the same size of x but with every element of x
    normalized:
    x = (x - np.mean(x)) / np.std(x)

    Parameters:
    ----------
    x -> np.array

    Returns:
    -------
    normalized -> np.array

    Examples:
    --------
    >> x = np.array([1, 4, 6, 7, 8])
    >> mean_normalization(x)
    [-1.69222822 -0.48349378  0.32232919  0.72524067  1.12815215]
    """
    import numpy as np

    normalized = (x - np.mean(x)) / np.std(x)

    return normalized

if __name__ == "__main__":
    import numpy as np

    a = np.array([1, 4, 6, 7, 8])
    print(mean_normalization(a))
