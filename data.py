import numpy as np
import os


def load_data(train = True):
    """
    Load the data from disk

    Parameters
    ----------
    train : bool
        Load training data if true, else load test data
    Returns
    -------
        Tuple:
            Images
            Labels
    """
    # given by the startercode
    directory = 'train' if train else 'test'
    patterns = np.load(os.path.join('./data/', directory, 'images.npz'))['arr_0']
    labels = np.load(os.path.join('./data/', directory, 'labels.npz'))['arr_0']
    return patterns.reshape(len(patterns), -1), labels


def z_score_normalize(X, u = None, xd = None):
    """
    Performs z-score normalization on X. 

    f(x) = (x - μ) / σ
        where 
            μ = mean of x
            σ = standard deviation of x

    Parameters
    ----------
    X : np.array
        The data to z-score normalize
    u (optional) : np.array
        The mean to use when normalizing
    sd (optional) : np.array
        The standard deviation to use when normalizing

    Returns
    -------
        Tuple:
            Transformed dataset with mean 0 and stdev 1
            Computed statistics (mean and stdev) for the dataset to undo z-scoring.
    """
    # calculate the mean and sd by columns
    avg = np.mean(X, axis = 0)
    sd = np.std(X, axis = 0)
    return (X - avg) / sd, avg, sd 


def min_max_normalize(X, _min = None, _max = None):
    """
    Performs min-max normalization on X. 

    f(x) = (x - min(x)) / (max(x) - min(x))

    Parameters
    ----------
    X : np.array
        The data to min-max normalize
    _min (optional) : np.array
        The min to use when normalizing
    _max (optional) : np.array
        The max to use when normalizing

    Returns
    -------
        Tuple:
            Transformed dataset with all values in [0,1]
            Computed statistics (min and max) for the dataset to undo min-max normalization.
    """
    # calculate the min and max by columns
    minimum = np.min(X, axis = 0)
    maximum = np.max(X, axis = 0)
    return (X - minimum) / (maximum - minimum), minimum, maximum


def onehot_encode(y):
    """
    Performs one-hot encoding on y.

    Ideas:
        NumPy's `eye` function

    Parameters
    ----------
    y : np.array
        1d array (length n) of targets (k)

    Returns
    -------
        2d array (shape n*k) with each row corresponding to 
        a one-hot encoded version of the original value.
    """
    # calculate the number of columns needed for the encoded matrix
    dem = np.max(y) + 1
    # return onehot encoded matrix
    return np.eye(dem)[y]
    

def onehot_decode(y):
    """
    Performs one-hot decoding on y.

    Ideas:
        NumPy's `argmax` function 

    Parameters
    ----------
    y : np.array
        2d array (shape n*k) with each row corresponding to 
        a one-hot encoded version of the original value.

    Returns
    -------
        1d array (length n) of targets (k)
    """
    # return the onehot decoded vector by using given `argmax` hint
    return np.argmax(y, axis=1)


def shuffle(dataset):
    """
    Shuffle dataset.

    Make sure that corresponding images and labels are kept together. 
    Ideas: 
        NumPy array indexing 
            https://numpy.org/doc/stable/user/basics.indexing.html#advanced-indexing

    Parameters
    ----------
    dataset
        Tuple containing
            Images (X)
            Labels (y)

    Returns
    -------
        Tuple containing
            Images (X)
            Labels (y)
    """
    # find the number of images in the data and save their indexes as an array
    idx = np.arange(len(dataset[0]))
    # shuffle the above array in-place
    np.random.shuffle(idx)
    # return the shuffled dataset
    return dataset[0][idx], dataset[1][idx]
    

def append_bias(X):
    """
    Append bias term for dataset.

    Parameters
    ----------
    X
        2d numpy array with shape (N,d)

    Returns
    -------
        2d numpy array with shape ((N+1),d)
    """
    # append a column of bias before the first column of the design matrix
    return np.insert(X,0,1,axis=1)


def generate_minibatches(dataset, batch_size=64):
    """
    Helper method to generate minibatches

    Parameters
    ----------
    dataset
        Tuple containing
            Images (X)
            Labels (y)
    batch_size
        int with default of 64
    Returns
    -------
        smaller Tuple:
            Images (X)
            Labels (y)
    """
    # given by the startercode without modification
    X, y = dataset
    l_idx, r_idx = 0, batch_size
    while r_idx < len(X):
        yield X[l_idx:r_idx], y[l_idx:r_idx]
        l_idx, r_idx = r_idx, r_idx + batch_size
    yield X[l_idx:], y[l_idx:]

    
def generate_k_fold_set(dataset, k = 10): 
    """
    Helper method to generate k fold set

    Parameters
    ----------
    dataset
        Tuple containing
            Images (X)
            Labels (y)
    k
        int with default of 10
    Returns
    -------
        two size 2 Tuples:
            train and validation datasets
    """
    # given by the startercode without modification
    X, y = dataset
    order = np.random.permutation(len(X))
    fold_width = len(X) // k
    l_idx, r_idx = 0, fold_width
    for i in range(k):
        train = np.concatenate([X[order[:l_idx]], X[order[r_idx:]]]), \
                np.concatenate([y[order[:l_idx]], y[order[r_idx:]]])
        validation = X[order[l_idx:r_idx]], y[order[l_idx:r_idx]]
        yield train, validation
        l_idx, r_idx = r_idx, r_idx + fold_width 
