"""Regression utils."""
import copy
import numpy as np



def dct_coefficients(n_volumes, cutoff_hz, tr):
    """Create basis functions for Discrete Cosine Transform as in SPM."""
    K = int(np.floor(2 * n_volumes * cutoff_hz * tr))
    D = np.ones((n_volumes, K+1))/np.sqrt(n_volumes)
    for k in np.arange(1,K+1):
        D[:, k] = (
            np.sqrt(2.0 / n_volumes)
            * np.cos((np.pi / n_volumes)
                     * (np.arange(0, n_volumes) + 1/2) * k)
        )
    return D


def _check_input_2d_array(X):
    """Check that X is np.ndarray and that is has two dimensions."""
    if not isinstance(X, np.ndarray):
        raise TypeError(f'{X} should be an array.')
    if not X.ndim == 2:
        raise TypeError(f'{X} should have two dimensions')


def _replace_zeros_with_nans(X):
    """Replace zeros with nans in an array X."""
    if not X.dtype.kind == 'f':
        raise TypeError(f'{X} should be floats.')
    Y = copy.deepcopy(X)
    Y[Y == 0] = np.nan
    return Y


def ols_deflation(X, D):
    """Ordinary least squares deflation."""
    _check_input_2d_array(X)
    _check_input_2d_array(D)
    if not X.shape[0] == D.shape[0]:
        raise TypeError(f'{X.shape} and {D.shape} have non-matching rows')

    # Return the least-squares solution to the linear matrix equation.
    betas = np.linalg.lstsq(D, X, rcond=None)[0]

    # Project out
    Y = copy.deepcopy(X)
    Y -= D @ betas

    return Y


def cross_validation_splits(num_volumes, num_splits):
    """Split data into disjoint splits."""
    num_retained = (num_volumes//num_splits)*num_splits
    # It is common practice to discard the first few volumes to allow
    # homogenization of the magnetic field. This function thus
    # discards the first couple of volumes if the total number of volumes cannot
    # be divided into <num_splits> equal sized chunks.
    indexer = np.arange(num_retained)+(num_volumes-num_retained-1)
    indexer = indexer.reshape(num_splits, num_volumes//num_splits)

    print(f'The first {num_volumes-num_retained} volumes are discarded.')
    return indexer


def ridge_regression_estimator(X_train, Y_train, X_val, alpha):
    """Fit Ridge regression model for a given alpha."""
    _check_input_2d_array(X_train)
    _check_input_2d_array(Y_train)
    _check_input_2d_array(X_val)

    if not X_train.shape[0] == Y_train.shape[0]:
        raise TypeError(
            f'{X_train.shape} and {Y_train.shape} have non-matching rows')
    if not isinstance(alpha, float):
        raise TypeError(f'{alpha} should be a float')

    # Compute empirical mean and standard deviation of training sets
    mu_X = X_train.mean(axis=0, keepdims=True)
    sd_X = X_train.std(axis=0, keepdims=True)
    mu_Y = Y_train.mean(axis=0, keepdims=True)
    sd_Y = Y_train.std(axis=0, keepdims=True)

    if np.isclose(sd_X, 0).any():
        raise ValueError('There are columns in X_train with zero std.')

    sd_Y = _replace_zeros_with_nans(sd_Y)

    # Create a deep copy of training set
    X = copy.deepcopy(X_train)
    Y = copy.deepcopy(Y_train)

    # Standardize
    X = (X - mu_X) / sd_X
    Y = (Y - mu_Y) / sd_Y

    # Estimate weights
    W = np.linalg.inv(X.T@X + alpha * np.eye(X.shape[1]))@X.T@Y

    # Return out-of-sample prediction
    return (((X_val - mu_X)/sd_X) @ W) * sd_Y + mu_Y


def compute_coefficient_of_determination(Y, P):
    """Compute coefficient of determination, i.e., R2."""
    _check_input_2d_array(Y)
    _check_input_2d_array(P)

    nominator = np.sum((Y - Y.mean(axis=0, keepdims=True))**2, axis=0)
    denominator = np.sum((Y - P)**2, axis=0)

    return 1 - np.divide(denominator, _replace_zeros_with_nans(nominator))


def cross_validate_ridge_model(X, Y, alphas, num_splits):
    """Cross-validate Ridge regression models."""
    _check_input_2d_array(X)
    _check_input_2d_array(Y)

    if not X.shape[0] == Y.shape[0]:
        raise TypeError(
            f'{X.shape} and {Y.shape} have non-matching rows')

    # Specify how many volumes that are considered
    num_volumes = X.shape[0]
    num_voxels = Y.shape[1]
    index = cross_validation_splits(num_volumes, num_splits)

    # The ouput will be stored in R2
    R2 = np.zeros((num_voxels, len(alphas)))

    # Iterate over all alphas
    for a, alpha in enumerate(alphas):

        if alpha == 0:
            print(f'Fitting OLS regression model with alpha={alpha}')
        else:
            print(f'Fitting Ridge regression model with alpha={alpha}')

        # Store predictions P for the given alpha
        P = np.zeros(Y.shape)

        # Iterate over splits
        for j in range(num_splits):

            print('.', end='')

            idx_folds = np.setdiff1d(np.arange(num_splits), j)

            # The indices are now as follows
            train_volumes = index[idx_folds, :].ravel()
            validation_volumes = index[j].ravel()

            assert not np.isin(validation_volumes, train_volumes).any(
            ), "The chunks must be non-overlapping"

            # Define training data and test data
            X_train = X[train_volumes, :]
            Y_train = Y[train_volumes, :]

            # Further define validation data
            X_val = X[validation_volumes, :]

            P[validation_volumes, :] = (
                ridge_regression_estimator(X_train, Y_train, X_val, alpha)
            )

        print('')
        # Estimate R2
        R2[..., a] = compute_coefficient_of_determination(Y, P)

    # For convienience and compute time, return only the highest R2 values

    return np.max(R2, axis=-1)
