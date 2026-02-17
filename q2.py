import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score


def _compute_metrics(model, trainx, trainy, valx, valy, testx, testy):
    """Return a dict of RMSE and R2 for train, val, and test sets."""
    results = {}
    for prefix, x, y in [('train', trainx, trainy),
                          ('val', valx, valy),
                          ('test', testx, testy)]:
        pred = model.predict(x)
        results[f'{prefix}-rmse'] = np.sqrt(mean_squared_error(y, pred))
        results[f'{prefix}-r2'] = r2_score(y, pred)
    return results


def preprocess_data(trainx, valx, testx):
    """Z-score standardisation using training-set statistics only.

    Parameters
    ----------
    trainx, valx, testx : np.ndarray
        Feature matrices (date column must already be removed).

    Returns
    -------
    trainx_s, valx_s, testx_s : np.ndarray
        Standardised feature matrices.
    """
    mean = trainx.mean(axis=0)
    std = trainx.std(axis=0)
    std[std == 0] = 1.0  # avoid division by zero for constant features
    trainx_s = (trainx - mean) / std
    valx_s = (valx - mean) / std
    testx_s = (testx - mean) / std
    return trainx_s, valx_s, testx_s


def eval_linear1(trainx, trainy, valx, valy, testx, testy):
    """Train linear regression on training data only."""
    model = LinearRegression()
    model.fit(trainx, trainy)
    return _compute_metrics(model, trainx, trainy, valx, valy, testx, testy)


def eval_linear2(trainx, trainy, valx, valy, testx, testy):
    """Train linear regression on training + validation data."""
    combined_x = np.concatenate([trainx, valx], axis=0)
    combined_y = np.concatenate([trainy, valy], axis=0)
    model = LinearRegression()
    model.fit(combined_x, combined_y)
    return _compute_metrics(model, trainx, trainy, valx, valy, testx, testy)


def eval_ridge1(trainx, trainy, valx, valy, testx, testy, alpha):
    """Train Ridge regression on training data only."""
    model = Ridge(alpha=alpha)
    model.fit(trainx, trainy)
    return _compute_metrics(model, trainx, trainy, valx, valy, testx, testy)


def eval_ridge2(trainx, trainy, valx, valy, testx, testy, alpha):
    """Train Ridge regression on training + validation data."""
    combined_x = np.concatenate([trainx, valx], axis=0)
    combined_y = np.concatenate([trainy, valy], axis=0)
    model = Ridge(alpha=alpha)
    model.fit(combined_x, combined_y)
    return _compute_metrics(model, trainx, trainy, valx, valy, testx, testy)


def eval_lasso1(trainx, trainy, valx, valy, testx, testy, alpha):
    """Train Lasso regression on training data only."""
    model = Lasso(alpha=alpha, max_iter=10000)
    model.fit(trainx, trainy)
    return _compute_metrics(model, trainx, trainy, valx, valy, testx, testy)


def eval_lasso2(trainx, trainy, valx, valy, testx, testy, alpha):
    """Train Lasso regression on training + validation data."""
    combined_x = np.concatenate([trainx, valx], axis=0)
    combined_y = np.concatenate([trainy, valy], axis=0)
    model = Lasso(alpha=alpha, max_iter=10000)
    model.fit(combined_x, combined_y)
    return _compute_metrics(model, trainx, trainy, valx, valy, testx, testy)
