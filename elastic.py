import numpy as np


def loss(x, y, beta, el, alpha):
    """Compute the Elastic Net objective value.

    f(beta) = 0.5 * ||y - X @ beta||_2^2
              + el * (alpha * ||beta||_2^2 + (1 - alpha) * ||beta||_1)

    Parameters
    ----------
    x : np.ndarray, shape (n, d)
    y : np.ndarray, shape (n,)
    beta : np.ndarray, shape (d,)
    el : float — regularisation strength (lambda).
    alpha : float — mixing parameter in [0, 1].

    Returns
    -------
    float — objective value.
    """
    residual = y - x @ beta
    data_loss = 0.5 * np.dot(residual, residual)
    l2_penalty = el * alpha * np.dot(beta, beta)
    l1_penalty = el * (1 - alpha) * np.sum(np.abs(beta))
    return data_loss + l2_penalty + l1_penalty


def grad_step(x, y, beta, el, alpha, eta):
    """Perform one proximal gradient descent step.

    1. Compute gradient of smooth part g(beta):
       grad = -X^T (y - X beta) + 2 * el * alpha * beta
    2. Gradient descent on smooth part:
       beta_tmp = beta - eta * grad
    3. Proximal operator (soft-thresholding) for L1 penalty:
       threshold = el * (1 - alpha) * eta

    Parameters
    ----------
    x : np.ndarray, shape (m, d) — data batch.
    y : np.ndarray, shape (m,) — target batch.
    beta : np.ndarray, shape (d,)
    el : float — regularisation strength (lambda).
    alpha : float — mixing parameter.
    eta : float — learning rate.

    Returns
    -------
    np.ndarray, shape (d,) — updated beta.
    """
    residual = y - x @ beta
    grad_smooth = -x.T @ residual + 2.0 * el * alpha * beta

    beta_tmp = beta - eta * grad_smooth

    threshold = el * (1.0 - alpha) * eta
    beta_new = np.sign(beta_tmp) * np.maximum(np.abs(beta_tmp) - threshold, 0.0)
    return beta_new


class ElasticNet:
    def __init__(self, el, alpha, eta, batch, epoch):
        self.el = el
        self.alpha = alpha
        self.eta = eta
        self.batch = batch
        self.epoch = epoch
        self.beta = None

    def coef(self):
        """Return the learned coefficients as a numpy array."""
        return self.beta

    def train(self, x, y):
        """Train the Elastic Net model via mini-batch SGD with proximal steps.

        Parameters
        ----------
        x : np.ndarray, shape (n, d)
        y : np.ndarray, shape (n,)

        Returns
        -------
        dict — {epoch_number: loss_value} computed on the full training set
               at the end of each epoch.
        """
        n, d = x.shape
        self.beta = np.zeros(d)
        loss_history = {}

        for ep in range(1, self.epoch + 1):
            indices = np.random.permutation(n)
            x_shuffled = x[indices]
            y_shuffled = y[indices]

            for start in range(0, n, self.batch):
                end = min(start + self.batch, n)
                x_batch = x_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                self.beta = grad_step(
                    x_batch, y_batch, self.beta,
                    self.el, self.alpha, self.eta,
                )

            loss_history[ep] = loss(x, y, self.beta, self.el, self.alpha)

        return loss_history

    def predict(self, x):
        """Predict target values.

        Parameters
        ----------
        x : np.ndarray, shape (m, d)

        Returns
        -------
        np.ndarray, shape (m,)
        """
        return x @ self.beta
