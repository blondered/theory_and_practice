"""
Linear regression and logistic regression using Stochastic Gradient 
Descent with different variations.

The main goal of this project is to apply different variations of 
Gradient Descent to simple linear models and then test their convergence
on different datasets.
The following variations of GS are implemented: 
    - Classic
    - Momentum
    - Momentum Nesterov
    - AdaGrad
    - RMSProp
    - Adam
    - Adam Normed
    
"""

import numpy as np

"""Functions for model output, loss and gradient computation"""


def logit(X, w):
    return X @ w


def sigmoid(x):
    return (np.exp(-x) + 1) ** (-1)


def logistic_proba(X, w):
    logits = logit(X, w)
    return sigmoid(logits)


def mse(y_pred, y):
    y_pred = np.array(y_pred)
    y = np.array(y)
    return np.sum((y_pred - y) ** 2) / len(y)


def log_loss(y_proba, y):
    y = np.array(y).reshape((len(y), 1))
    y[y != 1] = 0  # Противоположная метка для 1 класса будет 0
    y_proba = np.clip(y_proba, a_min=1e-8, a_max=1 - 1e-8)
    loss = -(y * np.log(y_proba) + (1 - y) *
             np.log(1 - y_proba)).sum() / len(y)
    return loss


def get_grad_mse(X, y, w):
    return X.T @ (X @ w - y) / len(y)


def get_grad_log_loss(X, y, w):
    y = np.array(y)
    y[y != 1] = 0  # Противоположная метка для 1 класса будет 0
    a = logistic_proba(X, w)
    grad = (X.T @ (a - y)) / len(y)
    return grad


"""Batch generator"""


def batch_generator(X, y, batch_size=100):
    X = np.array(X)
    y = np.array(y)
    perm = np.random.permutation(len(y))
    identifier = lambda X, inds: np.array([X[i] for i in inds])
    inds = lambda batch_start: perm[batch_start * batch_size:
                                    (batch_start + 1) * batch_size]
    for batch_start in range(len(y) // batch_size):
        yield identifier(X, inds(batch_start)), identifier(y, inds(batch_start))


"""Different optimizers for gradient descent"""


class Classic:

    def __init__(self, learning_rate=1e-3, get_grad=get_grad_mse):
        self.lr = learning_rate
        self.get_grad = get_grad

    def step(self, w, X, y):
        return w - self.lr * self.get_grad(X, y, w)


class Momentum:

    def __init__(self, learning_rate=3e-4, alpha=0.9,
                 get_grad=get_grad_mse):
        self.lr = learning_rate
        self.alpha = alpha
        self.momentum = 0
        self.get_grad = get_grad

    def step(self, w, X, y):
        self.momentum = self.alpha * self.momentum + self.lr * self.get_grad(X, y, w)
        return w - self.momentum


class MomentumNest(Momentum):

    def __init__(self, learning_rate=3e-4, alpha=0.9,
                 get_grad=get_grad_mse):
        super().__init__()

    def step(self, w, X, y):
        self.momentum = self.alpha * self.momentum + self.lr * self.get_grad(X, y, w - self.alpha * self.momentum)
        return w - self.momentum


class AdaGrad:

    def __init__(self, learning_rate=3e-4, alpha=0.9, eps=1e-6,
                 get_grad=get_grad_mse):
        self.lr = learning_rate
        self.alpha = alpha
        self.G_coef = 0
        self.eps = eps
        self.get_grad = get_grad

    def step(self, w, X, y):
        grad = self.get_grad(X, y, w)
        self.G_coef += grad ** 2
        return w - (self.lr / np.sqrt(self.G_coef + self.eps)) * grad


class RMSProp:

    def __init__(self, learning_rate=3e-2, alpha=0.9, eps=1e-7,
                 get_grad=get_grad_mse):
        self.lr = learning_rate
        self.alpha = alpha
        self.eps = eps
        self.G_coef = 0
        self.get_grad = get_grad

    def step(self, w, X, y):
        grad = self.get_grad(X, y, w)
        self.G_coef = self.alpha * self.G_coef + (1 - self.alpha) * (grad ** 2)
        return w - (self.lr / (np.sqrt(self.G_coef + self.eps))) * grad


class Adam:

    def __init__(self, learning_rate=3e-2, alpha=0.9, beta=0.999,
                 eps=1e-3, get_grad=get_grad_mse):
        self.alpha = alpha
        self.beta = beta
        self.lr = learning_rate
        self.v_coef = 0
        self.m_coef = 0
        self.eps = eps
        self.get_grad = get_grad

    def step(self, w, X, y):
        grad = self.get_grad(X, y, w)
        self.v_coef = self.beta * self.v_coef + (1 - self.beta) * (grad ** 2)
        self.m_coef = self.alpha * self.m_coef + (1 - self.alpha) * grad
        return w - (self.lr / (np.sqrt(self.v_coef + self.eps))) * self.m_coef


class AdamNormed:

    def __init__(self, learning_rate=3e-2, alpha=0.9, beta=0.99,
                 eps=1e-3, get_grad=get_grad_mse):
        self.alpha = alpha
        self.beta = beta
        self.lr = learning_rate
        self.v_coef = 0
        self.m_coef = 0
        self.num_iter = 1
        self.eps = eps
        self.norms = []
        self.get_grad = get_grad

    def step(self, w, X, y):
        grad = self.get_grad(X, y, w)
        self.v_coef = (self.beta * self.v_coef + (1 - self.beta) * (grad ** 2))
        self.m_coef = (self.alpha * self.m_coef + (1 - self.alpha) * grad)
        self.norms.append(1 / (1 - (self.beta ** self.num_iter)))
        self.v_coef = self.v_coef / (1 - (self.beta ** self.num_iter))
        self.m_coef = self.m_coef / (1 - (self.alpha ** self.num_iter))
        self.num_iter += 1
        return w - (self.lr / (np.sqrt(self.v_coef + self.eps))) * self.m_coef


"""Linear regressor and logistic regressor with SGD solver"""


class LinearRegression:
    """Linear regression with SGD training and opportunity to choose
    different SGD variations
    """

    def fit(self, X, y, optimizer=Classic(), epochs=300, batch_size=100):
        """Fit model
        
        Parameters
        __________
        X : training data, matrix of shape [n_samples,n_features]
        y : target values, vector of shape [n_samples, ] or [n_samples, 1]
        optimizer : SG variation optimizer class instance
        epochs : number of epochs for training
        batch_size : size of the batch in mini-batch generator
        
        Returns
        _________
        array of losses during training iterations
        """
        losses = []
        X = np.array(X)
        y = np.array(y)
        l, d = X.shape
        d += 1
        X = np.hstack((np.ones((l, 1)), X))
        self.w = np.zeros((d, 1))
        for i in range(epochs):
            for X_batch, y_batch in batch_generator(X, y, batch_size=batch_size):
                predictions = logit(X_batch, self.w)
                loss = mse(predictions, y_batch)
                losses.append(loss)
                self.w = optimizer.step(self.w, X_batch, y_batch)
        return losses

    def predict(self, X):
        """Make predictions for new data with trained model
        
        Parameters
        __________
        X : data, matrix of shape [n_samples,n_features]
        
        Returns
        _________
        predictions, array of shape [n_samples]
        """
        X = np.array(X)
        l, d = X.shape
        X = np.hstack((np.ones((l, 1)), X))
        return logit(X, self.w)


class LogisticRegression:
    """Logistic regression with SGD training and opportunity to choose
    different SGD variations
    """

    def fit(self, X, y, optimizer=Classic(), epochs=300, batch_size=100):
        """Fit model
        
        Parameters
        __________
        X : training data, matrix of shape [n_samples,n_features]
        y : target values, vector of shape [n_samples, ] or [n_samples, 1]
        optimizer : SG variation optimizer class instance
        epochs : number of epochs for training
        batch_size : size of the batch in mini-batch generator
        
        Returns
        _________
        array of losses during training iterations
        """
        weight_dist = np.inf
        optimizer.get_grad = get_grad_log_loss
        losses = []
        X = np.array(X)
        y = np.array(y)
        if len(y.shape) == 1:
            y = y[:, np.newaxis]
        l, d = X.shape
        d += 1
        X = np.hstack((np.ones((l, 1)), X))
        self.w = np.zeros((d, 1))
        for i in range(epochs):
            for X_batch, y_batch in batch_generator(X, y, batch_size=batch_size):
                probas = logistic_proba(X_batch, self.w)
                loss = log_loss(probas, y_batch)
                losses.append(loss)
                self.w = optimizer.step(self.w, X_batch, y_batch)
                new_w = optimizer.step(self.w, X_batch, y_batch)
        return losses

    def predict(self, X):
        """Make predictions of classes for new data with trained model
        
        Parameters
        __________
        X : data, matrix of shape [n_samples,n_features]
        
        Returns
        _________
        class predictions, array of shape [n_samples]
        """
        probas = self.predict_proba(X)
        for i in range(len(probas)):
            if probas[i] >= 0.5:
                probas[i] = 1
            else:
                probas[i] = 0
        return probas

    def predict_proba(self, X):
        """Make predictions of positive class probability for new data with trained model
        
        Parameters
        __________
        X : data, matrix of shape [n_samples,n_features]
        
        Returns
        _________
        positive class probability predictions, array of shape [n_samples]
        """
        X = np.array(X)
        l, d = X.shape
        X = np.hstack((np.ones((l, 1)), X))
        probas = logistic_proba(X, self.w)
        return probas
