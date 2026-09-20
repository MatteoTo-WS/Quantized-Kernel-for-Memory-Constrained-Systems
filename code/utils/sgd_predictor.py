import numpy as np

class SGDPredictor():
    def __init__(self, x_dim, lr=0.1, decay_rate=0.0, bias=False, clip_value=None, averaging=False, l2_penalty=0.0):
        self.lr = lr
        self.decay_rate = decay_rate
        self.bias_bool = bias
        self.clip_value = clip_value
        self.averaging = averaging
        self.l2_penalty = l2_penalty
        self._reset(x_dim)

    def _reset(self, x_dim):
        self.w = np.zeros(x_dim)
        self.b = 0.0
        self._w_sum = np.zeros(x_dim)
        self._b_sum = 0.0
        self._t_avg = 0

    def learning_rate(self, t):
        return self.lr / (1.0 + self.decay_rate * t)

    def partial_fit(self, x_batch, y_batch, t=0, D=None):
        n_samples = x_batch.shape[0]
        
        y_pred = x_batch @ self.w
        if self.bias_bool:
            y_pred = y_pred + self.b

        y_batch = y_batch.reshape(-1)
        residual = y_pred - y_batch
        
        grad_w = (2.0 / n_samples) * (x_batch.T @ residual)
        
        # 1. Regolarizzazione L2 (Tikhonov) per garantire stabilità
        if self.l2_penalty > 0:
            grad_w += 2.0 * self.l2_penalty * self.w
        
        # 2. Correzione De-biased
        if D is not None:
            penalty = D @ self.w if isinstance(D, np.ndarray) and D.ndim == 2 else D * self.w
            grad_w -= 2.0 * penalty

        if self.clip_value is not None:
            grad_w = np.clip(grad_w, -self.clip_value, self.clip_value)

        lr_t = self.learning_rate(t)
        self.w -= lr_t * grad_w

        if self.bias_bool:
            grad_b = (2.0 / n_samples) * np.sum(residual)
            if self.clip_value is not None:
                grad_b = np.clip(grad_b, -self.clip_value, self.clip_value)
            self.b -= lr_t * grad_b

        if self.averaging:
            self._w_sum += self.w
            self._b_sum += self.b
            self._t_avg += 1

    def predict(self, X):
        w = self._w_avg if (self.averaging and self._t_avg > 0) else self.w
        b = (self._b_sum / self._t_avg) if (self.averaging and self._t_avg > 0 and self.bias_bool) else self.b
        y_pred = X @ w
        if self.bias_bool:
            y_pred = y_pred + b
        return y_pred.flatten()

    @property
    def _w_avg(self):
        if self._t_avg == 0:
            return self.w
        return self._w_sum / self._t_avg

    def error(self, X, y):
        y_pred = self.predict(X)
        return np.mean((y_pred - y.flatten()) ** 2)