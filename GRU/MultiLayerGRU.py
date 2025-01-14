import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import matplotlib.pyplot as plt

class GRUCell:
    def __init__(self, input_dim, hidden_dim):
        self.W_z = np.random.randn(input_dim, hidden_dim) * 0.1
        self.U_z = np.random.randn(hidden_dim, hidden_dim) * 0.1
        self.b_z = np.zeros(hidden_dim)

        self.W_r = np.random.randn(input_dim, hidden_dim) * 0.1
        self.U_r = np.random.randn(hidden_dim, hidden_dim) * 0.1
        self.b_r = np.zeros(hidden_dim)

        self.W_h = np.random.randn(input_dim, hidden_dim) * 0.1
        self.U_h = np.random.randn(hidden_dim, hidden_dim) * 0.1
        self.b_h = np.zeros(hidden_dim)

        self.hidden_dim = hidden_dim

    def forward(self, x_t, h_prev):
        z_t = self.sigmoid(np.dot(x_t, self.W_z) + np.dot(h_prev, self.U_z) + self.b_z)
        r_t = self.sigmoid(np.dot(x_t, self.W_r) + np.dot(h_prev, self.U_r) + self.b_r)
        h_hat_t = np.tanh(np.dot(x_t, self.W_h) + r_t * (np.dot(h_prev, self.U_h)) + self.b_h)
        h_t = (1 - z_t) * h_prev + z_t * h_hat_t
        return h_t

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))


class MultiLayerGRU:
    def __init__(self, input_dim, hidden_dims):
        self.layers = []
        self.num_layers = len(hidden_dims)

        for i, hidden_dim in enumerate(hidden_dims):
            input_size = input_dim if i == 0 else hidden_dims[i - 1]
            self.layers.append(GRUCell(input_size, hidden_dim))
        self.hidden_dims = hidden_dims

    def forward(self, X):
        batch_size, seq_len, _ = X.shape
        h_t = [np.zeros((batch_size, hidden_dim)) for hidden_dim in self.hidden_dims]
        hidden_states = []

        for t in range(seq_len):
            x_t = X[:, t, :]
            for i, layer in enumerate(self.layers):
                h_t[i] = layer.forward(x_t, h_t[i])
                x_t = h_t[i]  
            hidden_states.append(h_t[-1]) 

        return np.stack(hidden_states, axis=1) 


# Multi-Layer GRU Model
class MultiLayerGRUModel:
    def __init__(self, input_dim, hidden_dims, output_dim, lr):
        self.gru = MultiLayerGRU(input_dim, hidden_dims)
        self.W_out = np.random.randn(hidden_dims[-1], output_dim) * 0.1
        self.b_out = np.zeros(output_dim)
        self.lr = lr

    def forward(self, X):
        hidden_states = self.gru.forward(X)
        last_hidden_state = hidden_states[:, -1, :] 
        return np.dot(last_hidden_state, self.W_out) + self.b_out, last_hidden_state

    def computeLoss(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)

    def train(self, X_train, y_train, epochs):
        for epoch in range(epochs):
            total_loss = 0
            for i in range(len(X_train)):
                X = X_train[i:i+1] 
                y_true = y_train[i:i+1]
                y_pred, last_hidden_state = self.forward(X)
                loss = self.computeLoss(y_pred, y_true)
                total_loss += loss

                d_loss = 2 * (y_pred - y_true) / y_true.size
                d_W_out = np.dot(last_hidden_state.T, d_loss)
                d_b_out = np.sum(d_loss, axis=0)

                self.W_out -= self.lr * d_W_out
                self.b_out -= self.lr * d_b_out

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(X_train):.4f}")


class AdamOptimizer:
    def __init__(self, params, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {key: np.zeros_like(value) for key, value in params.items()}
        self.v = {key: np.zeros_like(value) for key, value in params.items()}
        self.t = 0

    def update(self, grads):
        self.t += 1
        for key in self.params:
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            self.params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)


class MultiLayerGRUModelWithAdam(MultiLayerGRUModel):
    def __init__(self, input_dim, hidden_dims, output_dim, lr):
        super().__init__(input_dim, hidden_dims, output_dim, lr)
        self.optimizer = AdamOptimizer(
            params={"W_out": self.W_out, "b_out": self.b_out}, lr=lr
        )

    def train(self, X_train, y_train, epochs):
        for epoch in range(epochs):
            total_loss = 0
            for i in range(len(X_train)):
                X = X_train[i:i+1]
                y_true = y_train[i:i+1]
                y_pred, last_hidden_state = self.forward(X)
                loss = self.computeLoss(y_pred, y_true)
                total_loss += loss

                d_loss = 2 * (y_pred - y_true) / y_true.size
                grads = {
                    "W_out": np.dot(last_hidden_state.T, d_loss),
                    "b_out": np.sum(d_loss, axis=0),
                }

                self.optimizer.update(grads)

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(X_train):.4f}")

ticker = 'AAPL'
data = yf.download(ticker, period='5y', interval='1d')
prices = data['Close'].values.reshape(-1, 1)
scaler = MinMaxScaler()
prices_scaled = scaler.fit_transform(prices)

def create_seq(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len])
    return np.array(X), np.array(y)

seq_len = 50
X, y = create_seq(prices_scaled, seq_len)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

input_dim = 1
hidden_dims = [128, 128, 128]
output_dim = 1
learning_rate = 0.01

multi_layer_model = MultiLayerGRUModelWithAdam(input_dim, hidden_dims, output_dim, lr=learning_rate)
multi_layer_model.train(X_train, y_train, epochs=1)

# Predictions and Visualization
y_pred_scaled = [multi_layer_model.forward(X_test[i:i+1])[0] for i in range(len(X_test))]
y_pred = scaler.inverse_transform(np.array(y_pred_scaled).reshape(-1, 1))
y_actual = scaler.inverse_transform(y_test)

plt.figure(figsize=(12, 6))
plt.plot(y_actual, label="Actual Prices", color="blue")
plt.plot(y_pred, label="Predicted Prices", color="red")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Price")
plt.title("Stock Price Prediction (Multi-Layer GRU)")
plt.show()
