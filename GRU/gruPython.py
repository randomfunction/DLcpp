import numpy as np

class GRUCell:
    def __init__(self, input_dim, hidden_dim):
        #update gate (z)
        self.W_z = np.random.randn(input_dim, hidden_dim) * 0.1
        self.U_z = np.random.randn(hidden_dim, hidden_dim) * 0.1
        self.b_z = np.zeros(hidden_dim)
        #reset gate (r)
        self.W_r = np.random.randn(input_dim, hidden_dim) * 0.1
        self.U_r = np.random.randn(hidden_dim, hidden_dim) * 0.1
        self.b_r = np.zeros(hidden_dim)
        #candidate hidden state
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

# input_dim = 10  
# hidden_dim = 20
# gru_cell = GRUCell(input_dim, hidden_dim)
# x_t = np.random.randn(1, input_dim)
# h_prev = np.zeros((1, hidden_dim))
# h_t = gru_cell.forward(x_t, h_prev)
# print("Hidden state at time t:", h_t)

class GRU:
    def __init__(self, input_dim, hidden_dim):
        self.grucell = GRUCell(input_dim, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, X):
        batch_size, seq_len, input_dim = X.shape
        h_t = np.zeros((batch_size, self.hidden_dim))
        hidden_states = []

        for t in range(seq_len):
            x_t = X[:, t, :]
            h_t = self.grucell.forward(x_t, h_t)
            hidden_states.append(h_t)

        return np.stack(hidden_states, axis=1) 

# sequence_length = 5
# X = np.random.randn(1, sequence_length, input_dim) 
# gru = GRU(input_dim, hidden_dim)
# hidden_states = gru.forward(X)
# print("Hidden states for the sequence:", hidden_states)

#data preparation
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
ticker= 'AAPL'
data= yf.download(ticker, period='6mo', interval='1d')
prices= data['Close'].values.reshape(-1,1)
scaler= MinMaxScaler()
prices_scaled= scaler.fit_transform(prices)

def create_seq(data, seq_len):
    X,y=[],[]
    for i in range(len(data)-seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

seq_len=20
X,y= create_seq(prices_scaled, seq_len)
split= int(len(X)*0.8)
X_train, X_test= X[:split], X[split:]
y_train, y_test= y[:split], y[split:]

class GRUModel:
    def __init__(self, input_dim, hidden_dim, output_dim, lr):
        self.gru = GRU(input_dim, hidden_dim)
        self.W_out = np.random.randn(hidden_dim, output_dim) * 0.1
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


# Instantiate and train the model
input_dim = 1
hidden_dim = 20
output_dim = 1

model = GRUModel(input_dim, hidden_dim, output_dim, lr=0.001)
model.train(X_train, y_train, epochs=100)

y_pred_scaled = [model.forward(X_test[i:i+1])[0] for i in range(len(X_test))]
y_pred = scaler.inverse_transform(np.array(y_pred_scaled).reshape(-1, 1))
y_actual = scaler.inverse_transform(y_test)
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(y_actual, label="Actual Prices", color="blue")
plt.plot(y_pred, label="Predicted Prices", color="red")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Price")
plt.title("Stock Price Prediction")
plt.show()
