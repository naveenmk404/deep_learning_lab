import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Download stock market data
ticker = 'AAPL'
data = yf.download(ticker, start="2010-01-01", end="2022-12-31")[['Close']]

# Plot the closing prices
plt.figure(figsize=(10, 6))
plt.plot(data)
plt.title('Apple Inc. Closing Prices')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.show()

# Split and normalize data
train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train)
test_scaled = scaler.transform(test)

# Create dataset function
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

# Prepare datasets
time_step = 60
X_train, y_train = create_dataset(train_scaled, time_step)
X_test, y_test = create_dataset(test_scaled, time_step)
X_train = X_train.reshape(X_train.shape[0], time_step, 1)
X_test = X_test.reshape(X_test.shape[0], time_step, 1)

# Build and train LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
    LSTM(50),
    Dense(25),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))

# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Make and inverse transform predictions
train_predict = scaler.inverse_transform(model.predict(X_train))
test_predict = scaler.inverse_transform(model.predict(X_test))

# Prepare data for plotting
def plot_predictions(data, train_predict, test_predict, time_step):
    train_plot = np.empty_like(data)
    test_plot = np.empty_like(data)
    train_plot[:, :] = np.nan
    test_plot[:, :] = np.nan
    train_plot[time_step:len(train_predict) + time_step, :] = train_predict
    test_plot[len(train_predict) + (time_step * 2) + 1:len(data) - 1, :] = test_predict
    
    plt.figure(figsize=(10, 6))
    plt.plot(data, label='Original Data')
    plt.plot(train_plot, label='Train Predict')
    plt.plot(test_plot, label='Test Predict')
    plt.title('Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

plot_predictions(data, train_predict, test_predict, time_step)

