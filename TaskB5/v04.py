# File: stock_prediction.py
# Authors: Bao Vo and Cheong Koo
# Date: 14/07/2021(v1); 19/07/2021 (v2); 02/07/2024 (v3)

# Code modified from:
# Title: Predicting Stock Prices with Python
# YouTube link: https://www.youtube.com/watch?v=PuZY9q-aKLw
# By: NeuralNine


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import mplfinance as mpf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

# Load Data
COMPANY = 'CBA.AX'
TRAIN_START = '2020-01-01'
TRAIN_END = '2023-08-01'
data = yf.download(COMPANY, TRAIN_START, TRAIN_END)

# Prepare Data
PRICE_VALUE = "Close"
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[PRICE_VALUE].values.reshape(-1, 1))
PREDICTION_DAYS = 60

# Prepare the training data
x_train = []
y_train = []
scaled_data = scaled_data[:, 0]

for x in range(PREDICTION_DAYS, len(scaled_data)):
    x_train.append(scaled_data[x - PREDICTION_DAYS:x])
    y_train.append(scaled_data[x])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build the Model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the Model with Validation Split
history = model.fit(x_train, y_train, epochs=25, batch_size=32, validation_split=0.1)

# Test the model accuracy on existing data
TEST_START = '2023-08-02'
TEST_END = '2024-07-02'
test_data = yf.download(COMPANY, TEST_START, TEST_END)
actual_prices = test_data[PRICE_VALUE].values
total_dataset = pd.concat((data[PRICE_VALUE], test_data[PRICE_VALUE]), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - PREDICTION_DAYS:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)

# Prepare test data
x_test = []
for x in range(PREDICTION_DAYS, len(model_inputs)):
    x_test.append(model_inputs[x - PREDICTION_DAYS:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Make predictions
predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Multistep prediction function
def multistep_prediction(model, input_data, n_steps):
    predictions = []
    current_input = input_data[-PREDICTION_DAYS:]

    for _ in range(n_steps):
        current_input = np.reshape(current_input, (1, PREDICTION_DAYS, 1))
        next_pred = model.predict(current_input)
        predictions.append(next_pred[0, 0])
        current_input = np.append(current_input[0, 1:], next_pred[0, 0])

    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# Multivariate prediction function
def predict_multivariate_stock_price(model, data, features, scaler, prediction_days):
    """
    Forecast the stock price for a future day using multiple features (multivariate prediction).
    """
    recent_data = data[-prediction_days:][features].values
    scaled_data = scaler.transform(recent_data)
    scaled_data = scaled_data.reshape(1, prediction_days, len(features))

    predicted_value = model.predict(scaled_data)

    result = np.zeros((1, len(features)))
    result[0, 0] = predicted_value[0, 0]  # Store the predicted closing price

    return scaler.inverse_transform(result)[0, 0]

# Plotting functions
def plot_multivariate_predictions(actual_prices, multivariate_preds):
    plt.figure(figsize=(14, 7))
    plt.plot(actual_prices, label='Actual Prices', color='blue')

    # Create an index for the predicted values
    pred_index = np.arange(len(actual_prices), len(actual_prices) + len(multivariate_preds))
    plt.plot(pred_index, multivariate_preds, label='Multivariate Predictions', color='orange')

    plt.title('Multivariate Predictions vs Actual Prices')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

# Example usage for multivariate prediction
n_steps = 5  # Number of days to predict
multivariate_features = ['Close']  # You can add more features if available

# Prepare data for multivariate prediction
multivariate_data = data.copy()  # Add more features to this DataFrame if available
multivariate_preds = []

for i in range(n_steps):
    next_pred = predict_multivariate_stock_price(model, multivariate_data, multivariate_features, scaler, PREDICTION_DAYS)
    multivariate_preds.append(next_pred)

# Plot multivariate predictions
extended_actual_prices = np.concatenate((actual_prices, [None] * n_steps))
plot_multivariate_predictions(extended_actual_prices, multivariate_preds)

# Plot candlestick and boxplot
def plot_candlestick(data, n_days=1):
    resampled_data = data.resample(f'{n_days}D').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last'
    })
    resampled_data.dropna()
    mpf.plot(resampled_data, type='candle', style='charles', title='Candlestick Chart')

def plot_boxplot(data, n_days=1):
    if n_days < 1:
        raise ValueError("n_days must be >= 1")

    resampled_data = data.resample(f'{n_days}D').agg({'Close': 'median'})
    resampled_data.dropna(inplace=True)
    resampled_data['Period'] = resampled_data.index.to_period('W').astype(str)

    plt.figure(figsize=(12, 6))
    plt.boxplot(
        [resampled_data['Close'][resampled_data['Period'] == period] for period in resampled_data['Period'].unique()],
        labels=resampled_data['Period'].unique())
    plt.title('Boxplot Chart')
    plt.xlabel('Period')
    plt.ylabel('Closing Price')
    plt.xticks(rotation=45)
    plt.show()

plot_candlestick(data, n_days=5)
plot_boxplot(data, n_days=5)

# Predict next day
real_data = [model_inputs[len(model_inputs) - PREDICTION_DAYS:, 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(f"Prediction for the next day: {prediction}")

n_steps = 5
multistep_preds = multistep_prediction(model, model_inputs, n_steps)
print(f"Multistep predictions for the next {n_steps} days: {multistep_preds}")

# A few concluding remarks here:
# 1. The predictor is quite bad, especially if you look at the next day
# prediction, it missed the actual price by about 10%-13%
# Can you find the reason?
# 2. The code base at
# https://github.com/x4nth055/pythoncode-tutorials/tree/master/machine-learning/stock-prediction
# gives a much better prediction. Even though on the surface, it didn't seem
# to be a big difference (both use Stacked LSTM)
# Again, can you explain it?
# 3. Explore CNN techniques and combine with LSTM for better predictions.
