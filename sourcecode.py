import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# Loading and preprocessing the data
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    return df

# Creating sequences for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# Building the LSTM model
def build_model(seq_length, n_features):
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=(seq_length, n_features)),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(n_features)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Main function
def main():
    # Load data
    file_path = 'stock_data.csv'
    df = load_and_preprocess_data(file_path)

    # Scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    # Parameters
    seq_length = 10
    train_size = int(len(scaled_data) * 0.8)

    # Split data
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]

    # Create sequences
    X_train, y_train = create_sequences(train_data, seq_length)
    X_test, y_test = create_sequences(test_data, seq_length)

    # Build and train model
    model = build_model(seq_length, df.shape[1])
    history = model.fit(X_train, y_train, epochs=50, batch_size=32,
                       validation_split=0.1, verbose=1)

    # Make predictions
    predictions = model.predict(X_test)

    # Inverse transform predictions
    predictions = scaler.inverse_transform(predictions)
    y_test = scaler.inverse_transform(y_test)

    # Plot results for each stock
    plt.figure(figsize=(15, 10))
    for i, stock in enumerate(df.columns):
        plt.subplot(3, 2, i+1)
        plt.plot(y_test[:, i], label='Actual')
        plt.plot(predictions[:, i], label='Predicted')
        plt.title(f'{stock} Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
    plt.tight_layout()
    plt.savefig('stock_predictions.png')

if __name__ == '__main__':
    main()
