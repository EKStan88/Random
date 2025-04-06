import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load historical lottery data
# Assuming you have data in CSV format with past lottery results
# Example CSV format: columns 'draw_date', 'number1', 'number2', ..., 'number6'
data = pd.read_csv('LottoNumbers.csv')

# Preprocess data
# Extracting just the number columns for training
numbers = data[['number1', 'number2', 'number3', 'number4', 'number5', 'number6']]

# Normalize the numbers to a range between 0 and 1
scaler = MinMaxScaler()
numbers_scaled = scaler.fit_transform(numbers)

# Create sequences of previous results to predict the next draw
sequence_length = 20  # Use the past 10 draws to predict the next one
X = []
y = []

for i in range(len(numbers_scaled) - sequence_length):
    X.append(numbers_scaled[i:i+sequence_length])
    y.append(numbers_scaled[i+sequence_length])

X = np.array(X)
y = np.array(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)

# Build the neural network model
model = Sequential()
model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(6))  # 6 numbers in the lottery

model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# Predict the next lottery numbers
predicted_numbers = model.predict(X_test)

# Inverse scale the predicted numbers back to original range
predicted_numbers = scaler.inverse_transform(predicted_numbers)

# Round and convert to integer (lotto numbers are usually integers)
predicted_numbers = np.round(predicted_numbers).astype(int)

# Display the predicted numbers
print("Predicted next lotto numbers:", predicted_numbers[0])
