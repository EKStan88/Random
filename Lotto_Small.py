import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load your dataset (replace 'lottery_data.csv' with your actual CSV file)
data = pd.read_csv('LottoNumbers.csv')

# Replace column names with your actual dataset column names
# Example: 'number1', 'number2', 'number3', 'number4', 'number5', 'number6'
X = data[['number1', 'number2', 'number3', 'number4', 'number5', 'number6']]  # Features (past draws)
y = data[['number1', 'number2', 'number3', 'number4', 'number5', 'number6']]  # Targets (future draws)

# Initialize the model
model = LinearRegression()

# Check if there are enough rows to fit the model
if len(X) < 10:
    print("Warning: Your dataset has less than 10 rows, which might not be sufficient for training a reliable model.")

# Train the model on the full dataset (since you only have 9 rows)
model.fit(X, y)

# Generate a random input for prediction (simulating the next draw's numbers)
# Create a DataFrame for the next draw to match expected input format
next_draw = pd.DataFrame(np.random.randint(1, 50, size=(1, 6)), columns=['number1', 'number2', 'number3', 'number4', 'number5', 'number6'])
print(f"Next random draw input:\n{next_draw}")

# Predict the next draw numbers
predicted_numbers = model.predict(next_draw)
print(f"Predicted numbers for the next draw:\n{np.round(predicted_numbers).astype(int)}")
