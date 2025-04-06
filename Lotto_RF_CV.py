import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor

# Load your dataset (replace 'lottery_data.csv' with your actual CSV file)
data = pd.read_csv('LottoNumbers.csv')

# Replace column names with your actual dataset column names
# Example: 'number1', 'number2', ..., 'number6'
X = data[['number1', 'number2', 'number3', 'number4', 'number5', 'number6']]  # Features (past draws)
y = data[['number1', 'number2', 'number3', 'number4', 'number5', 'number6']]  # Targets (future draws)

# Initialize a more complex model: RandomForestRegressor
model = RandomForestRegressor(n_estimators=100)

# Split the data into training and test sets (use cross-validation for such a small dataset)
# Cross-validation for more robust training (using 5 folds)
cv_scores = cross_val_score(model, X, y, cv=5)

print(f"Cross-validation scores: {cv_scores}")
print(f"Mean cross-validation score: {np.mean(cv_scores)}")

# Train the model on the entire dataset
model.fit(X, y)

# Generate a random input for prediction (simulating the next draw's numbers)
next_draw = pd.DataFrame(np.random.randint(1, 50, size=(1, 6)), columns=['number1', 'number2', 'number3', 'number4', 'number5', 'number6'])
print(f"Next random draw input:\n{next_draw}")

# Predict the next draw numbers
predicted_numbers = model.predict(next_draw)

# Add some variability to make the predictions more realistic
predicted_numbers_with_variation = np.round(predicted_numbers).astype(int)

print(f"Predicted numbers for the next draw (with random variation):\n{predicted_numbers_with_variation}")
