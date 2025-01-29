import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

data = pd.read_csv('student_scores.csv')
print(data.columns)
gdp = np.array(data['GDP Per Capita'])  # Assuming the column name is 'Celsius'
life = np.array(data['Life Expectancy'])  # Assuming the column name is 'Fahrenheit'
plt.scatter(gdp,life)
model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer=tf.keras.optimizers.Adam(0.1), loss='mean_squared_error')
print("Training the model...")
history = model.fit(gdp, life, epochs=1, verbose=0)
print("Model training complete.")
weight, bias = model.layers[0].get_weights()
print(f"Weight: {weight[0][0]}")
print(f"Bias: {bias[0]}")
plt.plot(history.history['loss'])
plt.title('Model Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

gdp_input = float(input("Enter a GDP to predict life expectancy: "))
gdp_input_array = np.array([gdp_input], dtype=float)  # Ensure the input is a NumPy array
predicted_lifeEx= model.predict(gdp_input_array)[0][0]

print(f"Predicted Fahrenheit temperature: {predicted_lifeEx:.2f}")

# Plot actual vs predicted values
predicted_values = model.predict(gdp).flatten()  # Predict for training data
plt.scatter(gdp, life, color='blue', label='Actual')
plt.scatter(gdp, predicted_values, color='red', label='Predicted')
plt.plot(gdp, predicted_values, color='red', linestyle='dashed')
plt.title('Actual vs Predicted Values')
plt.xlabel('GDP')
plt.ylabel('Life expectancy')
plt.legend()
plt.show()