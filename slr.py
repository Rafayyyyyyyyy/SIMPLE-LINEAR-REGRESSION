import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Title of the app
st.title("Simple Linear Regression with Streamlit")

# Create a synthetic dataset (you can replace this with your CSV file)
# Generate some sample data for 'x' and 'y'
np.random.seed(42)
x = np.random.uniform(-10, 10, 100)  # 100 random values for 'x'
y = 3 * x + 7 + np.random.normal(0, 2, 100)  # y = 3x + 7 with some noise

# Convert to DataFrame
data = pd.DataFrame({'x': x, 'y': y})

# Show some data in the app
st.write("Sample Data", data.head())

# Train-test split (80% train, 20% test)
X = data[['x']]  # Independent variable
y = data['y']    # Dependent variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model to a file (this will be uploaded to GitHub)
joblib.dump(model, 'linear_regression_model.pkl')

# Predict on the test set (optional)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
st.write(f"Mean Squared Error on test data: {mse:.2f}")

# User input for prediction
x_input = st.number_input("Enter value of x:", min_value=-10.0, max_value=10.0, value=0.0)

# Make prediction when the button is clicked
if st.button("Predict"):
    prediction = model.predict(np.array([[x_input]]))
    st.write(f"Predicted value of y: {prediction[0]:.2f}")
