from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([3, 5, 7, 9, 11])  # y = 2x + 1

# Train a model
model = LinearRegression()
model.fit(X, y)

# Predict
X_test = np.array([[6], [7]])
y_pred = model.predict(X_test)

# Print model details
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Predictions for", X_test.flatten(), ":", y_pred)

# Plot
plt.scatter(X, y, color="blue", label="Training Data")
plt.plot(X, model.predict(X), color="green", label="Model Fit")
plt.scatter(X_test, y_pred, color="red", label="Predictions")
plt.legend()
plt.xlabel("X")
plt.ylabel("y")
plt.title("Simple Linear Regression with scikit-learn")
plt.grid(True)
plt.show()
