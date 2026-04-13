import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load iris dataset
iris = datasets.load_iris()
X = iris.data
# Let's predict petal length (feature 2) from other features, as a regression task
# Remove feature 2 from X to use others
X_reg = np.delete(X, 2, axis=1)
y_reg = X[:, 2]

# Split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Train Linear Regression model
reg = LinearRegression()
reg.fit(X_train, y_train)

# Predict and evaluate
y_pred = reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
