import linear_regression as lrw
import numpy as np
import matplotlib.pyplot as plt

# Load the model
model = lrw.LinearRegression()

plt.rcParams['figure.figsize'] = (12.0, 9.0)

# Generate sample data
rng = np.random.RandomState(0)
X_train = np.sort(5 * rng.rand(2000, 1), axis=0)
Y = 0.5 * X_train + 0.3
noise = rng.normal(0,1,Y.shape)
Y += noise

model.fit(X_train, Y, learning_rate=0.0001, epochs=1000)

m = model.getSlope
b = model.getIntercept

Y_pred = m * X_train + b

plt.scatter(X_train, Y)
plt.plot([min(X_train), max(X_train)], [min(Y_pred), max(Y_pred)], color='red') # predicted
plt.show()
print(f"Slope: {model.getSlope:.16f}, Intercept: {model.getIntercept:.16f}")

# Example prediction
# print(f"Prediction at x = 2: {model.predict(2):.2f}")