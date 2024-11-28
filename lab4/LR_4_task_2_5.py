import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

np.random.seed(0)
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.4 * X ** 2 + X + 4 + np.random.randn(m, 1)

# Лінійна регресія
linear_regressor = linear_model.LinearRegression()
linear_regressor.fit(X, y)
y_linearpred = linear_regressor.predict(X)

# Поліноміальна регресія
polynomial = PolynomialFeatures(degree=2)
X_poly = polynomial.fit_transform(X)
poly_regressor = linear_model.LinearRegression()
poly_regressor.fit(X_poly, y)
y_polypred = poly_regressor.predict(X_poly) # Use transformed X_poly

# Візуалізація результатів
plt.scatter(X, y, color='green')
plt.plot(X, y_linearpred, color='blue', label='linear', linewidth=4)
sort_indices = np.argsort(X[:, 0])
X_sorted = X[sort_indices]
y_polypred_sorted = y_polypred[sort_indices]
plt.plot(X_sorted, y_polypred_sorted, color='red', label='polynomial', linewidth=4)
plt.legend()
plt.xlabel("X")
plt.ylabel("y")
plt.show()

# Оцінка продуктивності для обох моделей
print("Linear regressor performance:")
print("Mean absolute error =", round(sm.mean_absolute_error(y, y_linearpred), 2))
print("Mean squared error =", round(sm.mean_squared_error(y, y_linearpred), 2))
print("Median absolute error =", round(sm.median_absolute_error(y, y_linearpred), 2))
print("Explain variance score =", round(sm.explained_variance_score(y, y_linearpred), 2))
print("R2 score =", round(sm.r2_score(y, y_linearpred), 2))
print("\nPolynomial regressor performance:")
print("Mean absolute error =", round(sm.mean_absolute_error(y, y_polypred), 2))
print("Mean squared error =", round(sm.mean_squared_error(y, y_polypred), 2))
print("Median absolute error =", round(sm.median_absolute_error(y, y_polypred), 2))
print("Explain variance score =", round(sm.explained_variance_score(y, y_polypred), 2))
print("R2 score =", round(sm.r2_score(y, y_polypred), 2))
print('\nPolynomial model coefficients:', poly_regressor.coef_)
print('Polynomial model intercept:', poly_regressor.intercept_)
