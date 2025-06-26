import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#LINEAR REGRESSION
df = pd.read_csv("C:\\Users\\Mohamed shapan\\Desktop\\SABA🙃\\Elevate Lab\\Task3\\pokemon_data.csv")  

X = df[['height']]
y = df['base_experience']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Evaluation:")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

print("\nModel Coefficients:")
print(f"Intercept: {model.intercept_:.2f}")
print(f"Height Coefficient: {model.coef_[0]:.2f}")


plt.scatter(X_test['height'], y_test, color='blue', label='Actual')
plt.plot(X_test['height'], model.predict(X_test), color='red', label='Predicted')
plt.xlabel('Height')
plt.ylabel('Base Experience')
plt.title('Simple Linear Regression (Height vs Base Experience)')
plt.legend()
plt.grid(True)
plt.show()


