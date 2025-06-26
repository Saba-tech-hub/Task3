import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression

data = {
    'Area': [1000, 1500, 2000, 2500, 3000],
    'Bedrooms': [2, 3, 4, 4, 5],
    'Price': [300000, 400000, 500000, 600000, 700000]
}
df = pd.DataFrame(data)

X = df[['Area', 'Bedrooms']]
y = df['Price']

model = LinearRegression()
model.fit(X, y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(df['Area'], df['Bedrooms'], y, color='blue', label='Actual Data')

area_range = np.linspace(df['Area'].min(), df['Area'].max(), 10)
bedrooms_range = np.linspace(df['Bedrooms'].min(), df['Bedrooms'].max(), 10)
area_grid, bedrooms_grid = np.meshgrid(area_range, bedrooms_range)
price_pred = model.predict(np.c_[area_grid.ravel(), bedrooms_grid.ravel()])
price_grid = price_pred.reshape(area_grid.shape)

ax.plot_surface(area_grid, bedrooms_grid, price_grid, color='red', alpha=0.5)

ax.set_xlabel('Area (sqft)')
ax.set_ylabel('Bedrooms')
ax.set_zlabel('Price')
ax.set_title('Multiple Linear Regression (3D Plot)')
plt.legend()
plt.show()
