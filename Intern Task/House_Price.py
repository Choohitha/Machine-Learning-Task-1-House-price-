import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# 1. Load dataset
df = pd.read_csv(r"C:\Users\Choohitha\Downloads\House_Price_dataset (2).csv")
print("First 5 rows of the dataset:")
print(df.head())
df.columns = ['area', 'bedrooms', 'bathrooms', 'price']
print(df.head())
X = df[['area', 'bedrooms', 'bathrooms']]
y = df['price']
print("\nDataset Info:")
print(df.info())

print("\nDataset Description:")
print(df.describe())

# Pairplot to visualize numeric feature relationships
sns.pairplot(df.select_dtypes(include=['number']))
plt.suptitle("Feature Relationships - Pairplot", y=1.02)
plt.show()
# Correlation heatmap for numeric columns
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
# 3. Data Preparation
X = df[['area', 'bedrooms', 'bathrooms']]  # numeric features
y = df['price']  # target variable

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Model Training
model = LinearRegression()
model.fit(X_train, y_train)
# 5. Predictions & Evaluation
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

plt.scatter(y_test, y_pred, color='blue')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.grid(True)
plt.show()

residuals = y_test - y_pred
sns.histplot(residuals, kde=True)
plt.title("Residuals Distribution")
plt.show()

