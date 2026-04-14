# 13-04-2026

## What we'll do:
- setup environment & install libraries
- Create the Jupyter Notebook & load the dataset
- Exploratory Data Analysis
- Data Preprocessing
- Train/Test Split & Train the model
- Evaluate the model (MAE, RMSE, R²)
- Visualize result, save the model, final report

## Setup
- install : pip install numpy pandas matplotlib seaborn scikit-learn jupyter openpyxl joblib
- Uninstall : pip uninstall numpy pandas matplotlib seaborn scikit-learn jupyter openpyxl joblib -y
- Create : housing_project.ipynb in project folder

## Jupyter Notebook
- Click + Code to add first cell:


**Cell 1 - Imports**  
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
import seaborn as sns  
from sklearn.datasets import   fetch_california_housing  
print("Libraries loaded successfully!")  

---

# Cell 2 - Load Dataset
housing = fetch_california_housing()

# Convert to a pandas DataFrame
df = pd.DataFrame(housing.data, columns=housing.feature_names)

# Add the target column (house prices)
df['MedHouseVal'] = housing.target

print("Dataset loaded!")
print("Shape:", df.shape)
print("\nFirst 5 rows:")
df.head()

---

# Cell 3 - Basic Info
print("Shape:", df.shape)
print("\nColumn Names:", df.columns.tolist())
print("\nData Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isnull().sum())

---

# Cell 4 - Statistical Summary
df.describe().round(2)

---

# Cell 5 - Target Distribution
plt.figure(figsize=(8, 4))
sns.histplot(df['MedHouseVal'], bins=50, kde=True, color='steelblue')
plt.title('Distribution of Median House Values')
plt.xlabel('Median House Value ($100,000s)')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

---

# Cell 6 - Correlation Heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.show()

---

# Cell 7 - Top Correlations
correlations = df.corr()['MedHouseVal'].sort_values(ascending=False)
print("Correlations with House Value:\n")
print(correlations.round(3))

---

# Cell 8 - Boxplots for Outliers
plt.figure(figsize=(15, 8))
df.boxplot(figsize=(15, 8))
plt.xticks(rotation=45)
plt.title('Boxplots to Detect Outliers')
plt.tight_layout()
plt.show()

---

# Cell 9 - Features and Target
X = df.drop('MedHouseVal', axis=1)  # All columns except target
y = df['MedHouseVal']               # Target column only

print("Features shape:", X.shape)
print("Target shape:", y.shape)
print("\nFeatures used:\n", X.columns.tolist())

---

# Cell 10 - Feature Scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert back to DataFrame for readability
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

print("Before Scaling - MedInc mean:", round(X['MedInc'].mean(), 3))
print("After Scaling  - MedInc mean:", round(X_scaled['MedInc'].mean(), 3))
print("\nScaling done! All features now have mean≈0 and std≈1")

---

# 14-04-2026

## WHY?
| Setup              | WHY                                                                   |
|--------------------|-----------------------------------------------------------------------|
| Separate x and y   | Model needs features and target separate                              |
| StandardScaler     | Makes all features on the same scale so no single feature dominates   |

---

# Output / Result

- Boxplot: Population has massive outliers (up to 35,000!) — some areas are very densely populated. This is normal for real-world data.

- Cell 9: 20,640 rows and 8 features — this is also okay

- Cell 10: MedInc mean went from 3.871 → 0.0 — scaling worked perfectly! ✅

---

# Cell 11 - Train/Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print("Training set size:", X_train.shape)
print("Testing set size: ", X_test.shape)
print(f"\n80% data for training → {X_train.shape[0]} rows")
print(f"20% data for testing  → {X_test.shape[0]} rows")

---

# Cell 12 - Train the Model
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

print("Model trained successfully!")
print("\nModel Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"  {feature:12s} → {coef:.4f}")
print(f"\nIntercept: {model.intercept_:.4f}")

---

# Cell 13 - Predictions
y_pred = model.predict(X_test)

print("Predictions done!")
print("\nSample Predictions vs Actual:")
print(f"{'Actual':>10} | {'Predicted':>10}")
print("-" * 25)
for actual, predicted in zip(y_test[:8], y_pred[:8]):
    print(f"{actual:>10.3f} | {predicted:>10.3f}")

---

# WHY
| Setup              | WHY                                                                   |
|--------------------|-----------------------------------------------------------------------|
| train_test_split   | 80% data trains the model, 20% is kept aside to test it               |
| model.fit()        | Model learns the relationship between features and house prices       |
| model.predict()    | Model guesses prices for houses it has never model_selection          |


---

# Cell 14 - Evaluation Metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("=" * 40)
print("       MODEL EVALUATION RESULTS")
print("=" * 40)
print(f"  MAE  (Mean Absolute Error) : {mae:.4f}")
print(f"  RMSE (Root Mean Sq. Error) : {rmse:.4f}")
print(f"  R²   (R-Squared Score)     : {r2:.4f}")
print("=" * 40)
print(f"\n  Prices are in $100,000 units")
print(f"  MAE means avg error of ${mae*100000:.0f}")

---

# Cell 15 - Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.3, color='steelblue', edgecolors='k', linewidth=0.3)
plt.plot([y_test.min(), y_test.max()], 
         [y_test.min(), y_test.max()], 
         'r--', linewidth=2, label='Perfect Prediction')
plt.xlabel('Actual House Value')
plt.ylabel('Predicted House Value')
plt.title('Actual vs Predicted House Values')
plt.legend()
plt.tight_layout()
plt.show()

---

# Cell 16 - Residuals
residuals = y_test - y_pred

plt.figure(figsize=(8, 5))
sns.histplot(residuals, bins=50, kde=True, color='salmon')
plt.axvline(x=0, color='black', linestyle='--', linewidth=1.5)
plt.title('Residuals Distribution')
plt.xlabel('Residual (Actual - Predicted)')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

---

# Cell 17 - Save the Model and Scaler
import joblib

# Save the trained model
joblib.dump(model, 'california_housing_model.pkl')

# Save the scaler too (important for future predictions!)
joblib.dump(scaler, 'california_housing_scaler.pkl')

print("Model saved as: california_housing_model.pkl")
print("Scaler saved as: california_housing_scaler.pkl")

---

# Cell 18 - Reload and Test Saved Model
loaded_model = joblib.load('california_housing_model.pkl')
loaded_scaler = joblib.load('california_housing_scaler.pkl')

# Test with one sample house
sample = X.iloc[0:1]
sample_scaled = loaded_scaler.transform(sample)
prediction = loaded_model.predict(sample_scaled)

print("Model reloaded successfully!")
print(f"\nSample Prediction : ${prediction[0]*100000:.0f}")
print(f"Actual Value      : ${y.iloc[0]*100000:.0f}")


---

- create markdown and redner this 

"""

- at end run all