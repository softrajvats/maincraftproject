# California Housing Price Prediction
### ML Internship Project Report

---

**Submitted by:** [Your Name]
**Internship Program:** AI/ML Internship
**Date:** April 2026
**Tools Used:** Python 3.14 · Pandas · Scikit-learn · Matplotlib · Seaborn · Jupyter Notebook

---

## Table of Contents

1. [Objective](#1-objective)
2. [Dataset Overview](#2-dataset-overview)
3. [Project Workflow](#3-project-workflow)
4. [Exploratory Data Analysis](#4-exploratory-data-analysis)
5. [Data Preprocessing](#5-data-preprocessing)
6. [Model Training](#6-model-training)
7. [Model Evaluation](#7-model-evaluation)
8. [Results & Visualizations](#8-results--visualizations)
9. [Model Saving & Deployment](#9-model-saving--deployment)
10. [Key Findings](#10-key-findings)
11. [Limitations & Future Improvements](#11-limitations--future-improvements)
12. [Project Structure](#12-project-structure)
13. [How to Run](#13-how-to-run)

---

## 1. Objective

The goal of this project is to introduce the end-to-end Machine Learning workflow by training a **Linear Regression** model on the **California Housing dataset**. The project covers every stage of a real ML pipeline — from data loading and exploration to model training, evaluation, and saving.

**Primary Goals:**
- Load and explore a real-world housing dataset
- Perform Exploratory Data Analysis (EDA) and visualization
- Preprocess data using feature scaling
- Train a Linear Regression model with an 80/20 train-test split
- Evaluate model performance using MAE, RMSE, and R² metrics
- Save the trained model for future use
- Document findings in a Jupyter Notebook and presentation

---

## 2. Dataset Overview

| Property       | Details                                      |
|----------------|----------------------------------------------|
| **Source**     | Scikit-learn built-in (`fetch_california_housing`) |
| **Origin**     | 1990 California Census data                  |
| **Samples**    | 20,640 rows                                  |
| **Features**   | 8 numeric input features                     |
| **Target**     | Median House Value (in $100,000 units)       |
| **Missing Values** | None                                     |

### Feature Descriptions

| Feature       | Description                                        |
|---------------|----------------------------------------------------|
| `MedInc`      | Median income of households in the block           |
| `HouseAge`    | Median age of houses in the block                  |
| `AveRooms`    | Average number of rooms per household              |
| `AveBedrms`   | Average number of bedrooms per household           |
| `Population`  | Total population of the block                      |
| `AveOccup`    | Average number of household members                |
| `Latitude`    | Geographic latitude of the block                   |
| `Longitude`   | Geographic longitude of the block                  |
| `MedHouseVal` | **Target** — Median house value ($100,000 units)   |

---

## 3. Project Workflow

```
Raw Dataset
    │
    ▼
Data Loading & Exploration
    │
    ▼
Exploratory Data Analysis (EDA)
    │  - Distribution plots
    │  - Correlation heatmap
    │  - Feature correlation ranking
    ▼
Data Preprocessing
    │  - Feature/target separation
    │  - StandardScaler normalization
    ▼
Train / Test Split (80% / 20%)
    │
    ▼
Model Training (LinearRegression)
    │
    ▼
Predictions on Test Set
    │
    ▼
Evaluation (MAE · RMSE · R²)
    │
    ▼
Visualization (Actual vs Predicted · Residuals)
    │
    ▼
Save Model (.pkl)
```

---

## 4. Exploratory Data Analysis

### 4.1 Basic Statistics

The dataset was inspected for shape, data types, and missing values.

```python
print("Shape:", df.shape)         # (20640, 9)
print("Missing Values:", df.isnull().sum())  # All zeros — no missing data
```

**Key statistical observations:**
- `MedHouseVal` ranges from `0.15` to `5.00` ($15,000 – $500,000)
- `Population` has extreme outliers — values reaching up to 35,682
- `AveRooms` and `AveBedrms` also contain high-end outliers
- `MedInc` has a mean of ~3.87 and ranges from 0.5 to 15

### 4.2 Target Distribution

The target variable `MedHouseVal` shows a right-skewed distribution with a notable spike at the maximum value of 5.0, suggesting the dataset caps house values at $500,000.

### 4.3 Feature Correlations with Target

| Feature       | Correlation | Direction |
|---------------|-------------|-----------|
| `MedInc`      | **+0.688**  | Strong positive ↑ |
| `AveRooms`    | +0.151      | Weak positive ↑ |
| `HouseAge`    | +0.106      | Weak positive ↑ |
| `Longitude`   | -0.046      | Weak negative ↓ |
| `AveBedrms`   | -0.047      | Weak negative ↓ |
| `Population`  | -0.025      | Negligible ↓ |
| `AveOccup`    | -0.023      | Negligible ↓ |
| `Latitude`    | **-0.144**  | Weak negative ↓ |

> **Insight:** `MedInc` is by far the strongest predictor of housing prices — richer neighborhoods have higher home values.

### 4.4 Visualizations Produced

- **Histogram** — Distribution of Median House Values
- **Heatmap** — Full feature correlation matrix
- **Bar chart** — Feature correlations with target variable
- **Boxplots** — Outlier detection across all features

---

## 5. Data Preprocessing

### 5.1 Feature & Target Separation

```python
X = df.drop('MedHouseVal', axis=1)   # Shape: (20640, 8)
y = df['MedHouseVal']                 # Shape: (20640,)
```

### 5.2 Feature Scaling

`StandardScaler` was applied to normalize all features so no single feature dominates due to scale differences.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

**Effect of scaling:**

| Feature   | Before (Mean) | After (Mean) |
|-----------|---------------|--------------|
| `MedInc`  | 3.871         | ~0.000       |
| All others | Various      | ~0.000       |

After scaling, every feature has **mean ≈ 0** and **standard deviation ≈ 1**.

---

## 6. Model Training

### 6.1 Train/Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
```

| Split         | Samples |
|---------------|---------|
| Training set  | 16,512  |
| Testing set   | 4,128   |

### 6.2 Model — Linear Regression

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
```

### 6.3 Learned Coefficients

| Feature       | Coefficient | Interpretation                          |
|---------------|-------------|------------------------------------------|
| `MedInc`      | **+0.8524** | Strongest positive effect on price       |
| `Latitude`    | **-0.8966** | Northward location lowers price          |
| `Longitude`   | **-0.8689** | Eastward location lowers price           |
| `AveBedrms`   | +0.3711     | More bedrooms → slightly higher price   |
| `AveRooms`    | -0.3051     | More rooms → slightly lower (correlated)|
| `HouseAge`    | +0.1224     | Older homes → marginally higher price   |
| `AveOccup`    | -0.0366     | More occupants → slightly lower price  |
| `Population`  | -0.0023     | Negligible effect                        |

**Intercept:** 2.0679

---

## 7. Model Evaluation

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)
```

### Results

| Metric | Value  | Interpretation                              |
|--------|--------|----------------------------------------------|
| **MAE**  | 0.5332 | Average prediction error of **~$53,320**  |
| **RMSE** | 0.7456 | Higher penalty for large errors            |
| **R²**   | 0.5758 | Model explains **57.6%** of price variance |

### Interpretation

- The model performs reasonably well as a **baseline** for this dataset
- An R² of ~0.58 is typical for Linear Regression on the California Housing dataset
- The MAE of ~$53K means the model's predictions are off by about $53,320 on average
- RMSE being higher than MAE indicates some larger prediction errors exist, particularly for high-value homes

---

## 8. Results & Visualizations

### 8.1 Sample Predictions vs Actual Values

| Actual | Predicted | Difference |
|--------|-----------|------------|
| 0.477  | 0.719     | +0.242     |
| 0.458  | 1.764     | +1.306     |
| 5.000  | 2.710     | -2.290     |
| 2.186  | 2.839     | +0.653     |
| 2.780  | 2.605     | -0.175     |
| 1.587  | 2.012     | +0.425     |
| 1.982  | 2.646     | +0.664     |
| 1.575  | 2.169     | +0.594     |

> **Note:** The model struggles with high-value homes (e.g., actual = 5.000, predicted = 2.710). This is a known limitation of linear models on capped data.

### 8.2 Visualizations

**Actual vs Predicted Plot**
A scatter plot comparing actual vs predicted values. Points close to the red diagonal line (`y = x`) represent accurate predictions. The spread increases for higher-priced homes.

**Residuals Distribution**
A histogram of residuals (`actual - predicted`) centered near 0, indicating no major systematic bias. The slight right skew is expected due to the price cap at 5.0.

---

## 9. Model Saving & Deployment

Both the trained model and the scaler were saved using `joblib` for future reuse:

```python
import joblib

joblib.dump(model,  'california_housing_model.pkl')
joblib.dump(scaler, 'california_housing_scaler.pkl')
```

### Loading and Using the Saved Model

```python
loaded_model  = joblib.load('california_housing_model.pkl')
loaded_scaler = joblib.load('california_housing_scaler.pkl')

# Predict on new data
sample_scaled = loaded_scaler.transform(new_data)
prediction    = loaded_model.predict(sample_scaled)
print(f"Predicted House Value: ${prediction[0] * 100000:.0f}")
```

> **Important:** Always apply the same `StandardScaler` used during training when making new predictions. Using unscaled data will produce incorrect results.

---

## 10. Key Findings

1. **Median Income is the dominant predictor** — `MedInc` has a correlation of 0.688 with house prices, far higher than any other feature.

2. **Location matters significantly** — Both `Latitude` and `Longitude` have large coefficients, confirming that geographic location strongly influences house prices in California.

3. **Population has negligible impact** — Despite being a visible feature, `Population` contributes almost nothing to the prediction (coefficient: -0.0023).

4. **Linear Regression is a strong baseline** — With an R² of 0.5758, it captures over half the variance in house prices without any complex tuning.

5. **No missing data** — The dataset was clean and required no imputation, making it ideal for a first ML project.

6. **Outliers exist but were retained** — `Population` has extreme outliers (up to 35,682) but were kept to preserve real-world data characteristics.

---

## 11. Limitations & Future Improvements

### Current Limitations

| Limitation | Description |
|------------|-------------|
| Linear assumption | Linear Regression assumes a linear relationship — real housing data is non-linear |
| Outlier sensitivity | Extreme outliers in `Population` may be affecting model accuracy |
| Price cap | The dataset caps `MedHouseVal` at 5.0 ($500K), limiting prediction accuracy for luxury homes |
| No cross-validation | A single train/test split may produce slightly varied results across runs |

### Suggested Improvements

| Improvement | Expected Benefit |
|-------------|-----------------|
| **Random Forest Regressor** | R² > 0.80 expected |
| **XGBoost / Gradient Boosting** | Best-in-class performance |
| **Outlier removal** | Cleaner model, potentially lower RMSE |
| **Feature engineering** | Add price-per-room, distance-to-coast, etc. |
| **K-Fold Cross Validation** | More reliable performance estimate |
| **Hyperparameter tuning** | Optimal model configuration via GridSearchCV |

---

## 12. Project Structure

```
📁 AI + ML/
├── 📓 housing_project.ipynb          ← Main Jupyter Notebook
├── 🤖 california_housing_model.pkl   ← Saved trained model
├── ⚖️  california_housing_scaler.pkl  ← Saved StandardScaler
├── 📄 California_Housing_Report.md   ← This report
└── 📊 California_Housing_Prediction.pptx ← Presentation slides
```

---

## 13. How to Run

### Prerequisites

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter joblib
```

### Steps

1. Clone or open the project folder in VS Code
2. Open `housing_project.ipynb` in Jupyter
3. Select the correct Python kernel (Python 3.14)
4. Run all cells: `Kernel → Restart & Run All`
5. View outputs, plots, and evaluation metrics inline

### Quick Prediction (using saved model)

```python
import joblib, numpy as np

model  = joblib.load('california_housing_model.pkl')
scaler = joblib.load('california_housing_scaler.pkl')

# [MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]
new_house = np.array([[5.0, 20, 6.0, 1.0, 800, 3.0, 34.05, -118.25]])
scaled    = scaler.transform(new_house)
price     = model.predict(scaled)

print(f"Predicted Price: ${price[0] * 100000:,.0f}")
```

---

## Summary

This project successfully demonstrates a complete end-to-end ML pipeline using the California Housing dataset. A Linear Regression model was trained, evaluated, and saved with the following results:

| Metric | Value  |
|--------|--------|
| MAE    | 0.5332 (~$53,320 avg error) |
| RMSE   | 0.7456 |
| R²     | 0.5758 (57.6% variance explained) |

The project covers all core ML concepts — data loading, EDA, preprocessing, training, evaluation, visualization, and model persistence — making it a solid foundation for more advanced ML work.

---

*Report generated for AI/ML Internship Submission · April 2026*