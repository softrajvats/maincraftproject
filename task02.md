# run this in terminal

pip install scikit-learn pandas numpy matplotlib seaborn

# make a new file -> task02.ipynb

-----------------------------------------

# do testing first

```
import sklearn
print(sklearn.__version__)
```

------------------------------------------

# 🔷 Cell 1 — Import Libraries

```
# Cell 1: Import all required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("✅ All libraries imported successfully!")
```

- This one is going to take some time ~ have patience wait and let it load all the heavy data
