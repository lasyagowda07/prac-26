import pandas as pd 
import numpy as np
import sklearn
import matplotlib
import seaborn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

#house_price_path = "/Users/lasyar/Desktop/Self_learn/prac-26/python_prac/archive (5)/train.csv"
house_price = pd.read_csv("train.csv")

# print(house_price.describe())
# print(house_price.head())
# print(house_price.shape[0])
# print(house_price.isnull().sum())
# house_price.columns
y= house_price['TARGET(PRICE_IN_LACS)' ]
# print(house_price['TARGET(PRICE_IN_LACS)'].describe())

# # 3) Choose features (inputs)
# # Drop target (obviously) and drop ADDRESS (text column with too many unique values)
# X = house_price.drop(columns=["TARGET(PRICE_IN_LACS)", "ADDRESS"])

# # 4) Convert categorical columns into numbers (one-hot encoding)
# X = pd.get_dummies(X, drop_first=True)

# # 5) Split into train/validation (test on unseen data)
# X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=1)

# # 6) Train baseline model
# model = LinearRegression()
# model.fit(X_train, y_train)

# # 7) Predict on validation set
# preds = model.predict(X_val)

# # 8) Evaluate (MAE is easy to interpret)
# mae = mean_absolute_error(y_val, preds)
# print("Baseline Linear Regression MAE:", mae)

df = house_price.copy()
upper_cap = df['TARGET(PRICE_IN_LACS)'].quantile(0.99)
\
df = df[df['TARGET(PRICE_IN_LACS)'] <= upper_cap]

y = df['TARGET(PRICE_IN_LACS)']

X = df.drop(columns=['TARGET(PRICE_IN_LACS)', 'ADDRESS'])
X = pd.get_dummies(X, drop_first=True)

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, random_state=1
)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

from sklearn.metrics import mean_absolute_error
preds = model.predict(X_val)
mae = mean_absolute_error(y_val, preds)

print("Option B (Outlier-filtered) MAE for baseline:", mae)


# import pandas as pd
# import numpy as np

# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_absolute_error

# # 1) Load data
# df = pd.read_csv("train.csv")

# # 2) Define target and apply log transform
# y_raw = df["TARGET(PRICE_IN_LACS)"]
# y = np.log1p(y_raw)   # log(1 + price) to handle skew safely

# # 3) Define features
# # Drop target (to avoid leakage) and ADDRESS (high-cardinality text)
# X = df.drop(columns=["TARGET(PRICE_IN_LACS)", "ADDRESS"])

# # 4) One-hot encode categorical variables
# X = pd.get_dummies(X, drop_first=True)

# # 5) Train/validation split
# X_train, X_val, y_train, y_val = train_test_split(
#     X, y, random_state=1
# )

# # 6) Train model
# model = LinearRegression()
# model.fit(X_train, y_train)

# # 7) Predict in log space
# pred_log = model.predict(X_val)

# # 8) Convert predictions back to original scale (lakhs)
# pred = np.expm1(pred_log)
# y_val_original = np.expm1(y_val)

# # 9) Evaluate
# mae = mean_absolute_error(y_val_original, pred)
# print("Option C (Log-transformed) MAE:", mae)



# errors = y_val - preds
# print(errors.describe())
# print(errors.abs().sort_values(ascending=False).head(10))



rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    random_state=1,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

rf_preds = rf_model.predict(X_val)
rf_mae = mean_absolute_error(y_val, rf_preds)

print("Random Forest MAE:", rf_mae)

import pandas as pd

feature_importance = pd.Series(
    rf_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print(feature_importance.head(10))



