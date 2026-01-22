import pandas as pd 
import numpy as np
import sklearn
import matplotlib
import seaborn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

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
print("99th percentile price:", upper_cap)
df = df[df['TARGET(PRICE_IN_LACS)'] <= upper_cap]
print(df['TARGET(PRICE_IN_LACS)'].describe())
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

print("Option B (Outlier-filtered) MAE:", mae)