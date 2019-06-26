#DATASET: https://www.kaggle.com/himanshupoddar/zomato-bangalore-restaurants

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#from sklearn.preprocessing import LabelEncoder
#from sklearn.preprocessing import OneHotEncoder

filename = 'zomato.csv'
df = pd.read_csv(filename, dtype={'approx_cost(for two people)': str})
print("length before NA drop: ", len(df))
df.dropna(inplace=True)
print("length after NA drop: ", len(df))
df['cost'] = df['approx_cost(for two people)'].astype(str)
df['cost'] = df['cost'].apply(lambda x: float(x.replace(",", "")))
df['score'] = df['rate'].apply(lambda x: x.split("/")[0])
df['score'] = df['score'].apply(lambda x: 0.0 if x == 'NEW' else float(x))

# Get a list of the interesting features
dfsub = df.iloc[:, [2, 3, 4, 6, 8, 9, 11, 15, 16, 17, 18]]

# Do one-hot encoding on categorical variables
cat_cols = ['online_order', 'book_table', 'location', 'rest_type', 'cuisines', 'listed_in(type)', 'listed_in(city)']
dflist = []
for val in cat_cols:
    temp = pd.get_dummies(dfsub.loc[:, val])
    dflist.append(temp)

dfoh = pd.concat(dflist, axis=1)
dfsub = dfsub.drop(cat_cols, axis=1)
dfsub = dfsub.drop('name', axis=1)
dffinal = pd.concat([dfoh, dfsub], axis=1)

#dffinal['votes'] = (dffinal['votes'] - dffinal['votes'].mean()) / dffinal['votes'].std()

y = dffinal.iloc[:, -1].values
X = dffinal.iloc[:, :-1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
xg_reg = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1, max_depth=5)
xg_reg.fit(X_train, y_train)
preds = xg_reg.predict(X_test)

plt.scatter(preds, y_test)
plt.show()