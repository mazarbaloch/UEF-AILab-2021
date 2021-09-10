import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

df = pd.read_csv('data/breast-cancer.txt')

df.dropna(inplace=True)
df = df._get_numeric_data()

df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)

# scaler2 = StandardScaler()
# scaler2.fit(X)

y = np.array(y == 4).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(X_train, X_test, y_train, y_test)
