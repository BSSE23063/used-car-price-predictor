import pandas as pd
import numpy as np
import re 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('used_cars.csv')

df['milage'] = df['milage'].astype(str).str.replace(' mi.', '', regex=False)
df['milage'] = df['milage'].str.replace(' mi', '', regex=False) 
df['milage'] = df['milage'].str.replace(',', '', regex=False)
df['milage'] = pd.to_numeric(df['milage'], errors='coerce')
df['milage'] = df['milage'].fillna(df['milage'].mean())

df['price'] = df['price'].astype(str).str.replace('$', '', regex=False)
df['price'] = df['price'].str.replace(',', '', regex=False)
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df['price'] = df['price'].fillna(df['price'].mean())

def get_hp(text):
    if pd.isna(text): return None
    match = re.search(r'(\d+\.?\d*)HP', str(text))
    if match:
        return float(match.group(1))
    return None

df['Horsepower'] = df['engine'].apply(get_hp)
df['Horsepower'] = df['Horsepower'].fillna(df['Horsepower'].mean())

df['accident_binary'] = df['accident'].apply(lambda x: 1 if 'At least 1' in str(x) else 0)

df['clean_title_binary'] = df['clean_title'].apply(lambda x: 1 if x == 'Yes' else 0)

df['Car_Age'] = 2026 - df['model_year']

df = df[(df['price'] > 1000) & (df['price'] < 100000)]

df = pd.get_dummies(df, columns=['fuel_type', 'brand'], prefix=['fuel', 'brand'], drop_first=True)

garbage_cols = ['fuel_', 'fuel_not supported', 'fuel_']
df = df.drop(columns=[c for c in garbage_cols if c in df.columns], errors='ignore')

cols_to_drop = ['model', 'engine', 'transmission', 'int_col', 'ext_col', 'accident', 'clean_title', 'model_year']
df = df.drop(columns=cols_to_drop, errors='ignore')

print(f"Final Data Shape: {df.shape}")
print("Data Types Check (Should all be numbers):")
print(df.select_dtypes(include=['object']).columns) 

X = df.drop('price', axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
score = r2_score(y_test, y_pred)

print(f"\nModel Accuracy (R^2 Score): {score:.4f}")

comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print("\n--- First 5 Predictions ---")
print(comparison.head())