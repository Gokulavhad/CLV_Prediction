 # We'll use basic RFM and a regression model like XGBoost for prediction.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# Load the data
data_path = "online_retail.xlsx"
data = pd.read_excel(data_path)

# Data preprocessing
data.dropna(subset=['Customer ID'], inplace=True)
data = data[~data['Invoice'].astype(str).str.startswith('C')]
data['TotalPrice'] = data['Quantity'] * data['Price']

# Set reference date
latest_date = data['InvoiceDate'].max()

# RFM Calculation
rfm = data.groupby('Customer ID').agg({
    'InvoiceDate': lambda x: (latest_date - x.max()).days,
    'Invoice': 'nunique',
    'TotalPrice': 'sum'
}).reset_index()

rfm.columns = ['Customer ID', 'Recency', 'Frequency', 'Monetary']

# Add Customer Tenure
customer_first_purchase = data.groupby('Customer ID')['InvoiceDate'].min().reset_index()
customer_first_purchase.columns = ['Customer ID', 'FirstPurchaseDate']
rfm = rfm.merge(customer_first_purchase, on='Customer ID')
rfm['Tenure'] = (latest_date - rfm['FirstPurchaseDate']).dt.days

# Use Monetary as target for future value approximation
X = rfm[['Recency', 'Frequency', 'Tenure']]
y = rfm['Monetary']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = GradientBoostingRegressor()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

# Save results
rfm['PredictedCLV'] = model.predict(rfm[['Recency', 'Frequency', 'Tenure']])

# Output
output_path = "CLV_Prediction.csv"
rfm.to_csv(output_path, index=False)

output_path, mae

