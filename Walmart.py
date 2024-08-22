import pickle
import pandas as pd

# Load the dataset
file_path = 'Walmart.csv'
data = pd.read_csv(file_path)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# Convert 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')

# Feature Engineering: Extract year, month, and week from the Date
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Week'] = data['Date'].dt.isocalendar().week

# Drop the original Date column
data = data.drop('Date', axis=1)

# Define features and target variable
x = data.drop('Weekly_Sales', axis=1)
y = data['Weekly_Sales']

# Split the data into training and testing sets (80% train, 20% test)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

x_train_scaled[:5], x_test_scaled[:5]

from sklearn.ensemble import RandomForestRegressor

# Initialize and train the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(x_train, y_train)

# Make predictions on the test set
#y_pred_rf = rf_model.predict(x_test)

pickle.dump(rf_model, open('best_sales_prediction_model.pkl', 'wb'))