from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the trained model and scaler
with open('best_sales_prediction_model.pkl', 'rb') as file:
    rf_model = pickle.load(file)


@app.route('/')
def index():
    return render_template('prediction_page.html')

@app.route('/predict', methods=['POST'])
def prediction_page():
    try:
        # Extract input values from the form
        store = int(request.form['store'])
        dept = int(request.form['dept'])
        is_holiday = int(request.form['is_holiday'])
        temperature = float(request.form['temperature'])
        fuel_price = float(request.form['fuel_price'])
        cpi = float(request.form['cpi'])
        unemployment = float(request.form['unemployment'])
        year = int(request.form['year'])
        month = int(request.form['month'])
        week = int(request.form['week'])

        # Create a DataFrame from the inputs
        input_data = pd.DataFrame({
            'Store': [store],
            'Dept': [dept],
            'IsHoliday': [is_holiday],
            'Temperature': [temperature],
            'Fuel_Price': [fuel_price],
            'CPI': [cpi],
            'Unemployment': [unemployment],
            'Year': [year],
            'Month': [month],
            'Week': [week]
        })

        # Scale the input data
        scaler=StandardScaler()
        input_data_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = rf_model.predict(input_data_scaled)[0]

        return render_template('index.html', prediction=prediction)

    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
