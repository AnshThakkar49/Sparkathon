from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the trained model and scaler
with open('best_sales_prediction_model.pkl', 'rb') as file:
    rf_model = pickle.load(file)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/p')
def predict():
    return render_template('prediction_page.html')

@app.route('/predict', methods=['GET','POST'])
def prediction_page():
    try:
        # Extract input values from the form
        store = int(request.form['store'])
        is_holiday = int(request.form['holiday'])
        temperature = float(request.form['temperature'])
        fuel_price = float(request.form['fuel_price'])
        cpi = float(request.form['cpi'])
        unemployment = float(request.form['unemployment'])
        year = (request.form['year'])
        month = int(request.form['month'])
        week = int(request.form['week'])

        # Create a DataFrame from the inputs
        input_data = np.array([[store, is_holiday, temperature, fuel_price, cpi, unemployment, year, month, week]])

        # Scale the input data
        #scaler=StandardScaler()
        #input_data_scaled = scaler.fit(input_data)
        #input_data_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = rf_model.predict(input_data)

        return render_template('result.html', prediction=prediction, year=year, month=month, week=week)

    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
