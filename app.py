import pickle
import numpy as np
from flask import Flask, render_template, request

# Load the model
with open('best_sales_prediction_model.pkl', 'rb') as file:
    rf_model = pickle.load(file)

# Check what was loaded
print(f"Loaded object type: {type(rf_model)}")

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('prediction_page.html')

@app.route('/predict', methods=['POST'])
def prediction_page():
    year = request.form['year']
    month = request.form['month']
    week = request.form['week']

    # Convert input data to the required format
    arr = np.array([[week, month, year]])
    
    # Make sure to handle the prediction correctly
    try:
        prediction_value = rf_model.predict(arr)
    except AttributeError as e:
        return f"Error: {str(e)}"
    
    return render_template('result.html', 
                           year=year, 
                           month=month, 
                           week=week, 
                           prediction_value=prediction_value[0])

if __name__ == '__main__':
    app.run(debug=True)
