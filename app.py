from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('car_price_model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    vehicle_age = int(request.form['vehicle_age'])
    km_driven = int(request.form['km_driven'])
    mileage = float(request.form['mileage'])
    engine = int(request.form['engine'])
    max_power = float(request.form['max_power'])
    seats = int(request.form['seats'])

    seller_type = request.form['seller_type']
    fuel_type = request.form['fuel_type']
    transmission_type = request.form['transmission_type']

    final_features = np.array([[
        vehicle_age,
        km_driven,
        mileage,
        engine,
        max_power,
        seats,

        1 if seller_type == 'Individual' else 0,
        1 if seller_type == 'Trustmark Dealer' else 0,

        1 if fuel_type == 'Diesel' else 0,
        1 if fuel_type == 'LPG' else 0,
        1 if fuel_type == 'Petrol' else 0,

        1 if transmission_type == 'Manual' else 0
    ]])

    prediction = model.predict(final_features)

    return render_template(
        'index.html',
        prediction_text=f'Predicted Price: ₹ {round(float(prediction[0]), 2)}'
    )

if __name__ == '__main__':
    app.run(debug=True, port=8001)
