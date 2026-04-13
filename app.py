from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model, le_soil, le_crop, le_fertilizer = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    temp = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    moisture = float(request.form['moisture'])
    soil = le_soil.transform([request.form['soil']])[0]
    crop = le_crop.transform([request.form['crop']])[0]
    n = float(request.form['n'])
    p = float(request.form['p'])
    k = float(request.form['k'])

    data = np.array([[temp, humidity, moisture, soil, crop, n, p, k]])

    prediction = model.predict(data)
    result = le_fertilizer.inverse_transform(prediction)[0]

    return render_template('index.html', prediction_text=f"Recommended Fertilizer: {result}")

if __name__ == "__main__":
    app.run(debug=True)