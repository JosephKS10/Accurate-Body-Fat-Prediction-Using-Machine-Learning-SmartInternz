from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')

# Route for rendering the HTML form
@app.route('/')
def home():
    age = 0;
    return render_template('index.html',check=False, age=age)

# Route for handling form submission
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    age = int(request.form['age'])
    weight = float(request.form['weight'])
    height = float(request.form['height'])
    neck = float(request.form['neck'])
    chest = float(request.form['chest'])
    abdomen = float(request.form['abdomen'])
    hip = float(request.form['hip'])
    thigh = float(request.form['thigh'])
    knee = float(request.form['knee'])
    ankle = float(request.form['ankle'])
    biceps = float(request.form['bicep'])
    forearm = float(request.form['forearm'])
    wrist = float(request.form['wrist'])
    density = 1.0708

    features = [density, age, weight, height, neck, chest, abdomen, hip, thigh, knee, ankle, biceps, forearm, wrist]
    prediction = model.predict([features])[0]
    return render_template('index.html', prediction=prediction, check=True,age=age)

if __name__ == '__main__':
    app.run(debug=True)