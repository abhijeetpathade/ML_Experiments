from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load Trained Logistic Regression Model
model = pickle.load(open("BCModel2.pkl", "rb"))


@app.route('/')
def home():
    return render_template("index2.html")


@app.route('/predict', methods=['POST'])
def predict():

    # Read user inputs
    age = float(request.form['age'])
    salary = float(request.form['salary'])

    # Prediction
    prediction = model.predict(np.array([[age, salary]]))

    # Convert numeric prediction to text
    if prediction[0] == 1:
        result = "Customer is Likely to PURCHASE"
    else:
        result = "Customer is NOT Likely to PURCHASE"

    return render_template(
        "index2.html",
        prediction_text=result
    )


if __name__ == "__main__":
    app.run(debug=True)