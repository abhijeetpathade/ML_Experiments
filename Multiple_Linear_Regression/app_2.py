from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained Multiple Linear Regression model
model = pickle.load(open("MLRModel__.pkl", "rb"))


@app.route('/')
def home():
    return render_template("index_2.html")


@app.route('/predict', methods=['POST'])
def predict():

    # Read inputs from the HTML form
    tv = float(request.form['tv'])
    Newspaper = float(request.form['Newspaper'])

    # Predict sales
    prediction = model.predict(np.array([[tv, Newspaper]]))

    return render_template(
        "index_2.html",
        prediction_text=f"Predicted Sales : {prediction[0]:.2f}"
    )


if __name__ == "__main__":
    app.run(debug=True)
