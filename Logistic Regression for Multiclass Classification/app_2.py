from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load Trained Model
model = pickle.load(open("MCModel_2.pkl", "rb"))


@app.route('/')
def home():
    return render_template("index_2.html")


@app.route('/predict', methods=['POST'])
def predict():

    # Read input values
    working_hours = float(request.form['working_hours'])
    monthly_salary = float(request.form['monthly_salary'])

    # Prediction
    prediction = model.predict([[working_hours, monthly_salary]])

    # Convert class number into satisfaction level
    if prediction[0] == 0:
        satisfaction = "Low Satisfaction"

    elif prediction[0] == 1:
        satisfaction = "Medium Satisfaction"

    else:
        satisfaction = "High Satisfaction"

    return render_template(
        "index_2.html",
        prediction_text=f"Predicted Job Satisfaction : {satisfaction}"
    )


if __name__ == "__main__":
    app.run(debug=True)
