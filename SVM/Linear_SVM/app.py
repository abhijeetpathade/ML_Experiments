
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained Linear SVM pipeline
model = joblib.load("linear_svm_banknote_model.pkl")


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None
    values = {
        "variance": "",
        "skewness": "",
        "curtosis": "",
        "entropy": ""
    }

    if request.method == "POST":
        try:
            # Read the four banknote features
            for feature in values:
                values[feature] = request.form.get(feature, "").strip()

            if any(value == "" for value in values.values()):
                raise ValueError("Please enter all four feature values.")

            features = np.array([[
                float(values["variance"]),
                float(values["skewness"]),
                float(values["curtosis"]),
                float(values["entropy"])
            ]])

            if not np.isfinite(features).all():
                raise ValueError("Please enter valid finite numbers.")

            # Predict using the trained model
            result = int(model.predict(features)[0])

            if result == 0:
                prediction = {
                    "label": "Genuine Banknote",
                    "description": "The model classifies this sample as genuine.",
                    "category": "genuine"
                }
            else:
                prediction = {
                    "label": "Counterfeit Banknote",
                    "description": "The model classifies this sample as counterfeit.",
                    "category": "counterfeit"
                }

        except ValueError as exc:
            error = str(exc) or "Please enter valid numerical values."
        except Exception:
            error = "Unable to classify the sample. Check the model file and input values."

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        values=values
    )


if __name__ == "__main__":
    app.run(debug=True)
