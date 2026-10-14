
from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained Non-Linear SVM pipeline
model = joblib.load("nonlinear_svm_banknote_model.pkl")


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
            for feature in values:
                values[feature] = request.form.get(feature, "").strip()
                if not values[feature]:
                    raise ValueError("Please enter all four feature values.")

            sample = pd.DataFrame([{
                "variance": float(values["variance"]),
                "skewness": float(values["skewness"]),
                "curtosis": float(values["curtosis"]),
                "entropy": float(values["entropy"])
            }])

            result = int(model.predict(sample)[0])

            if result == 0:
                prediction = "Genuine Banknote"
            else:
                prediction = "Counterfeit Banknote"

        except ValueError as e:
            error = str(e) or "Please enter valid numerical values."
        except Exception:
            error = "Prediction failed. Check the input values and trained model."

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        values=values
    )


if __name__ == "__main__":
    app.run(debug=True)
