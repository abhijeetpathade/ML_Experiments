from flask import Flask, render_template, request
import joblib



app = Flask(__name__)

# Load trained model and vectorizer
model = joblib.load("naive_bayes_model2.pkl")
vectorizer = joblib.load("vectorizer2.pkl")

# Sample college messages for dropdown
college_messages = [
    {
        "category": "Academic",
        "message": "The DBMS practical examination is scheduled for Tuesday in Lab 2."
    },
    {
        "category": "Academic",
        "message": "Students must submit the Machine Learning assignment before Friday."
    },
    {
        "category": "Academic",
        "message": "The semester examination timetable has been released on the college portal."
    },
    {
        "category": "Academic",
        "message": "The Data Science lecture has been shifted to Room 305."
    },

    {
        "category": "Placement",
        "message": "The placement cell has announced a campus recruitment drive for eligible students."
    },
    {
        "category": "Placement",
        "message": "Shortlisted students must report to the placement office tomorrow."
    },
    {
        "category": "Placement",
        "message": "The aptitude test for the campus placement will be conducted on Saturday."
    },
    {
        "category": "Placement",
        "message": "Eligible students must upload their resumes to the placement portal."
    },

    {
        "category": "Promotional",
        "message": "Exclusive student offer! Get 60 percent off on selected products."
    },
    {
        "category": "Promotional",
        "message": "Congratulations! Claim your special reward and free gift now."
    },
    {
        "category": "Promotional",
        "message": "Limited time discount available for students. Shop now."
    },
    {
        "category": "Promotional",
        "message": "Special sale today! Get exciting offers on your favorite products."
    }
]


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    message = ""
    selected_category = ""
    probabilities = None

    if request.method == "POST":

        selected_index = request.form.get("selected_message")
        custom_message = request.form.get("custom_message", "").strip()

        # Give priority to custom message
        if custom_message:
            message = custom_message
            selected_category = "Custom Message"

        # Otherwise use dropdown message
        elif selected_index:
            index = int(selected_index)

            if 0 <= index < len(college_messages):
                message = college_messages[index]["message"]
                selected_category = college_messages[index]["category"]

        # Make prediction
        if message:

            message_vector = vectorizer.transform([message])

            prediction = model.predict(message_vector)[0]

            probability_values = model.predict_proba(message_vector)[0]

            class_names = model.classes_

            probabilities = []

            for class_name, probability in zip(
                class_names,
                probability_values
            ):
                probabilities.append({
                    "class": class_name,
                    "probability": round(probability * 100, 2)
                })

    return render_template(
        "index2.html",
        messages=college_messages,
        prediction=prediction,
        message=message,
        selected_category=selected_category,
        probabilities=probabilities
    )


if __name__ == "__main__":
    app.run(debug=True)