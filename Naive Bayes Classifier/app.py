from flask import Flask, render_template, request
import joblib

app = Flask(__name__)


# =========================================================
# LOAD TRAINED MODEL AND VECTORIZER
# =========================================================

vectorizer = joblib.load("vectorizer.pkl")
model = joblib.load("naive_bayes_model.pkl")


# =========================================================
# 12 NEW COLLEGE-CONTEXT MESSAGES
# =========================================================

college_messages = [
    {
        "category": "Workshop Announcement",
        "message": "A hands-on Artificial Intelligence workshop will be conducted in the Computer Lab on Friday at 10 AM."
    },
    {
        "category": "Exam Notice",
        "message": "The internal examination timetable has been released. Students must check the schedule on the college portal."
    },
    {
        "category": "Placement Notification",
        "message": "The placement cell has announced a campus recruitment drive. Eligible students must register before Wednesday."
    },
    {
        "category": "Scholarship Information",
        "message": "Students eligible for the government scholarship must submit the required documents to the college office before the deadline."
    },
    {
        "category": "College Event",
        "message": "The annual technical festival will be held next week. Students interested in participating should register with their department."
    },
    {
        "category": "Assignment Reminder",
        "message": "Students must submit the Machine Learning assignment on the LMS before Friday evening."
    },
    {
        "category": "Practical Notice",
        "message": "The DBMS practical examination will be conducted in Lab 2 tomorrow. Students should bring their lab manual."
    },
    {
        "category": "Lecture Update",
        "message": "Today's Data Science lecture has been shifted to Room 305 and will begin at 11 AM."
    },
    {
        "category": "Internship Notice",
        "message": "Students selected for the summer internship must report to the training and placement office tomorrow."
    },
    {
        "category": "Coding Competition",
        "message": "Registration is open for the college coding competition. Interested students should register before the closing date."
    },
    {
        "category": "Shopping Offer",
        "message": "Exclusive student offer! Get 60 percent off on selected products. Shop now and claim your discount."
    },
    {
        "category": "Prize Promotion",
        "message": "Congratulations! You have been selected for a special reward. Click the link now to claim your free gift."
    }
]


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    message = ""
    selected_category = ""
    probabilities = None

    if request.method == "POST":

        # Get selected message number
        selected_index = request.form.get("selected_message")

        # Get custom message
        custom_message = request.form.get("custom_message", "").strip()

        # -------------------------------------------------
        # If user entered a custom message, use that
        # Otherwise use selected college message
        # -------------------------------------------------

        if custom_message:
            message = custom_message
            selected_category = "Custom Message"

        elif selected_index is not None and selected_index != "":
            index = int(selected_index)

            if 0 <= index < len(college_messages):
                message = college_messages[index]["message"]
                selected_category = college_messages[index]["category"]

        # -------------------------------------------------
        # CLASSIFICATION
        # -------------------------------------------------

        if message:

            # Convert message using the SAME vectorizer
            # used during training
            message_vector = vectorizer.transform([message])

            # Predict class
            prediction = model.predict(message_vector)[0]

            # Get prediction probabilities
            probability_values = model.predict_proba(message_vector)[0]

            # Get class names from the trained model
            class_names = model.classes_

            probabilities = []

            for class_name, probability in zip(
                class_names, probability_values
            ):
                probabilities.append({
                    "class": class_name,
                    "probability": round(probability * 100, 2)
                })


    return render_template(
        "index.html",
        messages=college_messages,
        prediction=prediction,
        message=message,
        selected_category=selected_category,
        probabilities=probabilities
    )


# =========================================================
# RUN FLASK APPLICATION
# =========================================================

if __name__ == "__main__":
    app.run(debug=True)