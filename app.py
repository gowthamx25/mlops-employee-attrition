from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import pickle

app = Flask(__name__)

# Load trained model
with open("logistic_regression_model.pkl", "rb") as file:
    logreg_classifier = pickle.load(file)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract data from form
        features = [
            float(request.form["job_level"]),
            float(request.form["monthly_income"]),
            float(request.form["total_working_years"]),
            float(request.form["percent_salary_hike"]),
            float(request.form["performance_rating"]),
            float(request.form["years_at_company"]),
            float(request.form["years_in_current_role"]),
            float(request.form["years_with_curr_manager"])
        ]

        # Convert to DataFrame
        x_new = pd.DataFrame([features], columns=[
            'JobLevel', 'MonthlyIncome', 'TotalWorkingYears',
            'PercentSalaryHike', 'PerformanceRating', 'YearsAtCompany',
            'YearsInCurrentRole', 'YearsWithCurrManager'
        ])
        
        # Make prediction
        prediction = logreg_classifier.predict(x_new)[0]
        probability = logreg_classifier.predict_proba(x_new)[0][1]

        # Determine prediction message
        message = "Employee will leave the company" if prediction == 1 else "Employee will not leave the company"

        # Pass results to result.html
        return render_template("result.html", prediction=message, probability=round(probability * 100, 2))

    except Exception as e:
        return render_template("result.html", error=str(e))

if __name__ == "__main__":
    app.run(debug=True)
