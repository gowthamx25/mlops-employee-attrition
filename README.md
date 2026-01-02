# MLOps Employee Attrition Prediction

An end-to-end MLOps project that predicts employee attrition using a machine learning model deployed with Flask and tracked using DVC.

---

## 🚀 Project Overview
This project demonstrates a complete MLOps workflow:
- Data versioning using DVC
- Model training and serialization
- Flask-based web application for inference

---

## 🛠 Tech Stack
- Python
- Flask
- Scikit-learn
- Pandas
- DVC
- HTML/CSS

---

## ⚙️ How to Run Locally

### 1. Clone the repository


git clone https://github.com/gowthamx25/mlops-employee-attrition.git

cd mlops-employee-attrition


### 2. Install dependencies


pip install -r requirements.txt


### 3. Pull data & model using DVC


dvc pull


### 4. Run the Flask app


python app.py


### 5. Open browser


http://127.0.0.1:5000/


---

## 📊 Output
The web app predicts whether an employee is likely to leave the organization based on input features.

---

## 🔮 Future Improvements
- CI/CD pipeline integration
- Model retraining automation
- Cloud deployment
- Monitoring & logging
