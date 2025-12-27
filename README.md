# 🩺 Hypertension Risk Prediction System

A **Machine Learning powered Django web application** that predicts the risk of **Hypertension (High Blood Pressure)** based on user health inputs.  
This project is designed as a **real-world healthcare ML application**, suitable for **final year projects, interviews, and resumes**.

---

## 🔍 Project Description

Hypertension is one of the most common and dangerous lifestyle diseases. Early detection helps prevent heart attacks, strokes, and kidney failure.

This project:
- Uses a **trained Machine Learning classification model**
- Integrates the model into a **Django web application**
- Accepts user health details via a web form
- Predicts whether the user is **at risk of hypertension**

---

## ✨ Features

✔ User-friendly web interface  
✔ Machine Learning model integration  
✔ Real-time prediction  
✔ Django backend with SQLite database  
✔ Pre-trained model (`hypertension_model.sav`)  
✔ Clean project structure  

---

## 🧠 Machine Learning Overview

- **Problem Type:** Binary Classification  
- **Target:** Hypertension Risk (Yes / No)  
- **Model:** Trained using Scikit-Learn  
- **Model Storage:** Serialized using `.sav` file  
- **Prediction Flow:**  
  User Input → Preprocessing → ML Model → Result

---

## 🛠️ Technologies Used

| Layer | Technology |
|-----|-----------|
| Programming | Python 3 |
| Framework | Django |
| ML Library | Scikit-Learn |
| Data Handling | Pandas, NumPy |
| Database | SQLite |
| Frontend | HTML, CSS (Django Templates) |

---

## 📁 Project Structure

```text
Hypertension-risk-predection/
│
├── assets/                   # Static files (CSS, images)
├── media/                    # Uploaded / generated media
├── users/                    # Django app (user management)
├── bloodlevel/               # Django app (prediction logic)
│
├── hypertension_model.sav    # Trained ML model
├── db.sqlite3                # Database
├── manage.py                 # Django entry point
├── requirement.txt           # Python dependencies
└── README.md                 # Project documentation



---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/premsaikamadani/Hypertension-risk-predection.git
cd Hypertension-risk-predection

2️⃣ Create Virtual Environment
python -m venv venv

Windows
venv\Scripts\activate

Linux / Mac
source venv/bin/activate

3️⃣ Install Dependencies
pip install -r requirement.txt

▶️ Run the Application
python manage.py makemigrations
python manage.py migrate
python manage.py runserver

Open browser and visit:
👉 http://127.0.0.1:8000/

📊 Prediction Workflow
User enters health data
Django backend receives input
Data is processed
Trained ML model predicts risk
Result displayed on UI


📦 Model Details
Model file: hypertension_model.sav
Loaded using joblib / pickle
Used directly inside Django views
No retraining required to run app

🚀 Future Enhancements
Deploy on AWS / Render
Use PostgreSQL instead of SQLite
Improve UI using Bootstrap / React
Add prediction history tracking

👤 Author
Prem Sai Kamdani
GitHub: https://github.com/premsaikamadani


