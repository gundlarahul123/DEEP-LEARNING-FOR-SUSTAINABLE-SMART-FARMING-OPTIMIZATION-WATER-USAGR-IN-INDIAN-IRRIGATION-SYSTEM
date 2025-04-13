# DEEP-LEARNING-FOR-SUSTAINABLE-SMART-FARMING-OPTIMIZATION-WATER-USAGR-IN-INDIAN-IRRIGATION-SYSTEM


"Deep Learning for Sustainable Smart Farming Optimization: Water Usage in Indian Irrigation System"

This README provides clarity, documentation, and organization suitable for GitHub and will help users or collaborators understand the scope, usage, and execution of your project.

markdown
Copy
Edit
# 🌾 Deep Learning for Sustainable Smart Farming Optimization: Water Usage in Indian Irrigation System

This Django-based web application applies machine learning models to predict and optimize water usage in Indian irrigation systems, helping in sustainable smart farming practices.

---

## 🚀 Features

- User Authentication (Register, Login, Logout)
- Prediction of water requirements using:
  - Random Forest Classifier
  - SVM
  - Decision Tree
  - Logistic Regression
  - Passive Aggressive Classifier
- Interactive dashboards to display model accuracy and predictions
- Visualization support for analytical insights

---

## 🛠️ Tech Stack

- **Backend**: Python, Django
- **Machine Learning**: scikit-learn, pandas, numpy, seaborn, matplotlib
- **Frontend**: HTML Templates (Jinja), Bootstrap
- **Data**: Custom CSV Dataset with water usage features

---

## 📁 Directory Structure

smart_farming/ │ ├── templates/ # HTML Templates │ ├── home1.html │ ├── loginform.html │ ├── register.html │ ├── result.html │ └── acc1.html │ ├── static/ # Static files (CSS, JS) │ ├── models.py # User model definition ├── views.py # Main logic (ML and Web views) ├── urls.py # URL Routing ├── smart_farming.csv # Main dataset └── manage.py # Django Management Script

yaml
Copy
Edit

---

## 📊 Dataset Description

The dataset (`smart_farming.csv`) contains 16 input features representing environmental and agricultural parameters such as:

- Soil Moisture
- Temperature
- Humidity
- pH levels
- Crop Type Indicators
- Water Requirements

`label`: Target variable predicting optimal water usage or farming action.

---

## 🧠 Machine Learning Models Used

| Model                         | Purpose                          |
|------------------------------|----------------------------------|
| RandomForestClassifier       | High accuracy ensemble model     |
| LogisticRegression           | Baseline linear model            |
| DecisionTreeClassifier       | Fast, interpretable predictions  |
| SVC (Support Vector Machine) | Accurate margin-based classifier |
| PassiveAggressiveClassifier  | Adaptive online learning         |

---

## 🔑 User Flow

1. **Register/Login** to the web app.
2. Input 16 values related to the environment/farming conditions.
3. Choose a model (e.g., Random Forest, SVM) to make predictions.
4. View predicted class/label and model accuracy.
5. Optionally visualize graphs and performance metrics.

---

## ⚙️ How to Run

### Step 1: Clone the Repository

git clone https://github.com/yourusername/smart-farming-water-usage.git
cd smart-farming-water-usage
Step 2: Create Virtual Environment & Install Dependencies
python -m venv venv
source venv/bin/activate  # For Linux/Mac
venv\Scripts\activate     # For Windows

pip install -r requirements.txt
Note: If requirements.txt is not present, manually install: Django, scikit-learn, numpy, pandas, matplotlib, seaborn, scipy

Step 3: Run Migrations
python manage.py makemigrations
python manage.py migrate
Step 4: Run the Server
python manage.py runserver
Access the app at: http://127.0.0.1:8000

📌 Screenshots
Add screenshots of:

Homepage

Prediction form

Results page

Accuracy comparison

📈 Future Enhancements
Integration with IoT sensors for live data

Deployment on cloud (Heroku/AWS)

More sophisticated deep learning models (LSTM, CNN)

Mobile app interface

📄 License
This project is open-source and available under the MIT License.

🙏 Acknowledgements
scikit-learn

Django Documentation

[Indian Agricultural Data Repositories]

👨‍💻 Author
Rahul – Smart Farming ML Project
LinkedIn • GitHub










