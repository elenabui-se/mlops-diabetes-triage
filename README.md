# MLOps Diabetes Triage

This repository contains a complete MLOps pipeline for a **Virtual Diabetes Clinic Triage** system.  
It predicts short-term diabetes progression risk using the open scikit-learn *diabetes* dataset and serves predictions via a FastAPI service.

---

## 👩‍💻 Developed by
- Dieu Nga Bui  
- Artem Matat  
- Mohammad Rezaeian Nojani  
- Che Sihang  

---

## 🧠 Project Overview

The goal is to automate the triage process by predicting patient risk scores and prioritizing follow-ups.  
The workflow includes model training, API deployment, CI/CD automation with GitHub Actions, and Docker containerization.

---

## 🚀 Tasks Completed

### **v0.1 – Baseline**
- Implemented and trained a **Linear Regression** model using scikit-learn.  
- Scaled the dataset using **StandardScaler**.  
- Saved artifacts (`model.joblib`, `scaler.joblib`).  
- Built an API using **FastAPI** with:
  - `/health` → API health check  
  - `/predict` → Predict diabetes progression  
- Created a **Dockerfile** for containerization.  
- Configured **GitHub Actions** for CI pipeline and automated testing.

### **v0.2 – Model Improvement**
- Added **RandomForestRegressor** for non-linear performance improvement.  
- Logged improved metrics (lower RMSE).  
- Updated API version to **v0.2**.  
- Published both **v0.1** and **v0.2** releases to GitHub.  
- Added `CHANGELOG.md` to document changes.

---

## 🧩 How to Run Locally

```bash
# 1. Clone repository
git clone https://github.com/elenabui-se/mlops-diabetes-triage.git
cd mlops-diabetes-triage

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate      # (Windows)
# or source venv/bin/activate (Mac/Linux)

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train model
python src/train_model.py

# 5. Run API
uvicorn main:app --reload
API will be available at → http://127.0.0.1:8000/docs

🩺 Example Requests
Health Check
GET /health
Response:

json
Copy code
{"status": "ok", "model_version": "v0.2"}
Predict Progression
POST /predict
Request body:

json
Copy code
{
  "age": 0.02, "sex": -0.044, "bmi": 0.06, "bp": -0.03,
  "s1": -0.02, "s2": 0.03, "s3": -0.02, "s4": 0.02,
  "s5": 0.02, "s6": -0.001
}
Response:

json
Copy code
{"prediction": 235.94}
🧾 CI/CD Pipeline
GitHub Actions runs on every push and pull request.

Linting, testing, and model smoke tests are automated.

Tagged versions (v0.1, v0.2) build Docker images and publish to GitHub Container Registry (GHCR).

📦 Releases
Version	Model	Description
v0.1	LinearRegression	Baseline model and working API
v0.2	RandomForestRegressor	Improved accuracy and model performance
