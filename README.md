MLOps Diabetes Triage

This repository contains the first part of our project - Machine Learning model training and FastAPI deployment.

Developed by Artem Matat

Tasks completed:

- Implemented and trained a Linear Regression model using scikit-learn.
- Scaled the dataset using StandardScaler.
- Saved the model and scaler (model.joblib, scaler.joblib).
- Built an API with FastAPI (main.py) including endpoints:
     - /health -> API health check
     - /predict -> Predict diabetes progression
- Created a Dockerfile to containerize the project.
- Generated requirements.txt for dependency management.

How to run locally:

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python src/train_model.py
uvicorn main:app --reload


API available at:
http://127.0.0.1:8000/docs
