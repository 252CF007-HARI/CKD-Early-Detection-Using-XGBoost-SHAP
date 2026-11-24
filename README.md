# CKD Project – Professional Complete

## Project
Machine Learning Approach for Early Detection of Chronic Kidney Disease.

## Methodology
1. Load UCI CKD dataset
2. Preprocess clinical data
3. PCA-based feature analysis
4. Use 12 selected original clinical features
5. Train XGBoost classifier
6. Evaluate using Accuracy, Precision, Recall, F1-score and ROC-AUC
7. Flask web application for CKD/Not CKD prediction
8. SHAP support is included for explainability

## 12 project features
age, bp, sg, bgr, bu, sc, sod, pot, hemo, pcv, wbcc, rbcc

## Run
Use Python 3.12 in the virtual environment:
.env\Scripts\python.exe -m pip install -r requirements.txt
.env\Scripts\python.exe train_model.py
.env\Scripts\python.exe app.py

Then open http://127.0.0.1:5000
