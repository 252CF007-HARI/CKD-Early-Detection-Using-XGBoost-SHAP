# 🩺 Explainable Machine Learning Approach for Early Detection of Chronic Kidney Disease

## 📌 About the Project

This project develops a machine learning-based web application for the early detection of Chronic Kidney Disease (CKD). It uses the XGBoost algorithm to predict whether a patient may have CKD based on medical parameters.

The system integrates explainable AI techniques using SHAP to provide insights into the factors influencing the prediction, making the model's output easier to understand.

## 🎯 Objectives

- Predict Chronic Kidney Disease using machine learning.
- Identify important medical features using PCA-based feature analysis.
- Implement XGBoost for classification.
- Provide prediction confidence and risk level.
- Explain predictions using SHAP.
- Develop an interactive web application using Flask.

## 🛠️ Technologies Used

- **Programming Language:** Python
- **Machine Learning:** XGBoost, Scikit-learn
- **Explainable AI:** SHAP
- **Data Processing:** Pandas, NumPy
- **Feature Analysis:** PCA
- **Web Framework:** Flask
- **Frontend:** HTML, CSS, Bootstrap, JavaScript
- **Visualization:** Matplotlib / Seaborn (if used)

## 📊 Dataset

The project uses the Chronic Kidney Disease dataset containing 400 patient records and 24 input features.

Feature analysis was conducted to identify relevant features for the prediction process.

## 🤖 Machine Learning Model

The XGBoost classifier is used to predict CKD status based on the selected medical features.

### Model Performance

| Metric | Score |
|---|---:|
| Accuracy | 96.25% |
| Precision | 100% |
| Recall | 94.00% |
| F1-Score | 96.91% |

*Results are based on the project's current evaluation and test split.*

## 🔍 Explainable AI

SHAP (SHapley Additive exPlanations) is implemented to explain model predictions by identifying the features that contribute to the predicted outcome.

This helps users understand which medical parameters influenced the prediction.

## ✨ Features

- CKD prediction
- Non-CKD prediction
- Prediction confidence score
- Risk level classification
- SHAP-based explanation
- Interactive prediction interface
- PDF patient report generation (if implemented)

## 🏗️ Project Architecture

1. Data collection
2. Data preprocessing
3. Feature analysis and selection
4. Model training using XGBoost
5. Model evaluation
6. SHAP explainability
7. Flask web application
8. Prediction and result visualization

## 🚀 Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/CKD-Early-Detection-Using-XGBoost-SHAP.git
```

### 2. Navigate to the project folder

```bash
cd CKD-Early-Detection-Using-XGBoost-SHAP
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
python app.py
```

Open the local URL displayed in your terminal to access the web application.

## ⚠️ Disclaimer

This project is developed for educational and research purposes only. It is not intended to replace professional medical diagnosis or advice.

