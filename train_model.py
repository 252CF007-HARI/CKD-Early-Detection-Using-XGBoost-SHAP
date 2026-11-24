"""
RenalCare AI
XGBoost CKD Prediction Model Training

Pipeline:

CKD Dataset
    ↓
Preprocessing
    ↓
Train/Test Split
    ↓
PCA-Based Feature Analysis
    ↓
12 Original Feature Selection
    ↓
XGBoost Training
    ↓
Evaluation
    ↓
Model Saving
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from xgboost import XGBClassifier


# =========================================================
# PROJECT PATH
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

UTILS_DIR = os.path.join(
    BASE_DIR,
    "utils"
)

if UTILS_DIR not in sys.path:
    sys.path.insert(
        0,
        UTILS_DIR
    )


from preprocess import preprocess_dataset

from feature_selection import (
    select_features_using_pca
)


# =========================================================
# MODEL DIRECTORY
# =========================================================

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# =========================================================
# MAIN TRAINING FUNCTION
# =========================================================

def train_xgboost_model():

    print("\n")
    print("=" * 70)
    print("RENALCARE AI - XGBOOST MODEL TRAINING")
    print("=" * 70)

    # =====================================================
    # 1. PREPROCESS DATASET
    # =====================================================

    X, y, available_features = preprocess_dataset()

    target_column = "class"

    # Current preprocessing function performs
    # categorical conversion internally.
    # Therefore, no separate encoders are required here.
    feature_encoders = {}

    target_encoder = None

    print(
        f"\nComplete feature count: "
        f"{X.shape[1]}"
    )

    # =====================================================
    # 2. TRAIN TEST SPLIT
    # =====================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(
        f"\nTraining records: "
        f"{len(X_train)}"
    )

    print(
        f"Testing records: "
        f"{len(X_test)}"
    )

    # =====================================================
    # 3. PCA-BASED FEATURE ANALYSIS
    # =====================================================

    (
        selected_features,
        importance_df,
        pca_information
    ) = select_features_using_pca(
        X_train,
        number_of_features=12
    )

    # =====================================================
    # 4. SELECT 12 ORIGINAL FEATURES
    # =====================================================

    X_train_selected = X_train[
        selected_features
    ]

    X_test_selected = X_test[
        selected_features
    ]

    print(
        "\nSelected feature count:",
        len(selected_features)
    )

    print(
        "\nSelected features:"
    )

    for feature in selected_features:
        print(
            f" - {feature}"
        )

    # =====================================================
    # 5. HANDLE CLASS IMBALANCE
    # =====================================================

    class_counts = np.bincount(
        y_train
    )

    if len(class_counts) >= 2:

        negative_count = class_counts[0]

        positive_count = class_counts[1]

        if positive_count > 0:

            scale_pos_weight = (
                negative_count
                /
                positive_count
            )

        else:

            scale_pos_weight = 1.0

    else:

        scale_pos_weight = 1.0

    print(
        "\nScale positive weight:",
        round(scale_pos_weight, 4)
    )

    # =====================================================
    # 6. INITIALIZE XGBOOST
    # =====================================================

    model = XGBClassifier(

        n_estimators=300,

        max_depth=4,

        learning_rate=0.05,

        subsample=0.90,

        colsample_bytree=0.90,

        min_child_weight=2,

        gamma=0.0,

        reg_alpha=0.05,

        reg_lambda=1.0,

        objective="binary:logistic",

        eval_metric="logloss",

        random_state=42,

        n_jobs=-1,

        scale_pos_weight=scale_pos_weight

    )

    # =====================================================
    # 7. TRAIN MODEL
    # =====================================================

    print(
        "\nTraining XGBoost..."
    )

    model.fit(
        X_train_selected,
        y_train
    )

    print(
        "XGBoost training completed."
    )

    # =====================================================
    # 8. PREDICTION
    # =====================================================

    y_pred = model.predict(
        X_test_selected
    )

    y_probability = (
        model.predict_proba(
            X_test_selected
        )[:, 1]
    )

    # =====================================================
    # 9. MODEL EVALUATION
    # =====================================================

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    try:

        roc_auc = roc_auc_score(
            y_test,
            y_probability
        )

    except ValueError:

        roc_auc = 0.0

    print("\n")
    print("=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)

    print(
        f"\nAccuracy  : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Precision : "
        f"{precision * 100:.2f}%"
    )

    print(
        f"Recall    : "
        f"{recall * 100:.2f}%"
    )

    print(
        f"F1-Score  : "
        f"{f1 * 100:.2f}%"
    )

    print(
        f"ROC-AUC   : "
        f"{roc_auc:.4f}"
    )

    # =====================================================
    # CLASSIFICATION REPORT
    # =====================================================

    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    # =====================================================
    # CONFUSION MATRIX
    # =====================================================

    print(
        "Confusion Matrix:"
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print(
        cm
    )

    # =====================================================
    # 10. SAVE MODEL
    # =====================================================

    model_path = os.path.join(
        MODEL_DIR,
        "xgboost_model.pkl"
    )

    joblib.dump(
        model,
        model_path
    )

    print(
        "\nXGBoost model saved:"
    )

    print(
        model_path
    )

    # =====================================================
    # 11. SAVE SELECTED FEATURES
    # =====================================================

    selected_features_path = os.path.join(
        MODEL_DIR,
        "selected_features.pkl"
    )

    joblib.dump(
        selected_features,
        selected_features_path
    )

    # =====================================================
    # 12. SAVE PREPROCESSING INFORMATION
    # =====================================================

    preprocessing_information = {

        "feature_encoders":
            feature_encoders,

        "target_encoder":
            target_encoder,

        "target_column":
            target_column,

        "all_features":
            list(X.columns),

        "selected_features":
            selected_features
    }

    preprocessing_path = os.path.join(
        MODEL_DIR,
        "preprocessor.pkl"
    )

    joblib.dump(
        preprocessing_information,
        preprocessing_path
    )

    # =====================================================
    # 13. CREATE FEATURE METADATA
    # =====================================================

    feature_metadata = []

    for feature in selected_features:

        if feature in feature_encoders:

            encoder = feature_encoders[
                feature
            ]

            options = [
                str(value)
                for value in encoder.classes_
            ]

            feature_type = "categorical"

        else:

            options = []

            feature_type = "numeric"

        feature_metadata.append({

            "name":
                feature,

            "label":
                feature.replace(
                    "_",
                    " "
                ).title(),

            "type":
                feature_type,

            "options":
                options

        })

    metadata_path = os.path.join(
        MODEL_DIR,
        "feature_metadata.pkl"
    )

    joblib.dump(
        feature_metadata,
        metadata_path
    )

    # =====================================================
    # 14. SAVE MODEL METRICS
    # =====================================================

    metrics = {

        "accuracy":
            float(accuracy),

        "precision":
            float(precision),

        "recall":
            float(recall),

        "f1_score":
            float(f1),

        "roc_auc":
            float(roc_auc),

        "training_records":
            int(len(X_train)),

        "testing_records":
            int(len(X_test)),

        "total_features":
            int(X.shape[1]),

        "selected_features_count":
            int(len(selected_features)),

        "selected_features":
            selected_features

    }

    metrics_path = os.path.join(
        MODEL_DIR,
        "model_metrics.pkl"
    )

    joblib.dump(
        metrics,
        metrics_path
    )

    # =====================================================
    # SAVE PCA INFORMATION
    # =====================================================

    pca_path = os.path.join(
        MODEL_DIR,
        "pca_analysis.pkl"
    )

    joblib.dump(
        pca_information,
        pca_path
    )

    # =====================================================
    # SAVE PCA FEATURE IMPORTANCE
    # =====================================================

    try:

        pca_importance_path = os.path.join(
            BASE_DIR,
            "dataset",
            "pca_feature_importance.csv"
        )

        importance_df.to_csv(
            pca_importance_path,
            index=False
        )

        print(
            "\nPCA feature importance saved:"
        )

        print(
            pca_importance_path
        )

    except Exception as error:

        print(
            "\nWarning: Could not save "
            "PCA feature importance."
        )

        print(
            str(error)
        )

    # =====================================================
    # FINAL
    # =====================================================

    print("\n")
    print("=" * 70)
    print("MODEL FILES SAVED")
    print("=" * 70)

    print(
        "\nmodels/xgboost_model.pkl"
    )

    print(
        "models/selected_features.pkl"
    )

    print(
        "models/preprocessor.pkl"
    )

    print(
        "models/feature_metadata.pkl"
    )

    print(
        "models/model_metrics.pkl"
    )

    print(
        "models/pca_analysis.pkl"
    )

    print(
        "\nTraining completed successfully."
    )

    print("=" * 70)

    return model, metrics


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    train_xgboost_model()