"""
RenalCare AI
Machine Learning Approach for Early Detection of Chronic Kidney Disease
"""

import os
import sys
import traceback
from datetime import datetime

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    send_file,
    flash
)


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

UTILS_DIR = os.path.join(
    BASE_DIR,
    "utils"
)

if UTILS_DIR not in sys.path:
    sys.path.insert(0, UTILS_DIR)


# ============================================================
# IMPORT PROJECT MODULES
# ============================================================

from predict import (
    predict_patient,
    load_selected_features
)

from shap_analysis import (
    get_top_shap_features
)

from pdf_report import (
    generate_pdf_report
)


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

app.secret_key = (
    "renalcare-ai-secure-session-key-2026"
)


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# ABOUT PAGE
# ============================================================

@app.route("/about")
def about():

    return render_template(
        "about.html"
    )


# ============================================================
# CONTACT PAGE
# ============================================================

@app.route("/contact")
def contact():

    return render_template(
        "contact.html"
    )


# ============================================================
# PREDICTION PAGE
# ============================================================

@app.route(
    "/predict",
    methods=["GET", "POST"]
)
def predict_page():

    # --------------------------------------------------------
    # GET REQUEST
    # Display prediction form
    # --------------------------------------------------------

    if request.method == "GET":

        try:

            import joblib

            feature_metadata_path = os.path.join(
                BASE_DIR,
                "models",
                "feature_metadata.pkl"
            )

            feature_metadata = joblib.load(
                feature_metadata_path
            )

        except Exception:

            feature_metadata = []

        return render_template(
            "predict.html",
            feature_metadata=feature_metadata
        )


    # --------------------------------------------------------
    # POST REQUEST
    # Process patient prediction
    # --------------------------------------------------------

    try:

        print("\n")
        print("=" * 70)
        print("RENALCARE AI - PREDICTION")
        print("=" * 70)


        # ----------------------------------------------------
        # LOAD SELECTED FEATURES
        # ----------------------------------------------------

        selected_features = (
            load_selected_features()
        )

        print("\nSelected Features:")
        print(selected_features)


        # ----------------------------------------------------
        # COLLECT FORM DATA
        # ----------------------------------------------------

        patient_data = {}

        for feature in selected_features:

            value = request.form.get(
                feature,
                ""
            )

            patient_data[feature] = value


        print("\nPatient Input:")
        print(patient_data)


        # ----------------------------------------------------
        # XGBOOST PREDICTION
        # ----------------------------------------------------

        print("\nRunning XGBoost prediction...")


        result = predict_patient(
            patient_data
        )


        print("\nPrediction completed.")

        print(
            "Prediction:",
            result["prediction"]
        )

        print(
            "Confidence:",
            result["confidence"]
        )

        print(
            "CKD Probability:",
            result["ckd_probability"]
        )

        print(
            "Not CKD Probability:",
            result["not_ckd_probability"]
        )

        print(
            "Risk:",
            result["risk"]
        )


        # ----------------------------------------------------
        # SHAP EXPLANATION
        # ----------------------------------------------------

        print(
            "\nRunning SHAP explanation..."
        )


        shap_features = (
            get_top_shap_features(
                result["input_data"],
                top_n=6
            )
        )


        print(
            "\nSHAP explanation completed."
        )

        print(
            "SHAP Features:",
            shap_features
        )


        # ----------------------------------------------------
        # CONVERT TO PERCENTAGES
        # ----------------------------------------------------

        confidence_percentage = (
            result["confidence"] * 100
        )

        probability_percentage = (
            result["ckd_probability"] * 100
        )


        # ----------------------------------------------------
        # CLEAN INPUT DATA FOR SESSION
        # ----------------------------------------------------

        clean_input_data = {}


        for key, value in patient_data.items():

            try:

                clean_input_data[key] = float(
                    value
                )

            except (
                ValueError,
                TypeError
            ):

                clean_input_data[key] = str(
                    value
                )


        # ----------------------------------------------------
        # CREATE REPORT DATA
        # ----------------------------------------------------

        report_data = {

            "prediction":
                result["prediction"],

            "confidence":
                confidence_percentage,

            "probability":
                probability_percentage,

            "risk":
                result["risk"],

            "input_data":
                clean_input_data,

            "shap_features":
                shap_features,

            "created_at":
                datetime.now().strftime(
                    "%d %B %Y, %I:%M %p"
                )
        }


        # ----------------------------------------------------
        # SAVE REPORT DATA
        # ----------------------------------------------------

        session[
            "latest_report"
        ] = report_data


        print("\n")
        print("=" * 70)
        print("PREDICTION SUCCESSFUL")
        print("=" * 70)
        print("\n")


        # ----------------------------------------------------
        # DISPLAY RESULT PAGE
        # ----------------------------------------------------

        return render_template(

            "result.html",

            prediction=
                result["prediction"],

            confidence=
                f"{confidence_percentage:.2f}",

            probability=
                f"{probability_percentage:.2f}",

            risk=
                result["risk"],

            shap_features=[

                (
                    item["feature"],
                    f"{item['shap_value']:+.4f}"
                )

                for item in shap_features

            ]

        )


    # ========================================================
    # PREDICTION ERROR
    # ========================================================

    except Exception as error:

        print("\n")
        print("=" * 70)
        print("PREDICTION ERROR")
        print("=" * 70)

        print("\nError:")
        print(
            str(error)
        )

        print(
            "\nComplete traceback:"
        )

        traceback.print_exc()

        print("=" * 70)
        print("\n")


        return render_template(

            "error.html",

            error_message=
                str(error)

        ), 500


# ============================================================
# REPORT PAGE
# ============================================================

@app.route("/report")
def report():

    report_data = session.get(
        "latest_report"
    )


    return render_template(

        "report.html",

        report_data=
            report_data

    )


# ============================================================
# DOWNLOAD PDF REPORT
# ============================================================

@app.route("/download-report")
def download_report():

    report_data = session.get(
        "latest_report"
    )


    # --------------------------------------------------------
    # CHECK REPORT
    # --------------------------------------------------------

    if not report_data:

        flash(
            "Please complete a prediction before downloading the report.",
            "warning"
        )

        return redirect(
            url_for("predict_page")
        )


    # --------------------------------------------------------
    # CREATE REPORT DIRECTORY
    # --------------------------------------------------------

    report_directory = os.path.join(
        BASE_DIR,
        "reports"
    )

    os.makedirs(
        report_directory,
        exist_ok=True
    )


    # --------------------------------------------------------
    # PDF FILE
    # --------------------------------------------------------

    filename = (
        "RenalCare_AI_Report.pdf"
    )

    output_path = os.path.join(
        report_directory,
        filename
    )


    # --------------------------------------------------------
    # GENERATE PDF
    # --------------------------------------------------------

    generate_pdf_report(

        report_data,

        output_path

    )


    # --------------------------------------------------------
    # SEND PDF TO USER
    # --------------------------------------------------------

    return send_file(

        output_path,

        as_attachment=True,

        download_name=filename,

        mimetype="application/pdf"

    )


# ============================================================
# 404 ERROR
# ============================================================

@app.errorhandler(404)
def page_not_found(error):

    return render_template(

        "error.html",

        error_message=
            "The requested page could not be found."

    ), 404


# ============================================================
# 500 ERROR
# ============================================================

@app.errorhandler(500)
def internal_server_error(error):

    return render_template(

        "error.html",

        error_message=
            "An internal server error occurred."

    ), 500


# ============================================================
# START FLASK APPLICATION
# ============================================================

if __name__ == "__main__":

    print("\n")

    print("=" * 70)
    print("RENALCARE AI")
    print("CKD Prediction System")
    print("=" * 70)

    print("\nApplication running at:")

    print(
        "http://127.0.0.1:5000"
    )

    print(
        "\nPress CTRL+C to stop."
    )

    print("=" * 70)

    print("\n")


    app.run(

        host="127.0.0.1",

        port=5000,

        debug=False

    )