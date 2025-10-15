from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)
CORS(app)

# =======================
# Load pre-trained XGBoost model and data (safe loads)
# =======================
model = None
X_test = None
y_test = None
dataset_df = None

# Load model (will raise if missing - keep as-is so user notices)
model = joblib.load("best_xgb_model.pkl")

# Try load X_test and y_test (for metrics & column alignment). If missing, keep None.
try:
    X_test = pd.read_csv("X_test.csv")
except Exception as e:
    print("Warning: X_test.csv not found or couldn't be read:", e)
    X_test = None

try:
    y_test = pd.read_csv("y_test.csv")
except Exception as e:
    print("Warning: y_test.csv not found or couldn't be read:", e)
    y_test = None

# Try load full dataset (TelcoChurn_Expanded.csv) to look up actual churns
try:
    # try common relative paths
    if os.path.exists("TelcoChurn_Expanded.csv"):
        dataset_df = pd.read_csv("TelcoChurn_Expanded.csv")
    elif os.path.exists("./data/TelcoChurn_Expanded.csv"):
        dataset_df = pd.read_csv("./data/TelcoChurn_Expanded.csv")
    else:
        # last resort attempt (may raise if not present)
        dataset_df = pd.read_csv("TelcoChurn_Expanded.csv")
    # Ensure numeric columns are numeric
    for c in ["tenure", "MonthlyCharges", "TotalCharges"]:
        if c in dataset_df.columns:
            dataset_df[c] = pd.to_numeric(dataset_df[c], errors="coerce")
    # Normalize churn values to consistent format (keep original but also add numeric)
    if "Churn" in dataset_df.columns:
        # dataset has Yes/No — we'll keep it and also provide numeric version
        dataset_df["Churn_numeric"] = dataset_df["Churn"].apply(lambda x: 1 if str(x).strip().lower() in ["yes","1","true"] else 0)
except Exception as e:
    print("Notice: TelcoChurn_Expanded.csv not loaded (it's optional) — dataset lookup disabled.", e)
    dataset_df = None


# Endpoint 1: Dashboard metrics
@app.route("/api/metrics", methods=["GET"])
def metrics():
    if X_test is None or y_test is None:
        return jsonify({"error": "X_test.csv or y_test.csv not available on server to compute metrics."}), 400

    from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
    try:
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        cm = confusion_matrix(y_test, y_pred).tolist()
        return jsonify({
            "accuracy": round(acc, 4),
            "auc": round(auc, 4),
            "confusion_matrix": cm,
            "classification_report": classification_report(y_test, y_pred, output_dict=True)
        })
    except Exception as e:
        return jsonify({"error": f"Failed to compute metrics: {str(e)}"}), 500


# Endpoint 2: SHAP values
@app.route("/api/shap", methods=["GET"])
def shap_values():
    if X_test is None:
        return jsonify({"error": "X_test.csv not available for SHAP calculations."}), 400
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_test)
        shap_summary = pd.DataFrame({
            'feature': X_test.columns,
            'importance': np.abs(shap_vals).mean(axis=0)
        }).sort_values(by='importance', ascending=False).to_dict(orient='records')
        return jsonify(shap_summary)
    except Exception as e:
        return jsonify({"error": f"SHAP calculation failed: {str(e)}"}), 500


# Endpoint 3: Predict single record (and optionally return dataset label if matched)
@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.json  # Expect JSON with feature values
        if data is None:
            return jsonify({"error": "No JSON body received"}), 400

        # Parse numeric features (the three you mentioned)
        def to_num(x):
            try:
                return float(x)
            except:
                return np.nan

        tenure = to_num(data.get("tenure", data.get("Tenure", None)))
        monthly = to_num(data.get("MonthlyCharges", data.get("MonthlyCharges", None)))
        total = to_num(data.get("TotalCharges", data.get("TotalCharges", None)))

        # Basic input validation
        if np.isnan(tenure) and np.isnan(monthly) and np.isnan(total):
            return jsonify({"error": "At least one of tenure/MonthlyCharges/TotalCharges must be provided and numeric."}), 400

        # 1) Try to find a matching row in dataset_df (if available)
        churn_from_dataset = None
        matched_row_info = None
        match_type = "none"
        match_distance = None

        if dataset_df is not None:
            # drop rows with missing numeric features
            candidates = dataset_df.dropna(subset=["tenure", "MonthlyCharges", "TotalCharges"]).copy()
            # tolerance for float equality
            eps_mc = 1e-2
            eps_tc = 1e-2

            # exact integer match for tenure + close for floats
            exact_mask = True
            if not np.isnan(tenure):
                exact_mask = exact_mask & (candidates["tenure"] == int(tenure))
            if not np.isnan(monthly):
                exact_mask = exact_mask & (candidates["MonthlyCharges"].sub(monthly).abs() <= eps_mc)
            if not np.isnan(total):
                exact_mask = exact_mask & (candidates["TotalCharges"].sub(total).abs() <= eps_tc)

            exact_matches = candidates[exact_mask]
            if len(exact_matches) > 0:
                # choose first exact match
                mr = exact_matches.iloc[0]
                churn_from_dataset = mr.get("Churn", None)
                matched_row_info = {
                    "tenure": float(mr["tenure"]),
                    "MonthlyCharges": float(mr["MonthlyCharges"]),
                    "TotalCharges": float(mr["TotalCharges"]),
                    "Churn": churn_from_dataset
                }
                match_type = "exact"
            else:
                # nearest neighbor based on normalized 3-feature distance
                feat_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
                X = candidates[feat_cols].values.astype(float)
                # compute min-max normalization to avoid scale issues
                mins = np.nanmin(X, axis=0)
                maxs = np.nanmax(X, axis=0)
                denom = (maxs - mins)
                denom[denom == 0] = 1.0
                X_scaled = (X - mins) / denom

                # build input vector (use provided values, fill missing with column mean)
                input_vec = np.array([
                    tenure if not np.isnan(tenure) else np.nanmean(candidates["tenure"]),
                    monthly if not np.isnan(monthly) else np.nanmean(candidates["MonthlyCharges"]),
                    total if not np.isnan(total) else np.nanmean(candidates["TotalCharges"])
                ], dtype=float)
                input_scaled = (input_vec - mins) / denom

                # compute distances
                dists = np.linalg.norm(X_scaled - input_scaled.reshape(1, -1), axis=1)
                idx = int(np.argmin(dists))
                mr = candidates.iloc[idx]
                churn_from_dataset = mr.get("Churn", None)
                matched_row_info = {
                    "tenure": float(mr["tenure"]),
                    "MonthlyCharges": float(mr["MonthlyCharges"]),
                    "TotalCharges": float(mr["TotalCharges"]),
                    "Churn": churn_from_dataset
                }
                match_type = "nearest"
                match_distance = float(dists[idx])

        # 2) Prepare model input same way as before (so your model still predicts)
        # Build a DataFrame from incoming JSON, then fill missing columns with X_test means (if X_test available)
        df_input = pd.DataFrame([data])

        # Convert any numeric-looking columns to numeric
        for k in ["tenure", "MonthlyCharges", "TotalCharges"]:
            if k in df_input.columns:
                df_input[k] = pd.to_numeric(df_input[k], errors="coerce")

        if X_test is not None:
            # Fill missing columns with X_test means (neutral values)
            for col in X_test.columns:
                if col not in df_input.columns:
                    df_input[col] = X_test[col].mean()
            # Reorder columns to match X_test ordering
            df_input = df_input[X_test.columns]
        else:
            # If we don't have X_test reference, keep what we have (model may fail if it expects more features)
            pass

        # Run model prediction (safe try)
        churn_model = None
        prob = None
        try:
            pred = model.predict(df_input)
            prob = float(model.predict_proba(df_input)[:, 1][0])
            churn_model = int(pred[0])
        except Exception as e:
            # Model prediction failed (likely because model expects different features); don't crash
            churn_model = None
            prob = None
            print("Model prediction failed for given input:", e)

        # Format response
        response = {
            "input": {
                "tenure": None if np.isnan(tenure) else (int(tenure) if float(tenure).is_integer() else float(tenure)),
                "MonthlyCharges": None if np.isnan(monthly) else float(monthly),
                "TotalCharges": None if np.isnan(total) else float(total),
            },
            "match_type": match_type,
            "matched_row": matched_row_info,
            "match_distance": match_distance,
            "churn_dataset": churn_from_dataset,   # e.g., "Yes" / "No" if available
            "churn_model": churn_model,            # e.g., 0/1 if available
            "probability": prob
        }

        return jsonify(response)
    except Exception as e:
        return jsonify({"error": f"Predict endpoint error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
