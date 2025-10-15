# =======================
# Install packages (Colab)
# =======================
!pip install -q seaborn shap xgboost lightgbm catboost imbalanced-learn optuna

# =======================
# Imports & setup
# =======================
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import shap
import optuna

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

from imblearn.combine import SMOTEENN
from xgboost import XGBClassifier

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (7, 5)

# =======================
# Upload data (Colab)
# =======================
from google.colab import files
uploaded = files.upload()
df = pd.read_csv(list(uploaded.keys())[0])
print("Loaded file:", list(uploaded.keys())[0], "shape:", df.shape)

# =======================
# Preprocessing
# =======================
if 'customerID' in df.columns:
    df = df.drop('customerID', axis=1)

if 'TotalCharges' in df.columns:
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

print("Missing before:", df.isnull().sum().sum())
df = df.dropna().reset_index(drop=True)
print("Missing after:", df.isnull().sum().sum())

# target encode
le = LabelEncoder()
df['Churn_enc'] = le.fit_transform(df['Churn'])
y = df['Churn_enc']
X_raw = df.drop(['Churn', 'Churn_enc'], axis=1)

# Feature engineering
X = X_raw.copy()
if {'MonthlyCharges','tenure','TotalCharges'}.issubset(X.columns):
    X['TotalRevenue'] = X['MonthlyCharges'] * X['tenure']
    X['AvgMonthlyRevenue'] = X['TotalCharges'] / (X['tenure'] + 1)

service_cols = [c for c in ['PhoneService','MultipleLines','OnlineSecurity','OnlineBackup',
                            'DeviceProtection','TechSupport','StreamingTV','StreamingMovies'] if c in X.columns]
if service_cols:
    X['ServiceCount'] = X[service_cols].apply(lambda row: (row == 'Yes').sum(), axis=1)

if 'SeniorCitizen' in X.columns and 'tenure' in X.columns:
    X['IsSeniorLowTenure'] = ((X['SeniorCitizen'] == 1) & (X['tenure'] < 12)).astype(int)

X = pd.get_dummies(X, drop_first=True)
X.columns = X.columns.str.replace(' ', '_')

print("Feature shape after FE & encoding:", X.shape)

# =======================
# Balance dataset
# =======================
smoteenn = SMOTEENN(random_state=42)
X_res, y_res = smoteenn.fit_resample(X, y)
print("After SMOTEENN:", X_res.shape, np.bincount(y_res))

X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
)
print("Train/Test sizes:", X_train.shape, X_test.shape)

# =======================
# Optuna for XGBoost only
# =======================
N_TRIALS = 30  # Increase for better tuning
CV_SPLITS = 3
skf = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=42)
scoring = "accuracy"

def objective_xgb(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1500),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.005, 0.3),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0),
        'scale_pos_weight': 1,
        'random_state': 42,
        'verbosity': 0,
        'use_label_encoder': False
    }
    model = XGBClassifier(**param, eval_metric='logloss', n_jobs=-1)
    scores = cross_val_score(model, X_train, y_train, cv=skf, scoring=scoring, n_jobs=-1)
    return scores.mean()

print("\n🔍 Hyperparameter tuning for XGBoost...")
study_xgb = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study_xgb.optimize(objective_xgb, n_trials=N_TRIALS, show_progress_bar=True)

print("\n✅ Best params for XGBoost:", study_xgb.best_params)

# =======================
# Train final best XGBoost
# =======================
best_xgb = XGBClassifier(**study_xgb.best_params,
                         use_label_encoder=False,
                         verbosity=0,
                         eval_metric='logloss',
                         random_state=42,
                         n_jobs=-1)
best_xgb.fit(X_train, y_train)

y_pred = best_xgb.predict(X_test)
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, best_xgb.predict_proba(X_test)[:,1])

print("\n📊 Final XGBoost Performance")
print(f"Test Accuracy: {acc:.4f} | Test AUC: {auc:.4f}")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("XGBoost Confusion Matrix")
plt.show()

# =======================
# SHAP for feature importance
# =======================
explainer = shap.TreeExplainer(best_xgb)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, plot_type="bar")
shap.summary_plot(shap_values, X_test)
