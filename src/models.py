"""
Digital Payment Fraud Detection - Model Training and Evaluation Module
Author: Senior ML Engineer & Data Scientist
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)
from imblearn.over_sampling import SMOTE
from src.preprocessing import prepare_data, FEATURE_COLUMNS, load_dataset

MODELS_DIR = 'models'

def train_and_evaluate_all(df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Trains and compares:
    1. Logistic Regression
    2. Decision Tree
    3. Random Forest
    4. Isolation Forest (Anomaly Detection)
    
    Evaluates:
    - Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix
    Handles imbalance with SMOTE strictly on training split to eliminate data leakage.
    Saves models and comprehensive evaluation metrics.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    if df is None:
        df = load_dataset()
        
    X, y = prepare_data(df)
    if y is None:
        raise ValueError("Dataset does not contain 'isFraud' target column.")
        
    # Split without data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
    
    # SMOTE on training split only
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
    
    metrics: Dict[str, Any] = {}
    roc_data: Dict[str, Any] = {}
    
    # ---------------- 1. Logistic Regression ----------------
    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    lr.fit(X_train_smote, y_train_smote)
    y_pred_lr = lr.predict(X_test_scaled)
    y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]
    
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
    # Downsample ROC curve points for efficient visualization storage
    step_idx = max(1, len(fpr_lr) // 100)
    roc_data['Logistic Regression'] = {
        'fpr': fpr_lr[::step_idx].tolist() + [1.0],
        'tpr': tpr_lr[::step_idx].tolist() + [1.0]
    }
    
    metrics['Logistic Regression'] = {
        'accuracy': float(accuracy_score(y_test, y_pred_lr)),
        'precision': float(precision_score(y_test, y_pred_lr, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred_lr, zero_division=0)),
        'f1_score': float(f1_score(y_test, y_pred_lr, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test, y_prob_lr)),
        'confusion_matrix': confusion_matrix(y_test, y_pred_lr).tolist(),
        'type': 'Supervised'
    }
    joblib.dump(lr, os.path.join(MODELS_DIR, 'logistic_regression.pkl'))
    
    # ---------------- 2. Decision Tree ----------------
    dt = DecisionTreeClassifier(max_depth=10, min_samples_split=10, random_state=42)
    dt.fit(X_train_smote, y_train_smote)
    y_pred_dt = dt.predict(X_test_scaled)
    y_prob_dt = dt.predict_proba(X_test_scaled)[:, 1]
    
    fpr_dt, tpr_dt, _ = roc_curve(y_test, y_prob_dt)
    step_idx = max(1, len(fpr_dt) // 100)
    roc_data['Decision Tree'] = {
        'fpr': fpr_dt[::step_idx].tolist() + [1.0],
        'tpr': tpr_dt[::step_idx].tolist() + [1.0]
    }
    
    metrics['Decision Tree'] = {
        'accuracy': float(accuracy_score(y_test, y_pred_dt)),
        'precision': float(precision_score(y_test, y_pred_dt, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred_dt, zero_division=0)),
        'f1_score': float(f1_score(y_test, y_pred_dt, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test, y_prob_dt)),
        'confusion_matrix': confusion_matrix(y_test, y_pred_dt).tolist(),
        'type': 'Supervised'
    }
    joblib.dump(dt, os.path.join(MODELS_DIR, 'decision_tree.pkl'))
    
    # ---------------- 3. Random Forest ----------------
    rf = RandomForestClassifier(n_estimators=100, max_depth=14, min_samples_split=8, random_state=42, n_jobs=-1)
    rf.fit(X_train_smote, y_train_smote)
    y_pred_rf = rf.predict(X_test_scaled)
    y_prob_rf = rf.predict_proba(X_test_scaled)[:, 1]
    
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
    step_idx = max(1, len(fpr_rf) // 100)
    roc_data['Random Forest'] = {
        'fpr': fpr_rf[::step_idx].tolist() + [1.0],
        'tpr': tpr_rf[::step_idx].tolist() + [1.0]
    }
    
    metrics['Random Forest'] = {
        'accuracy': float(accuracy_score(y_test, y_pred_rf)),
        'precision': float(precision_score(y_test, y_pred_rf, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred_rf, zero_division=0)),
        'f1_score': float(f1_score(y_test, y_pred_rf, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test, y_prob_rf)),
        'confusion_matrix': confusion_matrix(y_test, y_pred_rf).tolist(),
        'type': 'Supervised'
    }
    joblib.dump(rf, os.path.join(MODELS_DIR, 'random_forest.pkl'))
    
    # ---------------- 4. Isolation Forest (Anomaly Detection) ----------------
    # Unsupervised: trained on un-oversampled training features
    empirical_contamination = float(np.clip(y_train.mean(), 0.01, 0.15))
    iso = IsolationForest(
        n_estimators=100,
        contamination=empirical_contamination,
        random_state=42,
        n_jobs=-1
    )
    iso.fit(X_train_scaled)
    
    # IsolationForest outputs: -1 for outlier/anomaly (fraud), 1 for inlier (genuine)
    iso_raw_pred = iso.predict(X_test_scaled)
    y_pred_iso = np.where(iso_raw_pred == -1, 1, 0)
    
    # Anomaly scores: negative of decision function (higher = more anomalous)
    iso_scores = -iso.decision_function(X_test_scaled)
    # Normalize score between 0 and 1 for AUC
    min_s, max_s = iso_scores.min(), iso_scores.max()
    iso_prob = (iso_scores - min_s) / (max_s - min_s + 1e-9)
    
    fpr_iso, tpr_iso, _ = roc_curve(y_test, iso_prob)
    step_idx = max(1, len(fpr_iso) // 100)
    roc_data['Isolation Forest'] = {
        'fpr': fpr_iso[::step_idx].tolist() + [1.0],
        'tpr': tpr_iso[::step_idx].tolist() + [1.0]
    }
    
    metrics['Isolation Forest'] = {
        'accuracy': float(accuracy_score(y_test, y_pred_iso)),
        'precision': float(precision_score(y_test, y_pred_iso, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred_iso, zero_division=0)),
        'f1_score': float(f1_score(y_test, y_pred_iso, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test, iso_prob)),
        'confusion_matrix': confusion_matrix(y_test, y_pred_iso).tolist(),
        'type': 'Unsupervised Anomaly Detection'
    }
    joblib.dump(iso, os.path.join(MODELS_DIR, 'isolation_forest.pkl'))
    
    # ---------------- Select Best Supervised Model ----------------
    # High F1-score & Recall are paramount for fraud detection
    supervised_candidates = ['Logistic Regression', 'Decision Tree', 'Random Forest']
    best_model_name = max(
        supervised_candidates,
        key=lambda m: (metrics[m]['f1_score'] * 0.7 + metrics[m]['recall'] * 0.3)
    )
    
    # Save best model copy
    if best_model_name == 'Random Forest':
        joblib.dump(rf, os.path.join(MODELS_DIR, 'best_model.pkl'))
    elif best_model_name == 'Decision Tree':
        joblib.dump(dt, os.path.join(MODELS_DIR, 'best_model.pkl'))
    else:
        joblib.dump(lr, os.path.join(MODELS_DIR, 'best_model.pkl'))
        
    full_metrics_payload = {
        'models': metrics,
        'roc_curves': roc_data,
        'best_model': best_model_name,
        'best_metrics': metrics[best_model_name],
        'test_set_size': int(len(y_test)),
        'test_fraud_count': int(y_test.sum()),
        'test_genuine_count': int(len(y_test) - y_test.sum())
    }
    
    with open(os.path.join(MODELS_DIR, 'model_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(full_metrics_payload, f, indent=2)
        
    with open(os.path.join(MODELS_DIR, 'feature_columns.json'), 'w', encoding='utf-8') as f:
        json.dump(FEATURE_COLUMNS, f, indent=2)
        
    return full_metrics_payload

def load_trained_models() -> Dict[str, Any]:
    """
    Loads all saved models, scaler, and evaluation metrics.
    """
    artifacts: Dict[str, Any] = {}
    
    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
    if os.path.exists(scaler_path):
        artifacts['scaler'] = joblib.load(scaler_path)
        
    best_path = os.path.join(MODELS_DIR, 'best_model.pkl')
    if os.path.exists(best_path):
        artifacts['best_model'] = joblib.load(best_path)
        
    for name in ['logistic_regression', 'decision_tree', 'random_forest', 'isolation_forest']:
        p = os.path.join(MODELS_DIR, f'{name}.pkl')
        if os.path.exists(p):
            artifacts[name] = joblib.load(p)
            
    metrics_path = os.path.join(MODELS_DIR, 'model_metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r', encoding='utf-8') as f:
            artifacts['metrics'] = json.load(f)
            
    return artifacts
