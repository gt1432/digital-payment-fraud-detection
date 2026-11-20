"""
Digital Payment Fraud Detection - Transaction Prediction & Explanation Module
Author: Senior ML Engineer & Data Scientist
"""
import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from src.preprocessing import prepare_single_transaction

MODELS_DIR = 'models'

def load_prediction_artifacts(model_name: str = 'best_model'):
    """
    Loads the scaler and selected model for live inference.
    """
    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
    if not os.path.exists(scaler_path):
        raise FileNotFoundError("Preprocessing scaler not found. Please run model training first.")
    scaler = joblib.load(scaler_path)
    
    filename_map = {
        'best_model': 'best_model.pkl',
        'random_forest': 'random_forest.pkl',
        'decision_tree': 'decision_tree.pkl',
        'logistic_regression': 'logistic_regression.pkl',
        'isolation_forest': 'isolation_forest.pkl'
    }
    target_file = filename_map.get(model_name.lower().replace(' ', '_'), 'best_model.pkl')
    model_path = os.path.join(MODELS_DIR, target_file)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{target_file}' not found.")
        
    model = joblib.load(model_path)
    return scaler, model

def analyze_suspicious_indicators(
    trans_type: str,
    amount: float,
    oldbalance_org: float,
    newbalance_orig: float,
    oldbalance_dest: float,
    newbalance_dest: float,
    fraud_prob: float
) -> List[Dict[str, str]]:
    """
    Evaluates domain transaction patterns to explain suspicious behavior.
    """
    indicators = []
    trans_type = trans_type.upper()
    
    # 1. Transaction Type Vulnerability
    if trans_type in ['TRANSFER', 'CASH_OUT']:
        indicators.append({
            'code': 'TYPE_HIGH_RISK',
            'severity': 'HIGH' if trans_type == 'TRANSFER' else 'MEDIUM',
            'title': f'High-Risk Transaction Type: {trans_type}',
            'description': 'Historical data confirms 100% of verified payment frauds occur exclusively in TRANSFER and CASH_OUT operations.'
        })
    else:
        indicators.append({
            'code': 'TYPE_LOW_RISK',
            'severity': 'LOW',
            'title': f'Low-Risk Type: {trans_type}',
            'description': 'Standard payment or internal transfer type with near-zero historical fraud incidence in the PaySim dataset.'
        })
        
    # 2. Account Draining (Sender balance zeroed out)
    if oldbalance_org > 0 and newbalance_orig == 0:
        indicators.append({
            'code': 'ACCOUNT_DRAINED',
            'severity': 'HIGH',
            'title': 'Complete Origin Account Liquidation',
            'description': f'Origin account was completely emptied from ${oldbalance_org:,.2f} to $0.00, a textbook fraud signature.'
        })
        
    # 3. Origin Balance Discrepancy
    orig_expected_new = max(0.0, oldbalance_org - amount)
    orig_diff = abs(newbalance_orig - orig_expected_new)
    if orig_diff > 1.0:
        indicators.append({
            'code': 'ORIGIN_BALANCE_MISMATCH',
            'severity': 'HIGH',
            'title': 'Origin Balance Calculation Discrepancy',
            'description': f'Mismatch of ${orig_diff:,.2f} between expected post-transaction balance (${orig_expected_new:,.2f}) and recorded balance (${newbalance_orig:,.2f}).'
        })
        
    # 4. Destination Account Empty Anomaly (Mule account behavior)
    if oldbalance_dest == 0 and amount > 50000:
        indicators.append({
            'code': 'MULE_DESTINATION',
            'severity': 'HIGH',
            'title': 'High-Value Transfer to Zero-Balance Account',
            'description': f'A substantial sum of ${amount:,.2f} was sent to an account with zero prior transaction balance, indicative of an ephemeral mule wallet.'
        })
        
    # 5. Destination Balance Discrepancy
    dest_expected_new = oldbalance_dest + amount
    dest_diff = abs(newbalance_dest - dest_expected_new)
    if dest_diff > 1.0 and trans_type in ['TRANSFER', 'CASH_OUT']:
        indicators.append({
            'code': 'DEST_BALANCE_MISMATCH',
            'severity': 'MEDIUM',
            'title': 'Destination Balance Inconsistency',
            'description': f'Recipient balance failed to increase by the transmitted funds (expected ${dest_expected_new:,.2f}, recorded ${newbalance_dest:,.2f}).'
        })
        
    # 6. High Proportion of Sender Net Balance
    if oldbalance_org > 0 and (amount / oldbalance_org) >= 0.95:
        indicators.append({
            'code': 'CAPITAL_EXHAUSTION',
            'severity': 'HIGH',
            'title': 'Near-Total Capital Transfer',
            'description': f'Transaction consumes {(amount / oldbalance_org * 100):.1f}% of sender total capital.'
        })
        
    # 7. Model Statistical Confidence Alert
    if fraud_prob >= 0.70:
        indicators.append({
            'code': 'STATISTICAL_FRAUD_SURGE',
            'severity': 'CRITICAL',
            'title': 'Elevated Statistical Fraud Vector',
            'description': f'Ensemble model probability ({fraud_prob * 100:.1f}%) exceeds safety thresholds.'
        })
        
    return indicators

def predict_transaction(
    trans_type: str,
    amount: float,
    oldbalance_org: float,
    newbalance_orig: float,
    oldbalance_dest: float,
    newbalance_dest: float,
    step: int = 1,
    model_name: str = 'best_model'
) -> Dict[str, Any]:
    """
    Executes transaction inference and generates detailed risk analysis.
    """
    scaler, model = load_prediction_artifacts(model_name)
    
    # Feature extraction & alignment
    X_single = prepare_single_transaction(
        trans_type=trans_type,
        amount=amount,
        oldbalance_org=oldbalance_org,
        newbalance_orig=newbalance_orig,
        oldbalance_dest=oldbalance_dest,
        newbalance_dest=newbalance_dest,
        step=step
    )
    
    X_scaled = scaler.transform(X_single)
    
    # Model inference
    is_anomaly_detector = 'isolation' in str(type(model)).lower()
    
    if is_anomaly_detector:
        raw_pred = model.predict(X_scaled)[0]
        is_fraud = int(raw_pred == -1)
        score = -float(model.decision_function(X_scaled)[0])
        # Approximate sigmoid mapping for score to [0, 1] probability
        fraud_prob = float(1.0 / (1.0 + np.exp(-score * 8.0)))
    else:
        pred_label = model.predict(X_scaled)[0]
        is_fraud = int(pred_label)
        if hasattr(model, 'predict_proba'):
            fraud_prob = float(model.predict_proba(X_scaled)[0][1])
        else:
            fraud_prob = 1.0 if is_fraud == 1 else 0.0
            
    # Risk categorization
    if fraud_prob < 0.30:
        risk_level = 'LOW'
        status_label = 'GENUINE'
        color = '#30D158' # SpaceX clean green
    elif fraud_prob < 0.70:
        risk_level = 'MEDIUM'
        status_label = 'SUSPICIOUS / ELEVATED RISK'
        color = '#FF9F0A' # SpaceX amber warning
    else:
        risk_level = 'HIGH'
        status_label = 'FRAUDULENT'
        color = '#FF453A' # SpaceX critical red
        
    indicators = analyze_suspicious_indicators(
        trans_type=trans_type,
        amount=amount,
        oldbalance_org=oldbalance_org,
        newbalance_orig=newbalance_orig,
        oldbalance_dest=oldbalance_dest,
        newbalance_dest=newbalance_dest,
        fraud_prob=fraud_prob
    )
    
    return {
        'prediction': 'Fraudulent' if is_fraud == 1 else 'Genuine',
        'is_fraud': is_fraud,
        'fraud_probability': fraud_prob,
        'risk_level': risk_level,
        'status_label': status_label,
        'status_color': color,
        'indicators': indicators,
        'features_calculated': {
            'errorBalanceOrig': float(X_single['errorBalanceOrig'].iloc[0]),
            'errorBalanceDest': float(X_single['errorBalanceDest'].iloc[0]),
            'balanceRatioOrig': float(X_single['balanceRatioOrig'].iloc[0]),
            'balanceRatioDest': float(X_single['balanceRatioDest'].iloc[0]),
            'isZeroNewBalanceOrig': int(X_single['isZeroNewBalanceOrig'].iloc[0]),
            'isZeroOldBalanceDest': int(X_single['isZeroOldBalanceDest'].iloc[0])
        },
        'disclaimer': 'MODEL INFERENCE (PROBABILISTIC) - NOT A CONFIRMED LEGAL DETERMINATION'
    }
