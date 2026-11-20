"""
Digital Payment Fraud Detection - Preprocessing Module
Author: Senior ML Engineer & Data Scientist
"""
import os
import json
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any

TRANSACTION_TYPES = ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']

FEATURE_COLUMNS = [
    'amount',
    'oldbalanceOrg',
    'newbalanceOrig',
    'oldbalanceDest',
    'newbalanceDest',
    'errorBalanceOrig',
    'errorBalanceDest',
    'balanceRatioOrig',
    'balanceRatioDest',
    'isZeroNewBalanceOrig',
    'isZeroOldBalanceDest',
    'hour_of_day',
    'type_CASH_IN',
    'type_CASH_OUT',
    'type_DEBIT',
    'type_PAYMENT',
    'type_TRANSFER'
]

def load_dataset(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Load the PaySim dataset. Automatically locates sample or full file.
    """
    possible_paths = [
        filepath,
        'data/paysim_sample.csv',
        'data/PS_20174392719_1491204439457_log.csv',
        'paysim_sample.csv',
        'PS_20174392719_1491204439457_log.csv'
    ]
    
    target_path = None
    for p in possible_paths:
        if p and os.path.exists(p):
            target_path = p
            break
            
    if not target_path:
        raise FileNotFoundError(
            "PaySim dataset not found. Please ensure data/paysim_sample.csv or data/PS_20174392719_1491204439457_log.csv exists."
        )
        
    return pd.read_csv(target_path)

def load_dataset_summary() -> Optional[Dict[str, Any]]:
    """
    Load precomputed full dataset summary if available.
    """
    summary_path = 'data/dataset_summary.json'
    if os.path.exists(summary_path):
        with open(summary_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def validate_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validates dataset structure, datatypes, missing values, and duplicates.
    """
    required_cols = ['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
    missing_cols = [c for c in required_cols if c not in df.columns]
    
    null_counts = {c: int(df[c].isnull().sum()) for c in required_cols if c in df.columns}
    total_nulls = sum(null_counts.values())
    duplicates = int(df.duplicated(subset=['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig']).sum())
    
    return {
        'is_valid': len(missing_cols) == 0,
        'missing_columns': missing_cols,
        'total_rows': int(len(df)),
        'total_columns': int(len(df.columns)),
        'null_counts': null_counts,
        'total_nulls': total_nulls,
        'duplicate_records': duplicates,
        'dtypes': {c: str(t) for c, t in df.dtypes.items()}
    }

def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers domain fraud detection features:
    - errorBalanceOrig: discrepancy between sender balance change and amount
    - errorBalanceDest: discrepancy between receiver balance change and amount
    - balanceRatioOrig: amount / (oldbalanceOrg + 1)
    - balanceRatioDest: amount / (oldbalanceDest + 1)
    - isZeroNewBalanceOrig: account emptied flag
    - isZeroOldBalanceDest: empty receiver account flag
    - hour_of_day: step mod 24 temporal feature
    - One-hot transaction types
    """
    df_feat = data.copy()
    
    # 1. Balance discrepancy indicators
    df_feat['errorBalanceOrig'] = df_feat['oldbalanceOrg'] - df_feat['amount'] - df_feat['newbalanceOrig']
    df_feat['errorBalanceDest'] = df_feat['oldbalanceDest'] + df_feat['amount'] - df_feat['newbalanceDest']
    
    # 2. Balance ratios
    df_feat['balanceRatioOrig'] = df_feat['amount'] / (df_feat['oldbalanceOrg'] + 1.0)
    df_feat['balanceRatioDest'] = df_feat['amount'] / (df_feat['oldbalanceDest'] + 1.0)
    
    # 3. Behavioral flags
    df_feat['isZeroNewBalanceOrig'] = (df_feat['newbalanceOrig'] == 0).astype(int)
    df_feat['isZeroOldBalanceDest'] = (df_feat['oldbalanceDest'] == 0).astype(int)
    
    # 4. Temporal hour of day
    if 'step' in df_feat.columns:
        df_feat['hour_of_day'] = df_feat['step'] % 24
    else:
        df_feat['hour_of_day'] = 12
        
    # 5. One-hot encoded transaction types
    for t in TRANSACTION_TYPES:
        col = f'type_{t}'
        if 'type' in df_feat.columns:
            df_feat[col] = (df_feat['type'] == t).astype(int)
        else:
            df_feat[col] = 0
            
    return df_feat[FEATURE_COLUMNS]

def prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
    """
    Extracts features and target variable isFraud.
    Excludes isFraud and isFlaggedFraud from model input features.
    """
    X = engineer_features(df)
    y = df['isFraud'].values if 'isFraud' in df.columns else None
    return X, y

def prepare_single_transaction(
    trans_type: str,
    amount: float,
    oldbalance_org: float,
    newbalance_orig: float,
    oldbalance_dest: float,
    newbalance_dest: float,
    step: int = 1
) -> pd.DataFrame:
    """
    Constructs a feature row for a single transaction query.
    """
    row = pd.DataFrame([{
        'step': int(step),
        'type': str(trans_type).upper(),
        'amount': float(amount),
        'oldbalanceOrg': float(oldbalance_org),
        'newbalanceOrig': float(newbalance_orig),
        'oldbalanceDest': float(oldbalance_dest),
        'newbalanceDest': float(newbalance_dest)
    }])
    return engineer_features(row)
