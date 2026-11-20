"""
Unit and Integration Tests for Digital Payment Fraud Detection System
"""
import pytest
import os
import pandas as pd
import numpy as np

from src.preprocessing import (
    load_dataset,
    validate_dataset,
    engineer_features,
    prepare_single_transaction,
    FEATURE_COLUMNS
)
from src.predict import predict_transaction
from src.models import load_trained_models

def test_dataset_loading_and_validation():
    df = load_dataset()
    assert df is not None
    assert len(df) > 0
    report = validate_dataset(df)
    assert report['is_valid'] is True
    assert report['total_nulls'] == 0

def test_feature_engineering_columns():
    df = load_dataset()
    X = engineer_features(df.head(10))
    for col in FEATURE_COLUMNS:
        assert col in X.columns
    assert len(X.columns) == len(FEATURE_COLUMNS)

def test_feature_engineering_math():
    sample = pd.DataFrame([{
        'step': 1,
        'type': 'TRANSFER',
        'amount': 1000.0,
        'oldbalanceOrg': 1000.0,
        'newbalanceOrig': 0.0,
        'oldbalanceDest': 0.0,
        'newbalanceDest': 0.0
    }])
    X = engineer_features(sample)
    # errorBalanceOrig: 1000 - 1000 - 0 = 0
    assert X['errorBalanceOrig'].iloc[0] == 0.0
    # errorBalanceDest: 0 + 1000 - 0 = 1000
    assert X['errorBalanceDest'].iloc[0] == 1000.0
    # isZeroNewBalanceOrig: 1
    assert X['isZeroNewBalanceOrig'].iloc[0] == 1
    # isZeroOldBalanceDest: 1
    assert X['isZeroOldBalanceDest'].iloc[0] == 1

def test_model_loading_and_metrics():
    artifacts = load_trained_models()
    assert 'best_model' in artifacts
    assert 'scaler' in artifacts
    assert 'metrics' in artifacts
    metrics = artifacts['metrics']
    assert 'Random Forest' in metrics['models']
    assert 'Decision Tree' in metrics['models']
    assert 'Logistic Regression' in metrics['models']
    assert 'Isolation Forest' in metrics['models']

def test_fraud_prediction_scenario():
    # Attack scenario: Draining transfer
    res = predict_transaction(
        trans_type='TRANSFER',
        amount=300000.0,
        oldbalance_org=300000.0,
        newbalance_orig=0.0,
        oldbalance_dest=0.0,
        newbalance_dest=0.0,
        step=50
    )
    assert res['prediction'] == 'Fraudulent'
    assert res['risk_level'] == 'HIGH'
    assert res['fraud_probability'] >= 0.70
    assert len(res['indicators']) > 0

def test_genuine_prediction_scenario():
    # Legitimate consumer payment
    res = predict_transaction(
        trans_type='PAYMENT',
        amount=45.0,
        oldbalance_org=1200.0,
        newbalance_orig=1155.0,
        oldbalance_dest=0.0,
        newbalance_dest=0.0,
        step=10
    )
    assert res['prediction'] == 'Genuine'
    assert res['risk_level'] == 'LOW'
    assert res['fraud_probability'] < 0.30
