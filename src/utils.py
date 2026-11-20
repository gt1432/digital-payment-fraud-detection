"""
Digital Payment Fraud Detection - Utilities & Styling Module
SpaceX Industrial UI/UX Specification
Author: Senior ML Engineer & UI/UX Designer
"""
import streamlit as st
from typing import Dict, Any, List

def format_currency(val: float) -> str:
    """Formats numeric values to USD currency format."""
    return f"${val:,.2f}"

def format_count(val: int) -> str:
    """Formats large integers with comma separators."""
    return f"{val:,}"

def get_demo_scenarios() -> Dict[str, Dict[str, Any]]:
    """
    Returns realistic benchmark test vectors representing confirmed fraud
    and normal legitimate financial transactions.
    """
    return {
        "Draining Transfer Attack [FRAUD]": {
            "type": "TRANSFER",
            "amount": 250000.0,
            "oldbalanceOrg": 250000.0,
            "newbalanceOrig": 0.0,
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0,
            "step": 142,
            "description": "Attacker drains 100% of origin account to 0.00; destination mule account holds 0 prior transactions."
        },
        "High-Value Cash Out [FRAUD]": {
            "type": "CASH_OUT",
            "amount": 420000.0,
            "oldbalanceOrg": 420000.0,
            "newbalanceOrig": 0.0,
            "oldbalanceDest": 12500.0,
            "newbalanceDest": 12500.0,
            "step": 365,
            "description": "Immediate liquidation via Cash Out with destination balance ledger mismatch."
        },
        "Retail Merchant Payment [GENUINE]": {
            "type": "PAYMENT",
            "amount": 84.50,
            "oldbalanceOrg": 1540.0,
            "newbalanceOrig": 1455.50,
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0,
            "step": 24,
            "description": "Standard POS merchant payment with exact balance deduction."
        },
        "Corporate Cash In Deposit [GENUINE]": {
            "type": "CASH_IN",
            "amount": 65000.0,
            "oldbalanceOrg": 12000.0,
            "newbalanceOrig": 77000.0,
            "oldbalanceDest": 95000.0,
            "newbalanceDest": 30000.0,
            "step": 48,
            "description": "Verified high-liquidity capital deposit with clean ledger reconciliation."
        }
    }

def inject_spacex_css():
    """
    Injects custom CSS enforcing pure SpaceX minimal industrial design language.
    """
    css = """
    <style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600;700&display=swap');

    /* Global Dark Reset */
    html, body, [data-testid="stAppViewContainer"], .main {
        background-color: #000000 !important;
        color: #FFFFFF !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    }

    [data-testid="stHeader"] {
        background-color: rgba(0, 0, 0, 0.85) !important;
        backdrop-filter: blur(12px) !important;
        border-bottom: 1px solid #1A1A1A !important;
    }

    [data-testid="stSidebar"] {
        background-color: #050505 !important;
        border-right: 1px solid #1A1A1A !important;
    }

    [data-testid="stSidebarNav"] {
        background-color: transparent !important;
    }

    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 800 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.12em !important;
    }

    h1 {
        font-size: 1.85rem !important;
        border-left: 3px solid #FFFFFF;
        padding-left: 12px;
        margin-bottom: 1.5rem !important;
    }

    h2 {
        font-size: 1.25rem !important;
        letter-spacing: 0.14em !important;
        margin-top: 1.2rem !important;
    }

    h3 {
        font-size: 1.0rem !important;
        letter-spacing: 0.1em !important;
    }

    p, span, label, div {
        color: #D1D1D6;
    }

    /* Monospace elements */
    code, pre, .mono {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Minimal Industrial Cards */
    .spacex-card {
        background-color: #080808;
        border: 1px solid #1E1E1E;
        border-radius: 4px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        transition: border-color 0.2s ease;
    }

    .spacex-card:hover {
        border-color: #383838;
    }

    /* Metric Containers */
    .spacex-metric {
        background-color: #050505;
        border: 1px solid #1A1A1A;
        border-radius: 2px;
        padding: 1rem;
        text-align: left;
    }

    .spacex-metric-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.16em;
        color: #8E8E93;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }

    .spacex-metric-value {
        font-size: 1.65rem;
        font-weight: 800;
        color: #FFFFFF;
        letter-spacing: -0.02em;
        font-family: 'Inter', sans-serif;
    }

    .spacex-metric-sub {
        font-size: 0.75rem;
        color: #636366;
        margin-top: 0.3rem;
    }

    /* Ghost-Outlined Pill Buttons */
    .stButton > button {
        background-color: transparent !important;
        color: #FFFFFF !important;
        border: 1px solid #FFFFFF !important;
        border-radius: 9999px !important;
        padding: 0.45rem 1.6rem !important;
        font-size: 0.75rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.14em !important;
        transition: all 0.2s ease-in-out !important;
        box-shadow: none !important;
    }

    .stButton > button:hover {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        border-color: #FFFFFF !important;
    }

    .stButton > button:active {
        transform: scale(0.98);
    }

    /* Primary Accent Button */
    .stButton > button[kind="primary"] {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        border: 1px solid #FFFFFF !important;
    }

    .stButton > button[kind="primary"]:hover {
        background-color: #E5E5EA !important;
        color: #000000 !important;
    }

    /* Input Fields */
    input, select, textarea, [data-baseweb="input"], [data-baseweb="select"] {
        background-color: #0A0A0A !important;
        border: 1px solid #262626 !important;
        color: #FFFFFF !important;
        border-radius: 2px !important;
    }

    input:focus, select:focus, textarea:focus {
        border-color: #FFFFFF !important;
        outline: none !important;
    }

    /* Status Badges */
    .badge-critical {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        background-color: rgba(255, 69, 58, 0.15);
        border: 1px solid #FF453A;
        color: #FF453A;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        border-radius: 2px;
    }

    .badge-nominal {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        background-color: rgba(48, 209, 88, 0.15);
        border: 1px solid #30D158;
        color: #30D158;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        border-radius: 2px;
    }

    .badge-warning {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        background-color: rgba(255, 159, 10, 0.15);
        border: 1px solid #FF9F0A;
        color: #FF9F0A;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        border-radius: 2px;
    }

    /* Tables & DataFrames */
    [data-testid="stDataFrame"], table {
        border: 1px solid #1E1E1E !important;
        background-color: #050505 !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: transparent !important;
        border-bottom: 1px solid #1E1E1E !important;
        gap: 1.5rem !important;
    }

    .stTabs [data-baseweb="tab"] {
        color: #8E8E93 !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.12em !important;
        padding: 0.6rem 0.2rem !important;
        border: none !important;
        background-color: transparent !important;
    }

    .stTabs [aria-selected="true"] {
        color: #FFFFFF !important;
        border-bottom: 2px solid #FFFFFF !important;
    }

    /* Mobile Responsive Tweaks */
    @media (max-width: 768px) {
        h1 {
            font-size: 1.4rem !important;
        }
        .spacex-metric-value {
            font-size: 1.3rem !important;
        }
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
