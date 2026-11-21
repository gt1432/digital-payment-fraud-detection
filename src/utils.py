"""
Digital Payment Fraud Detection - Utilities & Styling Module
SENTINEL AI: Bespoke FinTech & Threat Radar Design System
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
    Returns benchmark test vectors representing confirmed fraud
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
            "badge": "CRITICAL THREAT",
            "badge_color": "#FF3366",
            "description": "Attacker drains 100% of origin account to 0.00; destination mule wallet has 0 prior transaction history."
        },
        "High-Value Cash Out [FRAUD]": {
            "type": "CASH_OUT",
            "amount": 420000.0,
            "oldbalanceOrg": 420000.0,
            "newbalanceOrig": 0.0,
            "oldbalanceDest": 12500.0,
            "newbalanceDest": 12500.0,
            "step": 365,
            "badge": "EXFILTRATION",
            "badge_color": "#FF3366",
            "description": "Immediate liquidation via Cash Out with destination balance ledger reconciliation mismatch."
        },
        "Retail Merchant POS [GENUINE]": {
            "type": "PAYMENT",
            "amount": 84.50,
            "oldbalanceOrg": 1540.0,
            "newbalanceOrig": 1455.50,
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0,
            "step": 24,
            "badge": "NOMINAL FLOW",
            "badge_color": "#00F5A0",
            "description": "Standard POS merchant payment with exact mathematical balance deduction."
        },
        "Corporate Cash In Deposit [GENUINE]": {
            "type": "CASH_IN",
            "amount": 65000.0,
            "oldbalanceOrg": 12000.0,
            "newbalanceOrig": 77000.0,
            "oldbalanceDest": 95000.0,
            "newbalanceDest": 30000.0,
            "step": 48,
            "badge": "VERIFIED DEPOSIT",
            "badge_color": "#00F2FE",
            "description": "Verified high-liquidity capital deposit with clean ledger reconciliation."
        }
    }

def inject_sentinel_css():
    """
    Injects custom CSS enforcing our signature Sentinel AI FinTech design system:
    - Deep Obsidian Canvas (#080C14 / #0D1322)
    - Cyber Cyan & Neon Cobalt accents (#00F2FE / #4FACFE)
    - Frosted glassmorphic cards with subtle ambient glow
    - Plus Jakarta Sans / Inter geometric typography
    - JetBrains Mono financial numbers
    - Glowing state badges and tactile gradient action buttons
    """
    css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    /* Global Obsidian Canvas */
    html, body, [data-testid="stAppViewContainer"], .main {
        background: #080C14 !important;
        background-image: 
            radial-gradient(at 10% 10%, rgba(0, 242, 254, 0.04) 0px, transparent 50%),
            radial-gradient(at 90% 90%, rgba(99, 102, 241, 0.04) 0px, transparent 50%) !important;
        color: #F1F5F9 !important;
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    [data-testid="stHeader"] {
        background-color: rgba(8, 12, 20, 0.85) !important;
        backdrop-filter: blur(16px) !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06) !important;
    }

    [data-testid="stSidebar"] {
        background-color: #0B101D !important;
        border-right: 1px solid rgba(255, 255, 255, 0.06) !important;
    }

    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] div, [data-testid="stSidebar"] label {
        color: #94A3B8 !important;
    }

    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.03em !important;
    }

    h1 {
        font-size: 2.25rem !important;
        line-height: 1.2 !important;
        background: linear-gradient(135deg, #FFFFFF 30%, #94A3B8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem !important;
    }

    h2 {
        font-size: 1.45rem !important;
        margin-top: 1.25rem !important;
    }

    h3 {
        font-size: 1.15rem !important;
    }

    p, span, label {
        color: #94A3B8;
        font-size: 0.92rem;
    }

    code, .mono {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Sentinel Hero Banner */
    .sentinel-hero {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(13, 19, 34, 0.95) 100%);
        border: 1px solid rgba(0, 242, 254, 0.18);
        border-radius: 20px;
        padding: 2.5rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.6), 0 0 35px -10px rgba(0, 242, 254, 0.12);
        margin-bottom: 2rem;
    }

    .sentinel-hero::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 350px;
        height: 350px;
        background: radial-gradient(circle, rgba(0, 242, 254, 0.15) 0%, transparent 70%);
        pointer-events: none;
    }

    /* Frosted Glassmorphic Cards */
    .sentinel-card {
        background: rgba(15, 23, 42, 0.65);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.35);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    .sentinel-card:hover {
        border-color: rgba(0, 242, 254, 0.25);
        transform: translateY(-2px);
    }

    /* Featured Threat / Champion Tier Card */
    .sentinel-card-featured {
        background: linear-gradient(135deg, rgba(16, 28, 48, 0.8) 0%, rgba(13, 22, 40, 0.8) 100%);
        border: 1px solid rgba(0, 242, 254, 0.4);
        border-radius: 16px;
        padding: 1.75rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 12px 32px -5px rgba(0, 242, 254, 0.15);
        position: relative;
    }

    /* Metrics Cards */
    .sentinel-metric {
        background: rgba(15, 23, 42, 0.6);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 14px;
        padding: 1.25rem 1.4rem;
        box-shadow: 0 8px 20px -4px rgba(0, 0, 0, 0.3);
    }

    .sentinel-metric-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #64748B;
        font-weight: 600;
        margin-bottom: 0.35rem;
    }

    .sentinel-metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.85rem;
        font-weight: 700;
        color: #FFFFFF;
        letter-spacing: -0.03em;
    }

    .sentinel-metric-sub {
        font-size: 0.78rem;
        color: #64748B;
        margin-top: 0.3rem;
    }

    /* Tactile Cyber Action Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 100%) !important;
        color: #060B16 !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 10px 22px !important;
        font-size: 0.88rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.01em !important;
        box-shadow: 0 4px 18px rgba(0, 242, 254, 0.28) !important;
        transition: all 0.2s ease-in-out !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 242, 254, 0.42) !important;
    }

    .stButton > button:active {
        transform: translateY(0);
    }

    /* Secondary Outlined Buttons */
    .stButton > button[kind="secondary"] {
        background: rgba(15, 23, 42, 0.6) !important;
        color: #F1F5F9 !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        box-shadow: none !important;
    }

    .stButton > button[kind="secondary"]:hover {
        border-color: #00F2FE !important;
        color: #00F2FE !important;
    }

    /* Form Inputs */
    input, select, textarea, [data-baseweb="input"], [data-baseweb="select"] {
        background-color: #0E1626 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: #FFFFFF !important;
        border-radius: 10px !important;
    }

    input:focus, select:focus, textarea:focus {
        border-color: #00F2FE !important;
        box-shadow: 0 0 0 2px rgba(0, 242, 254, 0.2) !important;
    }

    /* Badges & Cyber Tags */
    .badge-cyan {
        display: inline-block;
        padding: 3px 10px;
        background: rgba(0, 242, 254, 0.12);
        color: #00F2FE !important;
        border: 1px solid rgba(0, 242, 254, 0.35);
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        border-radius: 8px;
        text-transform: uppercase;
    }

    .badge-crimson {
        display: inline-block;
        padding: 3px 10px;
        background: rgba(239, 68, 68, 0.12);
        color: #FF3366 !important;
        border: 1px solid rgba(239, 68, 68, 0.35);
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        border-radius: 8px;
        text-transform: uppercase;
    }

    .badge-emerald {
        display: inline-block;
        padding: 3px 10px;
        background: rgba(16, 185, 129, 0.12);
        color: #00F5A0 !important;
        border: 1px solid rgba(16, 185, 129, 0.35);
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        border-radius: 8px;
        text-transform: uppercase;
    }

    .badge-amber {
        display: inline-block;
        padding: 3px 10px;
        background: rgba(245, 158, 11, 0.12);
        color: #F59E0B !important;
        border: 1px solid rgba(245, 158, 11, 0.35);
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        border-radius: 8px;
        text-transform: uppercase;
    }

    /* Interactive DataFrames */
    [data-testid="stDataFrame"], table {
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        background: rgba(15, 23, 42, 0.7) !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1.5rem !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08) !important;
    }

    .stTabs [data-baseweb="tab"] {
        color: #64748B !important;
        font-weight: 600 !important;
        padding: 0.75rem 0.2rem !important;
    }

    .stTabs [aria-selected="true"] {
        color: #00F2FE !important;
        border-bottom: 2px solid #00F2FE !important;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
