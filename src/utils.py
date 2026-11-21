"""
Digital Payment Fraud Detection - Utilities & Styling Module
SENTINEL AI: High-End Obsidian Cyber FinTech UI/UX
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
        "Draining Attack [FRAUD]": {
            "type": "TRANSFER",
            "amount": 250000.0,
            "oldbalanceOrg": 250000.0,
            "newbalanceOrig": 0.0,
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0,
            "step": 142,
            "badge": "CRITICAL ATTACK",
            "badge_color": "#FF0055",
            "description": "Attacker drains 100% of origin account to 0.00; destination mule wallet has 0 prior transaction history."
        },
        "Cash Out Exfiltration [FRAUD]": {
            "type": "CASH_OUT",
            "amount": 420000.0,
            "oldbalanceOrg": 420000.0,
            "newbalanceOrig": 0.0,
            "oldbalanceDest": 12500.0,
            "newbalanceDest": 12500.0,
            "step": 365,
            "badge": "EXFILTRATION",
            "badge_color": "#FF0055",
            "description": "Immediate liquidation via Cash Out with destination balance ledger reconciliation mismatch."
        },
        "Retail POS Merchant [GENUINE]": {
            "type": "PAYMENT",
            "amount": 84.50,
            "oldbalanceOrg": 1540.0,
            "newbalanceOrig": 1455.50,
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0,
            "step": 24,
            "badge": "NOMINAL FLOW",
            "badge_color": "#00FF9D",
            "description": "Standard POS merchant payment with exact mathematical balance deduction."
        },
        "Corporate Liquidity Deposit [GENUINE]": {
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
    Injects custom high-specificity CSS enforcing the Sentinel AI FinTech design system:
    - Deep Obsidian Canvas (#050811 / #0B1120) with ambient cyan / violet radial glow
    - Glowing glassmorphic cards with subtle animated pulse
    - Plus Jakarta Sans geometric typography with JetBrains Mono numbers
    - Vibrant cyber accents: Electric Cyan (#00F2FE), Neon Emerald (#00FF9D), Radiant Crimson (#FF0055)
    - High-specificity selectors to override Streamlit internal styles
    """
    css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    /* Global Obsidian Canvas */
    html, body, [data-testid="stAppViewContainer"], .main, .stApp {
        background: #050811 !important;
        background-image: 
            radial-gradient(at 0% 0%, rgba(0, 242, 254, 0.07) 0px, transparent 50%),
            radial-gradient(at 100% 100%, rgba(121, 40, 202, 0.07) 0px, transparent 50%) !important;
        color: #F8FAFC !important;
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    [data-testid="stHeader"] {
        background-color: rgba(5, 8, 17, 0.85) !important;
        backdrop-filter: blur(16px) !important;
        border-bottom: 1px solid rgba(0, 242, 254, 0.12) !important;
    }

    [data-testid="stSidebar"] {
        background-color: #080D1A !important;
        border-right: 1px solid rgba(0, 242, 254, 0.1) !important;
    }

    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] div, [data-testid="stSidebar"] label {
        color: #94A3B8 !important;
    }

    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 800 !important;
        letter-spacing: -0.03em !important;
    }

    h1 {
        font-size: 2.3rem !important;
        line-height: 1.2 !important;
        background: linear-gradient(135deg, #FFFFFF 20%, #00F2FE 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem !important;
    }

    h2 {
        font-size: 1.5rem !important;
        color: #F1F5F9 !important;
        margin-top: 1.25rem !important;
    }

    h3 {
        font-size: 1.15rem !important;
        color: #F1F5F9 !important;
    }

    p, span, label {
        color: #94A3B8;
        font-size: 0.92rem;
    }

    code, .mono {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Animated Live Telemetry Ticker */
    .sentinel-ticker {
        background: rgba(11, 17, 32, 0.85);
        border: 1px solid rgba(0, 242, 254, 0.2);
        border-radius: 12px;
        padding: 8px 16px;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        box-shadow: 0 4px 20px rgba(0, 242, 254, 0.08);
    }

    .ticker-pulse {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #00FF9D;
        box-shadow: 0 0 10px #00FF9D;
        margin-right: 8px;
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.4; transform: scale(1.2); }
        100% { opacity: 1; transform: scale(1); }
    }

    /* Hero Banner */
    .sentinel-hero {
        background: linear-gradient(135deg, rgba(14, 23, 42, 0.85) 0%, rgba(10, 16, 30, 0.95) 100%);
        border: 1px solid rgba(0, 242, 254, 0.25);
        border-radius: 20px;
        padding: 2.5rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 20px 50px -15px rgba(0, 0, 0, 0.7), 0 0 35px -10px rgba(0, 242, 254, 0.15);
        margin-bottom: 2rem;
    }

    .sentinel-hero::before {
        content: '';
        position: absolute;
        top: -60%;
        right: -25%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(0, 242, 254, 0.18) 0%, transparent 70%);
        pointer-events: none;
    }

    /* Glassmorphic Obsidian Cards */
    .sentinel-card {
        background: rgba(14, 22, 38, 0.65);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.4);
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }

    .sentinel-card:hover {
        border-color: rgba(0, 242, 254, 0.35);
        box-shadow: 0 14px 35px -5px rgba(0, 242, 254, 0.12);
        transform: translateY(-2px);
    }

    /* Featured Threat / Champion Tier Card */
    .sentinel-card-featured {
        background: linear-gradient(135deg, rgba(15, 27, 46, 0.85) 0%, rgba(11, 19, 36, 0.9) 100%);
        border: 1px solid rgba(0, 242, 254, 0.45);
        border-radius: 16px;
        padding: 1.75rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 12px 35px -5px rgba(0, 242, 254, 0.2);
        position: relative;
    }

    /* Metric Containers */
    .sentinel-metric {
        background: rgba(14, 22, 38, 0.6);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.25rem 1.4rem;
        box-shadow: 0 8px 24px -4px rgba(0, 0, 0, 0.35);
    }

    .sentinel-metric-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #64748B;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }

    .sentinel-metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.95rem;
        font-weight: 800;
        color: #FFFFFF;
        letter-spacing: -0.03em;
    }

    .sentinel-metric-sub {
        font-size: 0.78rem;
        color: #64748B;
        margin-top: 0.3rem;
    }

    /* Glowing Cyber Action Buttons */
    .stButton > button, div.stButton > button:first-child {
        background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 100%) !important;
        color: #050811 !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 10px 24px !important;
        font-size: 0.88rem !important;
        font-weight: 800 !important;
        letter-spacing: 0.01em !important;
        box-shadow: 0 4px 20px rgba(0, 242, 254, 0.32) !important;
        transition: all 0.2s ease-in-out !important;
    }

    .stButton > button:hover, div.stButton > button:first-child:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 28px rgba(0, 242, 254, 0.48) !important;
        filter: brightness(1.08) !important;
    }

    .stButton > button:active, div.stButton > button:first-child:active {
        transform: translateY(0) !important;
    }

    /* Secondary Outlined Buttons */
    .stButton > button[kind="secondary"] {
        background: rgba(14, 22, 38, 0.65) !important;
        color: #F8FAFC !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        box-shadow: none !important;
    }

    .stButton > button[kind="secondary"]:hover {
        border-color: #00F2FE !important;
        color: #00F2FE !important;
    }

    /* Form Inputs */
    input, select, textarea, [data-baseweb="input"], [data-baseweb="select"] {
        background-color: #0A101D !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        color: #FFFFFF !important;
        border-radius: 10px !important;
    }

    input:focus, select:focus, textarea:focus {
        border-color: #00F2FE !important;
        box-shadow: 0 0 0 2px rgba(0, 242, 254, 0.25) !important;
    }

    /* Badges & Cyber Tags */
    .badge-cyan {
        display: inline-block;
        padding: 3px 10px;
        background: rgba(0, 242, 254, 0.12);
        color: #00F2FE !important;
        border: 1px solid rgba(0, 242, 254, 0.4);
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        border-radius: 8px;
        text-transform: uppercase;
    }

    .badge-crimson {
        display: inline-block;
        padding: 3px 10px;
        background: rgba(255, 0, 85, 0.12);
        color: #FF0055 !important;
        border: 1px solid rgba(255, 0, 85, 0.4);
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        border-radius: 8px;
        text-transform: uppercase;
    }

    .badge-emerald {
        display: inline-block;
        padding: 3px 10px;
        background: rgba(0, 255, 157, 0.12);
        color: #00FF9D !important;
        border: 1px solid rgba(0, 255, 157, 0.4);
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
        border: 1px solid rgba(245, 158, 11, 0.4);
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
        background: rgba(14, 22, 38, 0.7) !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1.5rem !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08) !important;
    }

    .stTabs [data-baseweb="tab"] {
        color: #64748B !important;
        font-weight: 700 !important;
        padding: 0.75rem 0.2rem !important;
    }

    .stTabs [aria-selected="true"] {
        color: #00F2FE !important;
        border-bottom: 2px solid #00F2FE !important;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
