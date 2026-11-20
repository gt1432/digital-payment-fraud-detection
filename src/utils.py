"""
Digital Payment Fraud Detection - Utilities & Styling Module
MongoDB Dual-Mode Design System
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
            "tag": "Database & Security",
            "tag_color": "#6C47FF",
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
            "tag": "Search & Retrieval",
            "tag_color": "#FA541C",
            "description": "Immediate liquidation via Cash Out with destination balance ledger reconciliation failure."
        },
        "Retail Merchant Payment [GENUINE]": {
            "type": "PAYMENT",
            "amount": 84.50,
            "oldbalanceOrg": 1540.0,
            "newbalanceOrig": 1455.50,
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0,
            "step": 24,
            "tag": "Atlas Cloud",
            "tag_color": "#0068FF",
            "description": "Standard POS merchant payment with exact mathematical balance reduction."
        },
        "Corporate Cash In Deposit [GENUINE]": {
            "type": "CASH_IN",
            "amount": 65000.0,
            "oldbalanceOrg": 12000.0,
            "newbalanceOrig": 77000.0,
            "oldbalanceDest": 95000.0,
            "newbalanceDest": 30000.0,
            "step": 48,
            "tag": "Data Platform",
            "tag_color": "#00684A",
            "description": "Verified high-liquidity capital deposit with clean ledger reconciliation."
        }
    }

def inject_mongodb_css():
    """
    Injects custom CSS enforcing MongoDB visual identity:
    - Deep navy/teal hero bands (#001E2B)
    - Bright MongoDB green (#00ED64) CTA pills universally
    - Stark white documentation and pricing surfaces (#FFFFFF)
    - 12px rounded cards with 1px hairline borders
    - Colored category tags (purple, orange, green, teal)
    - Featured pricing/threat tier with pale mint (#EDFCF2) and green border
    - Code mockup cards with terminal-aesthetic dark canvas
    """
    css = """
    <style>
    /* Google Fonts: Inter / Euclid Geometric fallback & Source Code Pro */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=Source+Code+Pro:wght@400;500;600&display=swap');

    /* Root Canvas & Text Defaults */
    html, body, [data-testid="stAppViewContainer"], .main {
        background-color: #F9FBFA !important;
        color: #001E2B !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    }

    [data-testid="stHeader"] {
        background-color: rgba(255, 255, 255, 0.92) !important;
        backdrop-filter: blur(12px) !important;
        border-bottom: 1px solid #E8EDEB !important;
    }

    [data-testid="stSidebar"] {
        background-color: #001E2B !important;
        border-right: 1px solid #0A3B36 !important;
    }

    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] div, [data-testid="stSidebar"] label {
        color: #FFFFFF !important;
    }

    /* Headings - Euclid Circular A Style */
    h1, h2, h3, h4, h5, h6 {
        color: #001E2B !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: -0.5px !important;
    }

    h1 {
        font-size: 2.2rem !important;
        line-height: 1.15 !important;
        letter-spacing: -1px !important;
        margin-bottom: 0.5rem !important;
    }

    h2 {
        font-size: 1.5rem !important;
        letter-spacing: -0.5px !important;
        margin-top: 1.5rem !important;
    }

    h3 {
        font-size: 1.15rem !important;
        letter-spacing: 0px !important;
    }

    p, span, label {
        color: #3D4F58;
    }

    /* MongoDB Deep Navy Hero Band */
    .mongodb-hero {
        background: linear-gradient(135deg, #001E2B 0%, #023430 100%);
        border-radius: 16px;
        padding: 3rem 2.5rem;
        color: #FFFFFF !important;
        margin-bottom: 2rem;
        box-shadow: 0 12px 24px -4px rgba(0, 30, 43, 0.12);
        position: relative;
        overflow: hidden;
    }

    .mongodb-hero h1 {
        color: #FFFFFF !important;
        font-size: 2.5rem !important;
        letter-spacing: -1.2px !important;
        font-weight: 700 !important;
    }

    .mongodb-hero p {
        color: #C1C7C6 !important;
        font-size: 1.05rem !important;
        line-height: 1.55 !important;
        max-width: 780px;
    }

    /* MongoDB Code Mockup Card (Terminal Aesthetic) */
    .mongodb-code-mockup {
        background-color: #00141E;
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 12px;
        padding: 1.25rem;
        font-family: 'Source Code Pro', monospace !important;
        color: #E8EDEB;
        font-size: 0.85rem;
        box-shadow: 0 16px 40px -8px rgba(0, 0, 0, 0.4);
    }

    .mongodb-code-header {
        display: flex;
        align-items: center;
        gap: 6px;
        margin-bottom: 12px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding-bottom: 8px;
    }

    .dot-red { width: 10px; height: 10px; border-radius: 50%; background-color: #FF5F56; }
    .dot-yellow { width: 10px; height: 10px; border-radius: 50%; background-color: #FFBD2E; }
    .dot-green { width: 10px; height: 10px; border-radius: 50%; background-color: #27C93F; }

    /* Standard Cards (12px rounded, 1px hairline border) */
    .mongodb-card {
        background-color: #FFFFFF;
        border: 1px solid #E8EDEB;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px 0 rgba(0, 30, 43, 0.04);
        margin-bottom: 1.25rem;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }

    .mongodb-card:hover {
        box-shadow: 0 6px 16px 0 rgba(0, 30, 43, 0.08);
    }

    /* Featured Tier Card (Mint Background + Bright Green Border) */
    .mongodb-card-featured {
        background-color: #EDFCF2 !important;
        border: 2px solid #00ED64 !important;
        border-radius: 12px;
        padding: 1.75rem;
        box-shadow: 0 8px 24px 0 rgba(0, 237, 100, 0.15);
        margin-bottom: 1.25rem;
    }

    /* Metric Cards */
    .mongodb-metric {
        background-color: #FFFFFF;
        border: 1px solid #E8EDEB;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        box-shadow: 0 2px 6px rgba(0, 30, 43, 0.03);
    }

    .mongodb-metric-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #5C6C75;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }

    .mongodb-metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #001E2B;
        letter-spacing: -0.5px;
    }

    .mongodb-metric-sub {
        font-size: 0.78rem;
        color: #889397;
        margin-top: 0.25rem;
    }

    /* Universal MongoDB Green Pill Buttons */
    .stButton > button {
        background-color: #00ED64 !important;
        color: #001E2B !important;
        border: none !important;
        border-radius: 9999px !important;
        padding: 10px 24px !important;
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        letter-spacing: 0 !important;
        box-shadow: 0 2px 4px rgba(0, 30, 43, 0.08) !important;
        transition: all 0.15s ease-in-out !important;
    }

    .stButton > button:hover {
        background-color: #00C853 !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 10px rgba(0, 237, 100, 0.25) !important;
    }

    .stButton > button:active {
        background-color: #009639 !important;
        transform: translateY(0);
    }

    /* Secondary Outlined Pill Buttons */
    .stButton > button[kind="secondary"] {
        background-color: transparent !important;
        color: #001E2B !important;
        border: 1px solid #C1C7C6 !important;
    }

    .stButton > button[kind="secondary"]:hover {
        background-color: #F9FBFA !important;
        border-color: #001E2B !important;
    }

    /* Input Fields (8px rounded, height 44px) */
    input, select, textarea, [data-baseweb="input"], [data-baseweb="select"] {
        background-color: #FFFFFF !important;
        border: 1px solid #C1C7C6 !important;
        color: #001E2B !important;
        border-radius: 8px !important;
    }

    input:focus, select:focus, textarea:focus {
        border: 2px solid #00684A !important;
        outline: none !important;
    }

    /* Category Badges & Tags */
    .tag-purple {
        display: inline-block;
        padding: 3px 8px;
        background-color: #6C47FF;
        color: #FFFFFF !important;
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        border-radius: 4px;
        text-transform: uppercase;
    }

    .tag-orange {
        display: inline-block;
        padding: 3px 8px;
        background-color: #FA541C;
        color: #FFFFFF !important;
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        border-radius: 4px;
        text-transform: uppercase;
    }

    .tag-green {
        display: inline-block;
        padding: 3px 8px;
        background-color: #00ED64;
        color: #001E2B !important;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        border-radius: 4px;
        text-transform: uppercase;
    }

    .tag-teal {
        display: inline-block;
        padding: 3px 8px;
        background-color: #00684A;
        color: #FFFFFF !important;
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        border-radius: 4px;
        text-transform: uppercase;
    }

    .tag-mint-soft {
        display: inline-block;
        padding: 4px 10px;
        background-color: #EDFCF2;
        color: #00684A !important;
        font-size: 0.72rem;
        font-weight: 700;
        border-radius: 9999px;
        border: 1px solid rgba(0, 237, 100, 0.3);
    }

    .tag-red-soft {
        display: inline-block;
        padding: 4px 10px;
        background-color: #FFF1F0;
        color: #CF1322 !important;
        font-size: 0.72rem;
        font-weight: 700;
        border-radius: 9999px;
        border: 1px solid #FFA39E;
    }

    .tag-amber-soft {
        display: inline-block;
        padding: 4px 10px;
        background-color: #FFFBE6;
        color: #D46B08 !important;
        font-size: 0.72rem;
        font-weight: 700;
        border-radius: 9999px;
        border: 1px solid #FFE58F;
    }

    /* Tables & DataFrames */
    [data-testid="stDataFrame"], table {
        border: 1px solid #E8EDEB !important;
        border-radius: 8px !important;
        background-color: #FFFFFF !important;
    }

    /* Custom Scrollbars */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-thumb {
        background: #C1C7C6;
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #00684A;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
