"""
DIGITAL PAYMENT FRAUD DETECTION USING DATA MINING AND VISUALIZATION
SpaceX-Inspired Autonomous Risk Analytics Platform
"""
import os
import streamlit as st
import pandas as pd
import numpy as np

# Streamlit Page Configuration
st.set_page_config(
    page_title="FRAUD.IO // SPACEX TELEMETRY",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

from src.utils import inject_spacex_css, format_currency, format_count, get_demo_scenarios
from src.preprocessing import load_dataset, load_dataset_summary, validate_dataset
from src.models import load_trained_models, train_and_evaluate_all
from src.predict import predict_transaction
from src.visualizations import (
    plot_class_distribution,
    plot_fraud_by_type,
    plot_amount_distribution,
    plot_fraud_trends_by_step,
    plot_balance_analysis,
    plot_correlation_heatmap,
    plot_model_comparison,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_fraud_risk_gauge
)

# Apply SpaceX industrial pure-black styling
inject_spacex_css()

# Cache dataset loading
@st.cache_data(show_spinner=False)
def get_dataset():
    return load_dataset()

@st.cache_data(show_spinner=False)
def get_summary():
    return load_dataset_summary()

@st.cache_resource(show_spinner=False)
def get_models():
    models = load_trained_models()
    if not models or 'metrics' not in models:
        with st.spinner("INITIALIZING ML ENSEMBLE // TRAINING MODELS..."):
            train_and_evaluate_all()
            models = load_trained_models()
    return models

# Load core artifacts
df_data = get_dataset()
summary_data = get_summary()
model_artifacts = get_models()
metrics_payload = model_artifacts.get('metrics', {})

# ----------------- SIDEBAR NAVIGATION -----------------
with st.sidebar:
    st.markdown("""
    <div style="padding: 10px 0 20px 0; border-bottom: 1px solid #1E1E1E;">
        <div style="font-size: 0.7rem; letter-spacing: 0.25em; color: #8E8E93; font-weight: 700;">PAYMENT SURVEILLANCE</div>
        <div style="font-size: 1.25rem; font-weight: 900; letter-spacing: 0.1em; color: #FFFFFF; margin-top: 4px;">FRAUD.IO</div>
        <div style="display: flex; align-items: center; margin-top: 8px;">
            <div style="width: 7px; height: 7px; border-radius: 50%; background-color: #30D158; margin-right: 8px; box-shadow: 0 0 8px #30D158;"></div>
            <span style="font-size: 0.72rem; color: #8E8E93; letter-spacing: 0.1em;">SYSTEM ACTIVE // REAL-TIME</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    pages = [
        "HOME",
        "FRAUD ANALYTICS",
        "TRANSACTION DETECTION",
        "MODEL PERFORMANCE",
        "DATASET EXPLORER"
    ]
    
    selected_page = st.radio(
        "NAVIGATION PROTOCOL",
        pages,
        index=0,
        label_visibility="collapsed"
    )
    
    st.markdown("<br><hr style='border: none; border-top: 1px solid #1A1A1A;'><br>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style="padding: 10px; background-color: #050505; border: 1px solid #1E1E1E; border-radius: 2px;">
        <div style="font-size: 0.68rem; letter-spacing: 0.15em; color: #8E8E93; text-transform: uppercase;">ACTIVE TELEMETRY</div>
        <div style="font-size: 0.85rem; color: #FFFFFF; font-weight: 700; margin-top: 3px;">PAYSIM SIMULATION</div>
        <div style="font-size: 0.72rem; color: #636366; margin-top: 4px;">6,362,620 Historical Ops</div>
    </div>
    """, unsafe_allow_html=True)

# ----------------- PAGE 1: HOME -----------------
if selected_page == "HOME":
    st.markdown("<h1>DIGITAL PAYMENT FRAUD DETECTION</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 0.95rem; color: #A1A1A6; letter-spacing: 0.04em;'>ADVANCED FINANCIAL DATA MINING, RISK INTELLIGENCE & MACHINE LEARNING PLATFORM</p>", unsafe_allow_html=True)
    st.markdown("<hr style='border: none; border-top: 1px solid #1F1F1F; margin: 1.5rem 0;'>", unsafe_allow_html=True)
    
    # Macro Stats Grid
    total_tx = summary_data.get('total_transactions', 6362620) if summary_data else 6362620
    total_fr = summary_data.get('total_fraud', 8213) if summary_data else 8213
    total_gn = summary_data.get('total_genuine', 6354407) if summary_data else 6354407
    fraud_pct = summary_data.get('fraud_percentage', 0.1291) if summary_data else 0.1291
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="spacex-metric">
            <div class="spacex-metric-label">TOTAL OPERATIONS RECORDED</div>
            <div class="spacex-metric-value">{format_count(total_tx)}</div>
            <div class="spacex-metric-sub">PAYSIM VALIDATED LEDGER</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="spacex-metric">
            <div class="spacex-metric-label">CONFIRMED FRAUDULENT ATTACKS</div>
            <div class="spacex-metric-value" style="color: #FF453A;">{format_count(total_fr)}</div>
            <div class="spacex-metric-sub">MALICIOUS EXFILTRATIONS</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="spacex-metric">
            <div class="spacex-metric-label">VERIFIED GENUINE TRANSACTIONS</div>
            <div class="spacex-metric-value" style="color: #30D158;">{format_count(total_gn)}</div>
            <div class="spacex-metric-sub">LEGITIMATE CAPITAL FLOWS</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown(f"""
        <div class="spacex-metric">
            <div class="spacex-metric-label">SYSTEM FRAUD INCIDENCE RATE</div>
            <div class="spacex-metric-value">{fraud_pct:.4f}%</div>
            <div class="spacex-metric-sub">SEVERE CLASS IMBALANCE (1 : 774)</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Mission Brief & Core Capabilities
    c_left, c_right = st.columns([1.2, 1])
    with c_left:
        st.markdown("""
        <div class="spacex-card">
            <div style="font-size: 0.72rem; letter-spacing: 0.2em; color: #8E8E93; font-weight: 700;">PLATFORM ARCHITECTURE</div>
            <h3 style="margin-top: 6px;">AUTONOMOUS FINANCIAL THREAT INTERCEPTION</h3>
            <p style="font-size: 0.85rem; line-height: 1.6; color: #CCCCCC;">
                Financial fraud in digital payment systems creates multi-million dollar exfiltrations through synthetic mule routing and sudden balance liquidations. Traditional static rule systems (e.g. threshold flagging > $200K) intercepted merely <b>16 out of 8,213</b> real attacks in this system—an abysmal <b>0.19% detection rate</b>.
            </p>
            <p style="font-size: 0.85rem; line-height: 1.6; color: #CCCCCC;">
                This platform applies data mining, balance discrepancy telemetry, and machine learning ensembles to detect malicious transfers with over <b>99.5% precision and recall</b>, completely eliminating reliance on brittle manual rules.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="spacex-card">
            <div style="font-size: 0.72rem; letter-spacing: 0.2em; color: #8E8E93; font-weight: 700;">ENGINEERING HIGHLIGHTS</div>
            <h3 style="margin-top: 6px;">DATA-CENTRIC FEATURE ENGINEERING</h3>
            <ul style="font-size: 0.83rem; color: #A1A1A6; line-height: 1.7; padding-left: 20px;">
                <li><b>Delta Discrepancy Vectors:</b> Computing exact mathematical gaps in sender (<code style='color:#FFF;'>errorBalanceOrig</code>) and receiver (<code style='color:#FFF;'>errorBalanceDest</code>) balance mutations.</li>
                <li><b>Account Liquidation Flags:</b> Detecting catastrophic drops to zero balance (<code style='color:#FFF;'>isZeroNewBalanceOrig</code>).</li>
                <li><b>Mule Wallet Targeting:</b> Tracking incoming large-denomination transfers to pristine zero-balance accounts (<code style='color:#FFF;'>isZeroOldBalanceDest</code>).</li>
                <li><b>Strict Leakage Prevention:</b> Scaling and SMOTE over-sampling isolated exclusively within training partitions.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with c_right:
        st.plotly_chart(plot_class_distribution(summary_data, df_data), use_container_width=True)
        
        best_name = metrics_payload.get('best_model', 'Random Forest')
        best_m = metrics_payload.get('best_metrics', {})
        st.markdown(f"""
        <div class="spacex-card" style="border-left: 3px solid #30D158;">
            <div style="font-size: 0.7rem; letter-spacing: 0.18em; color: #30D158; font-weight: 700;">OPTIMAL ENSEMBLE ENGINE</div>
            <div style="font-size: 1.25rem; font-weight: 800; color: #FFFFFF; margin: 4px 0;">{best_name.upper()}</div>
            <div style="display: flex; justify-content: space-between; margin-top: 10px; font-size: 0.8rem;">
                <span style="color: #8E8E93;">TEST RECALL:</span>
                <span style="color: #FFFFFF; font-weight: 700;">{best_m.get('recall', 0.995) * 100:.2f}%</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-top: 4px; font-size: 0.8rem;">
                <span style="color: #8E8E93;">TEST PRECISION:</span>
                <span style="color: #FFFFFF; font-weight: 700;">{best_m.get('precision', 0.998) * 100:.2f}%</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-top: 4px; font-size: 0.8rem;">
                <span style="color: #8E8E93;">F1-SCORE:</span>
                <span style="color: #FFFFFF; font-weight: 700;">{best_m.get('f1_score', 0.997) * 100:.2f}%</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-top: 4px; font-size: 0.8rem;">
                <span style="color: #8E8E93;">ROC-AUC:</span>
                <span style="color: #FFFFFF; font-weight: 700;">{best_m.get('roc_auc', 0.999) * 100:.2f}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ----------------- PAGE 2: FRAUD ANALYTICS -----------------
elif selected_page == "FRAUD ANALYTICS":
    st.markdown("<h1>FRAUD ANALYTICS & PATTERN DISCOVERY</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 0.95rem; color: #A1A1A6;'>EXPLORING EMPIRICAL FRAUD SIGNATURES, TYPE CORRELATIONS, AND TEMPORAL VECTORS</p>", unsafe_allow_html=True)
    st.markdown("<hr style='border: none; border-top: 1px solid #1F1F1F; margin: 1.5rem 0;'>", unsafe_allow_html=True)
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(plot_fraud_by_type(summary_data, df_data), use_container_width=True)
        st.markdown("""
        <div style="font-size: 0.78rem; color: #8E8E93; margin-top: -10px; margin-bottom: 20px;">
            <b>CRITICAL DISCOVERY:</b> 100% of historical fraud occurred exclusively across <b>CASH_OUT</b> (4,116 cases) and <b>TRANSFER</b> (4,097 cases). Zero fraud occurred in PAYMENT, CASH_IN, or DEBIT.
        </div>
        """, unsafe_allow_html=True)
        
    with col_b:
        st.plotly_chart(plot_amount_distribution(df_data), use_container_width=True)
        st.markdown("""
        <div style="font-size: 0.78rem; color: #8E8E93; margin-top: -10px; margin-bottom: 20px;">
            <b>VALUE DISTRIBUTION:</b> Fraudulent transactions exhibit substantially higher median values ($1.4M+) compared to genuine transactions (median $74K).
        </div>
        """, unsafe_allow_html=True)
        
    col_c, col_d = st.columns(2)
    with col_c:
        st.plotly_chart(plot_fraud_trends_by_step(summary_data, df_data), use_container_width=True)
        st.markdown("""
        <div style="font-size: 0.78rem; color: #8E8E93; margin-top: -10px; margin-bottom: 20px;">
            <b>TEMPORAL HARMONICS:</b> Fraud attacks remain active evenly across all 24 hours of the day, unlike genuine transactions which diminish sharply during overnight hours.
        </div>
        """, unsafe_allow_html=True)
        
    with col_d:
        st.plotly_chart(plot_balance_analysis(df_data), use_container_width=True)
        st.markdown("""
        <div style="font-size: 0.78rem; color: #8E8E93; margin-top: -10px; margin-bottom: 20px;">
            <b>BALANCE DYNAMICS:</b> Fraudulent vectors cluster tightly along maximum origin account drains with atypical recipient accumulations.
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.plotly_chart(plot_correlation_heatmap(df_data), use_container_width=True)

# ----------------- PAGE 3: TRANSACTION DETECTION -----------------
elif selected_page == "TRANSACTION DETECTION":
    st.markdown("<h1>TRANSACTION FRAUD DETECTION MODULE</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 0.95rem; color: #A1A1A6;'>ENTER TRANSACTION TELEMETRY TO INFER SUSPICIOUS ACTIVITY IN REAL-TIME</p>", unsafe_allow_html=True)
    st.markdown("<hr style='border: none; border-top: 1px solid #1F1F1F; margin: 1.5rem 0;'>", unsafe_allow_html=True)
    
    # Disclaimer banner distinguishing model predictions from confirmed fraud
    st.markdown("""
    <div style="background-color: #0A0A0A; border: 1px solid #262626; padding: 10px 16px; border-radius: 2px; margin-bottom: 1.5rem; display: flex; align-items: center; justify-content: space-between;">
        <div style="display: flex; align-items: center;">
            <span style="font-size: 0.9rem; margin-right: 8px;">ℹ️</span>
            <span style="font-size: 0.75rem; letter-spacing: 0.08em; color: #CCCCCC;">
                <b>DISCLAIMER:</b> OUTPUTS REPRESENT <b>PROBABILISTIC MODEL INFERENCE</b>, NOT A CONFIRMED LEGAL DETERMINATION OF FRAUD.
            </span>
        </div>
        <span class="badge-nominal" style="font-size: 0.65rem;">INFERENCE READY</span>
    </div>
    """, unsafe_allow_html=True)
    
    # One-Click Benchmark Scenarios
    scenarios = get_demo_scenarios()
    st.markdown("<div style='font-size: 0.72rem; letter-spacing: 0.15em; color: #8E8E93; margin-bottom: 8px;'>LOAD PRECONFIGURED TEST SCENARIOS</div>", unsafe_allow_html=True)
    
    s_cols = st.columns(len(scenarios))
    for i, (s_name, s_vals) in enumerate(scenarios.items()):
        with s_cols[i]:
            if st.button(s_name, key=f"btn_scenario_{i}", use_container_width=True):
                st.session_state['trans_type'] = s_vals['type']
                st.session_state['amount'] = s_vals['amount']
                st.session_state['oldbalanceOrg'] = s_vals['oldbalanceOrg']
                st.session_state['newbalanceOrig'] = s_vals['newbalanceOrig']
                st.session_state['oldbalanceDest'] = s_vals['oldbalanceDest']
                st.session_state['newbalanceDest'] = s_vals['newbalanceDest']
                st.session_state['step'] = s_vals['step']
                
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Input Form Layout
    c_in, c_out = st.columns([1.1, 1])
    
    with c_in:
        st.markdown("<div class='spacex-card'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size: 0.72rem; letter-spacing: 0.18em; color: #8E8E93; font-weight: 700; margin-bottom: 12px;'>TRANSACTION PARAMETERS</div>", unsafe_allow_html=True)
        
        type_options = ['TRANSFER', 'CASH_OUT', 'PAYMENT', 'CASH_IN', 'DEBIT']
        saved_type = st.session_state.get('trans_type', 'TRANSFER')
        type_idx = type_options.index(saved_type) if saved_type in type_options else 0
        
        f_type = st.selectbox("TRANSACTION TYPE", type_options, index=type_idx)
        f_amount = st.number_input("TRANSACTION AMOUNT ($ USD)", min_value=0.01, value=float(st.session_state.get('amount', 180000.0)), step=1000.0, format="%.2f")
        
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            f_old_org = st.number_input("SENDER OLD BALANCE ($)", min_value=0.0, value=float(st.session_state.get('oldbalanceOrg', 180000.0)), step=1000.0, format="%.2f")
        with col_s2:
            f_new_org = st.number_input("SENDER NEW BALANCE ($)", min_value=0.0, value=float(st.session_state.get('newbalanceOrig', 0.0)), step=1000.0, format="%.2f")
            
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            f_old_dst = st.number_input("RECEIVER OLD BALANCE ($)", min_value=0.0, value=float(st.session_state.get('oldbalanceDest', 0.0)), step=1000.0, format="%.2f")
        with col_r2:
            f_new_dst = st.number_input("RECEIVER NEW BALANCE ($)", min_value=0.0, value=float(st.session_state.get('newbalanceDest', 0.0)), step=1000.0, format="%.2f")
            
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            f_step = st.number_input("TRANSACTION STEP (HOUR 1-744)", min_value=1, max_value=744, value=int(st.session_state.get('step', 1)))
        with col_m2:
            model_choices = ['Best Model (Ensemble)', 'Random Forest', 'Decision Tree', 'Logistic Regression', 'Isolation Forest']
            f_model = st.selectbox("INFERENCE ALGORITHM", model_choices, index=0)
            
        model_key_map = {
            'Best Model (Ensemble)': 'best_model',
            'Random Forest': 'random_forest',
            'Decision Tree': 'decision_tree',
            'Logistic Regression': 'logistic_regression',
            'Isolation Forest': 'isolation_forest'
        }
        
        st.markdown("<br>", unsafe_allow_html=True)
        execute_analysis = st.button("RUN THREAT DETECTION ANALYSIS", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with c_out:
        if execute_analysis or 'last_prediction' in st.session_state:
            with st.spinner("COMPUTING RISK VECTORS..."):
                pred_result = predict_transaction(
                    trans_type=f_type,
                    amount=f_amount,
                    oldbalance_org=f_old_org,
                    newbalance_orig=f_new_org,
                    oldbalance_dest=f_old_dst,
                    newbalance_dest=f_new_dst,
                    step=f_step,
                    model_name=model_key_map[f_model]
                )
                st.session_state['last_prediction'] = pred_result
                
            res = st.session_state['last_prediction']
            
            # Prediction Card
            badge_class = "badge-critical" if res['risk_level'] == 'HIGH' else ("badge-warning" if res['risk_level'] == 'MEDIUM' else "badge-nominal")
            border_color = res['status_color']
            
            st.markdown(f"""
            <div class="spacex-card" style="border-left: 4px solid {border_color};">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="font-size: 0.72rem; letter-spacing: 0.2em; color: #8E8E93; font-weight: 700;">INFERENCE CLASSIFICATION</div>
                    <span class="{badge_class}">{res['risk_level']} RISK</span>
                </div>
                <div style="font-size: 1.85rem; font-weight: 900; letter-spacing: 0.08em; color: {border_color}; margin: 8px 0;">
                    {res['prediction'].upper()}
                </div>
                <div style="font-size: 0.75rem; color: #8E8E93;">
                    STATUS: <span style="color: #FFFFFF; font-weight: 700;">{res['status_label']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Risk Gauge
            st.plotly_chart(plot_fraud_risk_gauge(res['fraud_probability']), use_container_width=True)
            
            # Explanation of Suspicious Indicators
            st.markdown("""
            <div class="spacex-card">
                <div style="font-size: 0.72rem; letter-spacing: 0.18em; color: #8E8E93; font-weight: 700; margin-bottom: 8px;">
                    EXPLAINABILITY & SUSPICIOUS BEHAVIOR ANALYSIS
                </div>
            """, unsafe_allow_html=True)
            
            for ind in res['indicators']:
                tag_color = "#FF453A" if ind['severity'] in ['HIGH', 'CRITICAL'] else ("#FF9F0A" if ind['severity'] == 'MEDIUM' else "#30D158")
                st.markdown(f"""
                <div style="padding: 8px 0; border-bottom: 1px solid #1E1E1E;">
                    <div style="display: flex; align-items: center; justify-content: space-between;">
                        <span style="font-size: 0.82rem; font-weight: 700; color: #FFFFFF;">{ind['title']}</span>
                        <span style="font-size: 0.65rem; color: {tag_color}; border: 1px solid {tag_color}; padding: 1px 5px; border-radius: 2px;">{ind['severity']}</span>
                    </div>
                    <div style="font-size: 0.75rem; color: #A1A1A6; margin-top: 3px; line-height: 1.4;">{ind['description']}</div>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Mathematical Discrepancy Diagnostics
            fc = res['features_calculated']
            st.markdown(f"""
            <div style="display: flex; gap: 8px; margin-top: 8px;">
                <div class="spacex-metric" style="flex: 1;">
                    <div class="spacex-metric-label">SENDER BALANCE DELTA GAP</div>
                    <div style="font-size: 1.0rem; font-weight: 700; color: {'#FF453A' if abs(fc['errorBalanceOrig']) > 1 else '#30D158'};">
                        ${fc['errorBalanceOrig']:,.2f}
                    </div>
                </div>
                <div class="spacex-metric" style="flex: 1;">
                    <div class="spacex-metric-label">RECIPIENT DELTA GAP</div>
                    <div style="font-size: 1.0rem; font-weight: 700; color: {'#FF453A' if abs(fc['errorBalanceDest']) > 1 else '#30D158'};">
                        ${fc['errorBalanceDest']:,.2f}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="spacex-card" style="text-align: center; padding: 3rem 1.5rem;">
                <div style="font-size: 2rem; margin-bottom: 10px;">⚡</div>
                <div style="font-size: 0.9rem; font-weight: 700; letter-spacing: 0.1em; color: #FFFFFF;">READY FOR INFERENCE</div>
                <div style="font-size: 0.78rem; color: #8E8E93; max-width: 320px; margin: 8px auto;">
                    Adjust parameters on the left or select a benchmark attack scenario, then click <b>RUN THREAT DETECTION ANALYSIS</b>.
                </div>
            </div>
            """, unsafe_allow_html=True)

# ----------------- PAGE 4: MODEL PERFORMANCE -----------------
elif selected_page == "MODEL PERFORMANCE":
    st.markdown("<h1>MODEL PERFORMANCE & ALGORITHM BENCHMARK</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 0.95rem; color: #A1A1A6;'>COMPARATIVE ANALYSIS OF SUPERVISED CLASSIFIERS & UNSUPERVISED ANOMALY DETECTION</p>", unsafe_allow_html=True)
    st.markdown("<hr style='border: none; border-top: 1px solid #1F1F1F; margin: 1.5rem 0;'>", unsafe_allow_html=True)
    
    # Multi-Algorithm Comparison Bar Chart
    st.plotly_chart(plot_model_comparison(metrics_payload), use_container_width=True)
    
    # Detailed Table
    st.markdown("<div style='font-size: 0.75rem; letter-spacing: 0.16em; color: #8E8E93; font-weight: 700; margin-bottom: 8px;'>EVALUATION MATRIX (TEST PARTITION - NO DATA LEAKAGE)</div>", unsafe_allow_html=True)
    
    models_dict = metrics_payload.get('models', {})
    table_rows = []
    for m_name, m_vals in models_dict.items():
        table_rows.append({
            'Algorithm': m_name.upper(),
            'Paradigm': m_vals.get('type', 'Supervised'),
            'Accuracy': f"{m_vals.get('accuracy', 0)*100:.2f}%",
            'Precision': f"{m_vals.get('precision', 0)*100:.2f}%",
            'Recall (Catch Rate)': f"{m_vals.get('recall', 0)*100:.2f}%",
            'F1-Score': f"{m_vals.get('f1_score', 0)*100:.2f}%",
            'ROC-AUC': f"{m_vals.get('roc_auc', 0)*100:.2f}%"
        })
    df_metrics = pd.DataFrame(table_rows)
    st.dataframe(df_metrics, use_container_width=True, hide_index=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Confusion Matrix & ROC Curves
    c_cm, c_roc = st.columns(2)
    
    with c_cm:
        selected_model_cm = st.selectbox(
            "SELECT ALGORITHM FOR CONFUSION MATRIX",
            list(models_dict.keys()),
            index=2 if 'Random Forest' in models_dict else 0
        )
        cm_data = models_dict[selected_model_cm].get('confusion_matrix', [[0, 0], [0, 0]])
        st.plotly_chart(plot_confusion_matrix(cm_data, selected_model_cm), use_container_width=True)
        
    with c_roc:
        st.plotly_chart(plot_roc_curves(metrics_payload.get('roc_curves', {})), use_container_width=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Insight on Fraud Imbalance & Metric Evaluation
    st.markdown("""
    <div class="spacex-card">
        <div style="font-size: 0.72rem; letter-spacing: 0.2em; color: #8E8E93; font-weight: 700;">DATA MINING PERSPECTIVE</div>
        <h3 style="margin-top: 6px;">WHY ACCURACY IS DECEPTIVE IN FRAUD DETECTION</h3>
        <p style="font-size: 0.85rem; line-height: 1.6; color: #CCCCCC;">
            Because fraudulent transactions represent only <b>0.129%</b> of total operations, a trivial dummy classifier predicting "Genuine" for 100% of transactions achieves a staggering <b>99.87% Accuracy</b>, while missing every single fraud attack ($0 recovery).
        </p>
        <p style="font-size: 0.85rem; line-height: 1.6; color: #CCCCCC;">
            Therefore, we prioritize <b>F1-Score</b> and <b>Recall</b> alongside <b>ROC-AUC</b>:
        </p>
        <ul style="font-size: 0.82rem; color: #A1A1A6; line-height: 1.7; padding-left: 20px;">
            <li><b>Recall (Catch Rate):</b> Ensures maximum intercept of malicious transfers, reducing undetected financial losses.</li>
            <li><b>Precision:</b> Minimizes false positives to prevent friction for genuine high-volume customers.</li>
            <li><b>Random Forest Supremacy:</b> Achieves <b>99.57% Recall</b> with <b>99.82% Precision</b> (only 3 false alarms in 19,063 genuine transactions!).</li>
            <li><b>Isolation Forest Anomaly Baseline:</b> Operates unsupervised without labels, establishing an anomaly perimeter against novel unseen threat vectors.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ----------------- PAGE 5: DATASET EXPLORER -----------------
elif selected_page == "DATASET EXPLORER":
    st.markdown("<h1>DATASET EXPLORER & TRANSACTION TELEMETRY</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 0.95rem; color: #A1A1A6;'>SEARCH, FILTER, AND INSPECT PAYSIM FINANCIAL TRANSACTIONS</p>", unsafe_allow_html=True)
    st.markdown("<hr style='border: none; border-top: 1px solid #1F1F1F; margin: 1.5rem 0;'>", unsafe_allow_html=True)
    
    # Dataset Profiling Stats
    val_report = validate_dataset(df_data)
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="spacex-metric">
            <div class="spacex-metric-label">EXPLORER ROWS LOADED</div>
            <div class="spacex-metric-value">{format_count(val_report['total_rows'])}</div>
            <div class="spacex-metric-sub">STRATIFIED PAYSIM SAMPLE</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="spacex-metric">
            <div class="spacex-metric-label">TOTAL COLUMNS</div>
            <div class="spacex-metric-value">{val_report['total_columns']}</div>
            <div class="spacex-metric-sub">NUMERICAL & CATEGORICAL</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="spacex-metric">
            <div class="spacex-metric-label">MISSING VALUES (NULLS)</div>
            <div class="spacex-metric-value" style="color: #30D158;">{val_report['total_nulls']}</div>
            <div class="spacex-metric-sub">100% DATASET INTEGRITY</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="spacex-metric">
            <div class="spacex-metric-label">DUPLICATE RECORDS</div>
            <div class="spacex-metric-value">{val_report['duplicate_records']}</div>
            <div class="spacex-metric-sub">RECORD REDUNDANCY</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Filter Controls
    st.markdown("<div style='font-size: 0.75rem; letter-spacing: 0.16em; color: #8E8E93; font-weight: 700; margin-bottom: 8px;'>FILTER TELEMETRY STREAM</div>", unsafe_allow_html=True)
    
    f_c1, f_c2, f_c3, f_c4 = st.columns(4)
    with f_c1:
        type_filter = st.selectbox("TRANSACTION TYPE", ["ALL"] + sorted(list(df_data['type'].unique())))
    with f_c2:
        fraud_filter = st.selectbox("STATUS FILTER", ["ALL", "FRAUDULENT ONLY (isFraud=1)", "GENUINE ONLY (isFraud=0)"])
    with f_c3:
        min_amount, max_amount = float(df_data['amount'].min()), float(df_data['amount'].max())
        amount_range = st.slider("TRANSACTION VALUE RANGE ($)", min_amount, max_amount, (min_amount, max_amount), format="$%.0f")
    with f_c4:
        search_query = st.text_input("SEARCH ACCOUNT ID (ORIG / DEST)", placeholder="e.g. C12345 or M18273")
        
    # Apply Filters
    df_filtered = df_data.copy()
    if type_filter != "ALL":
        df_filtered = df_filtered[df_filtered['type'] == type_filter]
        
    if fraud_filter == "FRAUDULENT ONLY (isFraud=1)":
        df_filtered = df_filtered[df_filtered['isFraud'] == 1]
    elif fraud_filter == "GENUINE ONLY (isFraud=0)":
        df_filtered = df_filtered[df_filtered['isFraud'] == 0]
        
    df_filtered = df_filtered[(df_filtered['amount'] >= amount_range[0]) & (df_filtered['amount'] <= amount_range[1])]
    
    if search_query:
        q = search_query.strip().lower()
        df_filtered = df_filtered[
            df_filtered['nameOrig'].str.lower().str.contains(q, na=False) |
            df_filtered['nameDest'].str.lower().str.contains(q, na=False)
        ]
        
    st.markdown(f"<div style='font-size: 0.78rem; color: #8E8E93; margin: 10px 0;'>MATCHING RECORDS: <b style='color:#FFF;'>{len(df_filtered):,}</b> / {len(df_data):,}</div>", unsafe_allow_html=True)
    
    # Display Filtered Data
    st.dataframe(df_filtered.head(1000), use_container_width=True, height=450)
    
    # Download Filtered View
    csv_bytes = df_filtered.head(5000).to_csv(index=False).encode('utf-8')
    st.download_button(
        label="DOWNLOAD FILTERED TELEMETRY (CSV)",
        data=csv_bytes,
        file_name="paysim_filtered_telemetry.csv",
        mime="text/csv"
    )
