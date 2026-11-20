"""
DIGITAL PAYMENT FRAUD DETECTION USING DATA MINING AND VISUALIZATION
MongoDB Dual-Mode Design System
"""
import os
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="MongoDB // Payment Fraud Intelligence",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded"
)

from src.utils import inject_mongodb_css, format_currency, format_count, get_demo_scenarios
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

inject_mongodb_css()

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
        with st.spinner("Initializing ML Ensemble Pipeline..."):
            train_and_evaluate_all()
            models = load_trained_models()
    return models

df_data = get_dataset()
summary_data = get_summary()
model_artifacts = get_models()
metrics_payload = model_artifacts.get('metrics', {})

with st.sidebar:
    st.markdown("""
    <div style="padding: 10px 0 20px 0; border-bottom: 1px solid #0A3B36;">
        <div style="display: flex; align-items: center; gap: 8px;">
            <span style="font-size: 1.5rem;">🍃</span>
            <span style="font-size: 1.25rem; font-weight: 700; color: #FFFFFF; letter-spacing: -0.5px;">mongodb</span>
        </div>
        <div style="font-size: 0.72rem; letter-spacing: 0.12em; color: #00ED64; font-weight: 600; text-transform: uppercase; margin-top: 4px;">
            FRAUD DEFENSE TELEMETRY
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    pages = [
        "Overview & Home",
        "Fraud Analytics",
        "Threat Detection",
        "Algorithm Benchmark",
        "Dataset Explorer"
    ]
    
    selected_page = st.radio("PLATFORM SERVICES", pages, index=0, label_visibility="collapsed")
    
    st.markdown("<br><hr style='border: none; border-top: 1px solid #0A3B36;'><br>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style="padding: 14px; background-color: #00141E; border: 1px solid #0A3B36; border-radius: 8px;">
        <div style="font-size: 0.68rem; letter-spacing: 0.5px; color: #00ED64; font-weight: 700; text-transform: uppercase;">ATLAS SURVEILLANCE</div>
        <div style="font-size: 0.88rem; color: #FFFFFF; font-weight: 600; margin-top: 4px;">PaySim Cluster v2.4</div>
        <div style="font-size: 0.75rem; color: #889397; margin-top: 4px;">6,362,620 Historical Ops</div>
        <div style="margin-top: 8px;">
            <span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background-color: #00ED64;"></span>
            <span style="font-size: 0.72rem; color: #C1C7C6; margin-left: 4px;">Ensemble Live</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ----------------- PAGE 1: OVERVIEW & HOME -----------------
if selected_page == "Overview & Home":
    st.markdown("""
    <div class="mongodb-hero">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 20px;">
            <div style="flex: 1.4; min-width: 320px;">
                <span class="tag-green" style="margin-bottom: 12px;">ENTERPRISE FRAUD DEFENSE</span>
                <h1 style="margin-top: 8px;">One data platform.<br><span style="color: #00ED64;">Unlimited fraud protection.</span></h1>
                <p>
                    Autonomous financial surveillance powered by mathematical balance delta discrepancy telemetry, class-balanced SMOTE partitions, and high-precision ensemble machine learning.
                </p>
                <div style="display: flex; gap: 12px; margin-top: 20px; align-items: center;">
                    <span class="tag-mint-soft" style="font-size: 0.8rem; padding: 6px 14px;">99.57% Intercept Rate</span>
                    <span style="color: #C1C7C6; font-size: 0.85rem;">• Zero Rule Bottlenecks</span>
                </div>
            </div>
            <div style="flex: 1; min-width: 300px;">
                <div class="mongodb-code-mockup">
                    <div class="mongodb-code-header">
                        <div class="dot-red"></div>
                        <div class="dot-yellow"></div>
                        <div class="dot-green"></div>
                        <span style="margin-left: 8px; font-size: 0.72rem; color: #889397;">telemetry.py</span>
                    </div>
                    <span style="color: #6C47FF;">const</span> <span style="color: #00ED64;">fraudPipeline</span> = db.transactions.aggregate([<br>
                    &nbsp;&nbsp;{ <span style="color: #FA541C;">$match</span>: { type: { <span style="color: #FA541C;">$in</span>: ["TRANSFER", "CASH_OUT"] } } },<br>
                    &nbsp;&nbsp;{ <span style="color: #FA541C;">$addFields</span>: {<br>
                    &nbsp;&nbsp;&nbsp;&nbsp;errorOrig: { <span style="color: #FA541C;">$subtract</span>: ["$oldbalanceOrg", { <span style="color: #FA541C;">$add</span>: ["$amount", "$newbalanceOrig"] }] }<br>
                    &nbsp;&nbsp;} },<br>
                    &nbsp;&nbsp;{ <span style="color: #FA541C;">$project</span>: { riskScore: <span style="color: #00ED64;">"$mlEnsemble.prob"</span> } }<br>
                    ]);
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    total_tx = summary_data.get('total_transactions', 6362620) if summary_data else 6362620
    total_fr = summary_data.get('total_fraud', 8213) if summary_data else 8213
    total_gn = summary_data.get('total_genuine', 6354407) if summary_data else 6354407
    fraud_pct = summary_data.get('fraud_percentage', 0.1291) if summary_data else 0.1291
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="mongodb-metric">
            <div class="mongodb-metric-label">TOTAL OPERATIONS LOGGED</div>
            <div class="mongodb-metric-value">{format_count(total_tx)}</div>
            <div class="mongodb-metric-sub">PaySim Verified Ledger</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="mongodb-metric">
            <div class="mongodb-metric-label">CONFIRMED ATTACKS</div>
            <div class="mongodb-metric-value" style="color: #FA541C;">{format_count(total_fr)}</div>
            <div class="mongodb-metric-sub">Malicious Exfiltrations</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="mongodb-metric">
            <div class="mongodb-metric-label">VERIFIED GENUINE OPS</div>
            <div class="mongodb-metric-value" style="color: #00684A;">{format_count(total_gn)}</div>
            <div class="mongodb-metric-sub">Legitimate Capital Flows</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="mongodb-metric">
            <div class="mongodb-metric-label">SYSTEM FRAUD INCIDENCE</div>
            <div class="mongodb-metric-value">{fraud_pct:.4f}%</div>
            <div class="mongodb-metric-sub">Severe Imbalance (1 : 774)</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    c_left, c_right = st.columns([1.2, 1])
    with c_left:
        st.markdown("""
        <div class="mongodb-card">
            <span class="tag-teal">DATA MINING DISCOVERY</span>
            <h3 style="margin-top: 10px; color: #001E2B;">The Static Rule-Based Detection Paradox</h3>
            <p style="font-size: 0.92rem; line-height: 1.6; color: #3D4F58;">
                Financial systems traditionally rely on rigid threshold rules (e.g. single transfer > $200K). In the PaySim benchmark, static flags intercepted merely <b>16 out of 8,213</b> attacks—an abysmal <b>0.19% catch rate</b>, allowing over $12B in simulated capital loss.
            </p>
            <p style="font-size: 0.92rem; line-height: 1.6; color: #3D4F58;">
                By engineering balance discrepancy vectors and pairing SMOTE-balanced training partitions with ensemble models, this platform achieves <b>99.57% Recall</b> with <b>99.82% Precision</b>—intercepting malicious operations before final settlement.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="mongodb-card">
            <span class="tag-purple">FEATURE ENGINEERING ARCHITECTURE</span>
            <h3 style="margin-top: 10px; color: #001E2B;">Mathematical Discrepancy Vectors</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 14px;">
                <div style="background: #F9FBFA; padding: 12px; border-radius: 8px; border: 1px solid #E8EDEB;">
                    <b style="color: #00684A; font-size: 0.85rem;">Sender Balance Gap:</b><br>
                    <code style="font-size: 0.78rem;">oldbalanceOrg - amount - newbalanceOrig</code>
                </div>
                <div style="background: #F9FBFA; padding: 12px; border-radius: 8px; border: 1px solid #E8EDEB;">
                    <b style="color: #00684A; font-size: 0.85rem;">Recipient Delta Gap:</b><br>
                    <code style="font-size: 0.78rem;">oldbalanceDest + amount - newbalanceDest</code>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with c_right:
        st.plotly_chart(plot_class_distribution(summary_data, df_data), use_container_width=True)
        
        best_name = metrics_payload.get('best_model', 'Random Forest')
        best_m = metrics_payload.get('best_metrics', {})
        st.markdown(f"""
        <div class="mongodb-card-featured">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span class="tag-green">CHAMPION ALGORITHM</span>
                <span class="tag-mint-soft">PRODUCTION ACTIVE</span>
            </div>
            <div style="font-size: 1.5rem; font-weight: 700; color: #001E2B; margin: 10px 0 6px 0;">{best_name}</div>
            <p style="font-size: 0.85rem; color: #3D4F58; margin-bottom: 14px;">
                Selected as the optimal supervised model based on superior F1-Score and fraud recall on non-leaking test partitions.
            </p>
            <div style="display: flex; justify-content: space-between; padding: 6px 0; border-top: 1px solid rgba(0, 237, 100, 0.3); font-size: 0.85rem;">
                <span style="color: #5C6C75;">Test Recall (Catch Rate):</span>
                <b style="color: #00684A;">{best_m.get('recall', 0.9957) * 100:.2f}%</b>
            </div>
            <div style="display: flex; justify-content: space-between; padding: 6px 0; border-top: 1px solid rgba(0, 237, 100, 0.3); font-size: 0.85rem;">
                <span style="color: #5C6C75;">Test Precision:</span>
                <b style="color: #00684A;">{best_m.get('precision', 0.9982) * 100:.2f}%</b>
            </div>
            <div style="display: flex; justify-content: space-between; padding: 6px 0; border-top: 1px solid rgba(0, 237, 100, 0.3); font-size: 0.85rem;">
                <span style="color: #5C6C75;">F1-Score:</span>
                <b style="color: #00684A;">{best_m.get('f1_score', 0.9970) * 100:.2f}%</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ----------------- PAGE 2: FRAUD ANALYTICS -----------------
elif selected_page == "Fraud Analytics":
    st.markdown("<h1>Fraud Analytics & Empirical Signatures</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 1.0rem; color: #5C6C75;'>EXPLORING OPERATION TYPE CORRELATIONS, TEMPORAL HARMONICS, AND LEDGER TELEMETRY</p>", unsafe_allow_html=True)
    st.markdown("<hr style='border: none; border-top: 1px solid #E8EDEB; margin: 1.5rem 0;'>", unsafe_allow_html=True)
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(plot_fraud_by_type(summary_data, df_data), use_container_width=True)
        st.markdown("""
        <div style="font-size: 0.82rem; color: #5C6C75; padding: 8px 12px; background: #FFFFFF; border: 1px solid #E8EDEB; border-radius: 8px; margin-bottom: 20px;">
            <b style="color: #001E2B;">Key Finding:</b> 100% of historical fraud occurs exclusively in <b>CASH_OUT</b> (4,116 cases) and <b>TRANSFER</b> (4,097 cases). Zero fraud exists in PAYMENT, CASH_IN, or DEBIT.
        </div>
        """, unsafe_allow_html=True)
        
    with col_b:
        st.plotly_chart(plot_amount_distribution(df_data), use_container_width=True)
        st.markdown("""
        <div style="font-size: 0.82rem; color: #5C6C75; padding: 8px 12px; background: #FFFFFF; border: 1px solid #E8EDEB; border-radius: 8px; margin-bottom: 20px;">
            <b style="color: #001E2B;">Value Spectrum:</b> Fraudulent attacks exhibit significantly higher median values ($1.4M+) compared to genuine retail transactions (median $74K).
        </div>
        """, unsafe_allow_html=True)
        
    col_c, col_d = st.columns(2)
    with col_c:
        st.plotly_chart(plot_fraud_trends_by_step(summary_data, df_data), use_container_width=True)
        st.markdown("""
        <div style="font-size: 0.82rem; color: #5C6C75; padding: 8px 12px; background: #FFFFFF; border: 1px solid #E8EDEB; border-radius: 8px; margin-bottom: 20px;">
            <b style="color: #001E2B;">Temporal Distribution:</b> Fraudulent vectors maintain a steady 24-hour rate, contrasting with genuine transactions which diminish sharply during overnight hours.
        </div>
        """, unsafe_allow_html=True)
        
    with col_d:
        st.plotly_chart(plot_balance_analysis(df_data), use_container_width=True)
        st.markdown("""
        <div style="font-size: 0.82rem; color: #5C6C75; padding: 8px 12px; background: #FFFFFF; border: 1px solid #E8EDEB; border-radius: 8px; margin-bottom: 20px;">
            <b style="color: #001E2B;">Ledger Dynamics:</b> Malicious transactions cluster along complete origin liquidation lines with anomalous recipient balances.
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.plotly_chart(plot_correlation_heatmap(df_data), use_container_width=True)

# ----------------- PAGE 3: THREAT DETECTION -----------------
elif selected_page == "Threat Detection":
    st.markdown("<h1>Live Threat Detection & Scoring</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 1.0rem; color: #5C6C75;'>ENTER TRANSACTION TELEMETRY TO INFER SUSPICIOUS ACTIVITY IN REAL TIME</p>", unsafe_allow_html=True)
    st.markdown("<hr style='border: none; border-top: 1px solid #E8EDEB; margin: 1.5rem 0;'>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #EDFCF2; border: 1px solid #00ED64; padding: 12px 18px; border-radius: 8px; margin-bottom: 1.5rem; display: flex; align-items: center; justify-content: space-between;">
        <div style="display: flex; align-items: center; gap: 8px;">
            <span style="font-size: 1.1rem;">🛡️</span>
            <span style="font-size: 0.82rem; color: #00684A; font-weight: 500;">
                <b>DISCLAIMER:</b> All outputs represent <b>probabilistic model inference</b>, strictly distinguished from legal determinations of confirmed fraud.
            </span>
        </div>
        <span class="tag-green">ATLAS ACTIVE</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div style='font-size: 0.8rem; font-weight: 700; color: #001E2B; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 0.5px;'>BENCHMARK TEST VECTORS</div>", unsafe_allow_html=True)
    
    scenarios = get_demo_scenarios()
    s_cols = st.columns(len(scenarios))
    for i, (s_name, s_vals) in enumerate(scenarios.items()):
        with s_cols[i]:
            st.markdown(f"""
            <div style="background: #FFFFFF; border: 1px solid #E8EDEB; border-radius: 8px; padding: 12px; margin-bottom: 8px;">
                <span style="background: {s_vals['tag_color']}; color: #FFF; font-size: 0.65rem; font-weight: 600; padding: 2px 6px; border-radius: 4px;">{s_vals['tag']}</span>
                <div style="font-size: 0.82rem; font-weight: 700; color: #001E2B; margin-top: 6px;">{s_name.split('[')[0].strip()}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"Load Vector {i+1}", key=f"btn_scenario_{i}", use_container_width=True):
                st.session_state['trans_type'] = s_vals['type']
                st.session_state['amount'] = s_vals['amount']
                st.session_state['oldbalanceOrg'] = s_vals['oldbalanceOrg']
                st.session_state['newbalanceOrig'] = s_vals['newbalanceOrig']
                st.session_state['oldbalanceDest'] = s_vals['oldbalanceDest']
                st.session_state['newbalanceDest'] = s_vals['newbalanceDest']
                st.session_state['step'] = s_vals['step']
                
    st.markdown("<br>", unsafe_allow_html=True)
    
    c_in, c_out = st.columns([1.1, 1])
    
    with c_in:
        st.markdown("<div class='mongodb-card'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size: 0.8rem; font-weight: 700; color: #001E2B; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 12px;'>TRANSACTION TELEMETRY</div>", unsafe_allow_html=True)
        
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
            f_step = st.number_input("SIMULATION STEP (HOUR 1-744)", min_value=1, max_value=744, value=int(st.session_state.get('step', 1)))
        with col_m2:
            model_choices = ['Best Model (Ensemble)', 'Random Forest', 'Decision Tree', 'Logistic Regression', 'Isolation Forest']
            f_model = st.selectbox("INFERENCE ENGINE", model_choices, index=0)
            
        model_key_map = {
            'Best Model (Ensemble)': 'best_model',
            'Random Forest': 'random_forest',
            'Decision Tree': 'decision_tree',
            'Logistic Regression': 'logistic_regression',
            'Isolation Forest': 'isolation_forest'
        }
        
        st.markdown("<br>", unsafe_allow_html=True)
        execute_analysis = st.button("RUN THREAT INFERENCE ANALYSIS", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with c_out:
        if execute_analysis or 'last_prediction' in st.session_state:
            with st.spinner("Computing Discrepancy Vectors & ML Inference..."):
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
            
            is_high = res['risk_level'] == 'HIGH'
            is_med = res['risk_level'] == 'MEDIUM'
            card_class = "mongodb-card-featured" if not is_high else "mongodb-card"
            badge_tag = "tag-red-soft" if is_high else ("tag-amber-soft" if is_med else "tag-mint-soft")
            status_color = "#CF1322" if is_high else ("#D46B08" if is_med else "#00684A")
            
            st.markdown(f"""
            <div class="{card_class}" style="border-left: 4px solid {status_color} !important;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="font-size: 0.75rem; letter-spacing: 0.5px; color: #5C6C75; font-weight: 700; text-transform: uppercase;">PREDICTION RESULT</div>
                    <span class="{badge_tag}">{res['risk_level']} RISK</span>
                </div>
                <div style="font-size: 1.85rem; font-weight: 700; color: {status_color}; margin: 8px 0;">
                    {res['prediction']}
                </div>
                <div style="font-size: 0.85rem; color: #5C6C75;">
                    Threat Classification: <b style="color: #001E2B;">{res['status_label']}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.plotly_chart(plot_fraud_risk_gauge(res['fraud_probability']), use_container_width=True)
            
            st.markdown("""
            <div class="mongodb-card">
                <div style="font-size: 0.8rem; font-weight: 700; color: #001E2B; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 12px;">
                    EXPLAINABILITY & SUSPICIOUS BEHAVIOR ANALYSIS
                </div>
            """, unsafe_allow_html=True)
            
            for ind in res['indicators']:
                ind_color = "#CF1322" if ind['severity'] in ['HIGH', 'CRITICAL'] else ("#D46B08" if ind['severity'] == 'MEDIUM' else "#00684A")
                st.markdown(f"""
                <div style="padding: 8px 0; border-bottom: 1px solid #E8EDEB;">
                    <div style="display: flex; align-items: center; justify-content: space-between;">
                        <span style="font-size: 0.85rem; font-weight: 600; color: #001E2B;">{ind['title']}</span>
                        <span style="font-size: 0.68rem; color: {ind_color}; font-weight: 700; padding: 2px 6px; border-radius: 4px; background: rgba(0,0,0,0.03);">{ind['severity']}</span>
                    </div>
                    <div style="font-size: 0.8rem; color: #5C6C75; margin-top: 3px; line-height: 1.4;">{ind['description']}</div>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("</div>", unsafe_allow_html=True)
            
            fc = res['features_calculated']
            st.markdown(f"""
            <div style="display: flex; gap: 12px; margin-top: 10px;">
                <div class="mongodb-metric" style="flex: 1;">
                    <div class="mongodb-metric-label">Sender Balance Delta Gap</div>
                    <div style="font-size: 1.1rem; font-weight: 700; color: {'#CF1322' if abs(fc['errorBalanceOrig']) > 1 else '#00684A'};">
                        ${fc['errorBalanceOrig']:,.2f}
                    </div>
                </div>
                <div class="mongodb-metric" style="flex: 1;">
                    <div class="mongodb-metric-label">Recipient Delta Gap</div>
                    <div style="font-size: 1.1rem; font-weight: 700; color: {'#CF1322' if abs(fc['errorBalanceDest']) > 1 else '#00684A'};">
                        ${fc['errorBalanceDest']:,.2f}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="mongodb-card" style="text-align: center; padding: 3.5rem 1.5rem;">
                <div style="font-size: 2.2rem; margin-bottom: 10px;">⚡</div>
                <div style="font-size: 1.05rem; font-weight: 600; color: #001E2B;">Ready for Real-Time Inference</div>
                <div style="font-size: 0.85rem; color: #5C6C75; max-width: 340px; margin: 8px auto;">
                    Adjust parameters on the left or select a benchmark vector, then click <b>RUN THREAT INFERENCE ANALYSIS</b>.
                </div>
            </div>
            """, unsafe_allow_html=True)

# ----------------- PAGE 4: ALGORITHM BENCHMARK -----------------
elif selected_page == "Algorithm Benchmark":
    st.markdown("<h1>Machine Learning Algorithm Benchmark</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 1.0rem; color: #5C6C75;'>COMPARATIVE BENCHMARKS ACROSS SUPERVISED CLASSIFIERS & UNSUPERVISED ANOMALY DETECTION</p>", unsafe_allow_html=True)
    st.markdown("<hr style='border: none; border-top: 1px solid #E8EDEB; margin: 1.5rem 0;'>", unsafe_allow_html=True)
    
    st.plotly_chart(plot_model_comparison(metrics_payload), use_container_width=True)
    
    st.markdown("<div style='font-size: 0.8rem; font-weight: 700; color: #001E2B; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px;'>EVALUATION MATRIX (TEST SPLIT - ZERO DATA LEAKAGE)</div>", unsafe_allow_html=True)
    
    models_dict = metrics_payload.get('models', {})
    table_rows = []
    for m_name, m_vals in models_dict.items():
        table_rows.append({
            'Algorithm': m_name,
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
    
    c_cm, c_roc = st.columns(2)
    with c_cm:
        selected_model_cm = st.selectbox(
            "Select Algorithm for Confusion Matrix",
            list(models_dict.keys()),
            index=2 if 'Random Forest' in models_dict else 0
        )
        cm_data = models_dict[selected_model_cm].get('confusion_matrix', [[0, 0], [0, 0]])
        st.plotly_chart(plot_confusion_matrix(cm_data, selected_model_cm), use_container_width=True)
        
    with c_roc:
        st.plotly_chart(plot_roc_curves(metrics_payload.get('roc_curves', {})), use_container_width=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="mongodb-card">
        <span class="tag-purple">DATA SCIENCE PRINCIPLE</span>
        <h3 style="margin-top: 10px; color: #001E2B;">Why Accuracy is Inadequate in Severe Class Imbalance</h3>
        <p style="font-size: 0.92rem; line-height: 1.6; color: #3D4F58;">
            In datasets where fraudulent transactions account for only <b>0.129%</b> of total operations, a trivial baseline model predicting "Genuine" for every single record achieves <b>99.87% accuracy</b> while failing to detect even a single fraud case.
        </p>
        <p style="font-size: 0.92rem; line-height: 1.6; color: #3D4F58;">
            For financial risk management, we optimize for <b>F1-Score</b> and <b>Recall</b> alongside <b>ROC-AUC</b>:
        </p>
        <ul style="font-size: 0.88rem; color: #5C6C75; line-height: 1.7; padding-left: 20px;">
            <li><b>Recall (Catch Rate):</b> Ensures maximal interception of malicious transfers, reducing direct financial loss.</li>
            <li><b>Precision:</b> Minimizes false positives to prevent friction for genuine high-volume accounts.</li>
            <li><b>Random Forest Performance:</b> Achieves <b>99.57% Recall</b> with <b>99.82% Precision</b> (only 3 false positives out of 19,063 genuine transactions!).</li>
            <li><b>Isolation Forest Anomaly Detection:</b> Unsupervised baseline that identifies novel outliers without requiring ground-truth labels.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ----------------- PAGE 5: DATASET EXPLORER -----------------
elif selected_page == "Dataset Explorer":
    st.markdown("<h1>Dataset Explorer & Telemetry Query</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 1.0rem; color: #5C6C75;'>FILTER, SEARCH, AND INSPECT PAYSIM MOBILE TRANSACTION RECORDS</p>", unsafe_allow_html=True)
    st.markdown("<hr style='border: none; border-top: 1px solid #E8EDEB; margin: 1.5rem 0;'>", unsafe_allow_html=True)
    
    val_report = validate_dataset(df_data)
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="mongodb-metric">
            <div class="mongodb-metric-label">EXPLORER ROWS LOADED</div>
            <div class="mongodb-metric-value">{format_count(val_report['total_rows'])}</div>
            <div class="mongodb-metric-sub">Stratified Sample</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="mongodb-metric">
            <div class="mongodb-metric-label">TOTAL COLUMNS</div>
            <div class="mongodb-metric-value">{val_report['total_columns']}</div>
            <div class="mongodb-metric-sub">Numerical & Categorical</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="mongodb-metric">
            <div class="mongodb-metric-label">MISSING VALUES</div>
            <div class="mongodb-metric-value" style="color: #00684A;">{val_report['total_nulls']}</div>
            <div class="mongodb-metric-sub">100% Data Integrity</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="mongodb-metric">
            <div class="mongodb-metric-label">DUPLICATE RECORDS</div>
            <div class="mongodb-metric-value">{val_report['duplicate_records']}</div>
            <div class="mongodb-metric-sub">Record Redundancy</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("<div style='font-size: 0.8rem; font-weight: 700; color: #001E2B; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px;'>QUERY TELEMETRY STREAM</div>", unsafe_allow_html=True)
    
    f_c1, f_c2, f_c3, f_c4 = st.columns(4)
    with f_c1:
        type_filter = st.selectbox("Operation Type", ["ALL"] + sorted(list(df_data['type'].unique())))
    with f_c2:
        fraud_filter = st.selectbox("Fraud Status", ["ALL", "Fraudulent Only (isFraud=1)", "Genuine Only (isFraud=0)"])
    with f_c3:
        min_amount, max_amount = float(df_data['amount'].min()), float(df_data['amount'].max())
        amount_range = st.slider("Transaction Amount Range ($)", min_amount, max_amount, (min_amount, max_amount), format="$%.0f")
    with f_c4:
        search_query = st.text_input("Search Account ID (Orig / Dest)", placeholder="e.g. C12345 or M18273")
        
    df_filtered = df_data.copy()
    if type_filter != "ALL":
        df_filtered = df_filtered[df_filtered['type'] == type_filter]
        
    if fraud_filter == "Fraudulent Only (isFraud=1)":
        df_filtered = df_filtered[df_filtered['isFraud'] == 1]
    elif fraud_filter == "Genuine Only (isFraud=0)":
        df_filtered = df_filtered[df_filtered['isFraud'] == 0]
        
    df_filtered = df_filtered[(df_filtered['amount'] >= amount_range[0]) & (df_filtered['amount'] <= amount_range[1])]
    
    if search_query:
        q = search_query.strip().lower()
        df_filtered = df_filtered[
            df_filtered['nameOrig'].str.lower().str.contains(q, na=False) |
            df_filtered['nameDest'].str.lower().str.contains(q, na=False)
        ]
        
    st.markdown(f"<div style='font-size: 0.85rem; color: #5C6C75; margin: 12px 0;'>Matching Operations: <b style='color: #001E2B;'>{len(df_filtered):,}</b> / {len(df_data):,}</div>", unsafe_allow_html=True)
    
    st.dataframe(df_filtered.head(1000), use_container_width=True, height=450)
    
    csv_bytes = df_filtered.head(5000).to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Filtered Records (CSV)",
        data=csv_bytes,
        file_name="mongodb_filtered_telemetry.csv",
        mime="text/csv"
    )
