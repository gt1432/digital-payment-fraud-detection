"""
Digital Payment Fraud Detection - Visualization Module
Sentinel AI: Bespoke FinTech & Threat Radar Design System
Author: Senior ML Engineer & UI/UX Designer
"""
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

SENTINEL_THEME = {
    'bg': 'rgba(0,0,0,0)',
    'card_bg': '#0F172A',
    'grid': 'rgba(255, 255, 255, 0.05)',
    'border': 'rgba(255, 255, 255, 0.08)',
    'text': '#F8FAFC',
    'text_muted': '#64748B',
    'cyan': '#00F2FE',
    'blue': '#4FACFE',
    'emerald': '#00F5A0',
    'crimson': '#FF3366',
    'violet': '#6366F1',
    'amber': '#F59E0B',
    'font': 'Plus Jakarta Sans, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
}

def apply_sentinel_layout(fig: go.Figure, title: str = "", height: int = 400) -> go.Figure:
    """
    Applies our signature Sentinel AI obsidian glassmorphic styling to any Plotly figure.
    """
    fig.update_layout(
        title={
            'text': title,
            'y': 0.95,
            'x': 0.03,
            'xanchor': 'left',
            'yanchor': 'top',
            'font': {
                'family': SENTINEL_THEME['font'],
                'size': 14,
                'color': SENTINEL_THEME['text'],
                'weight': 700
            }
        },
        paper_bgcolor=SENTINEL_THEME['bg'],
        plot_bgcolor=SENTINEL_THEME['bg'],
        font=dict(family=SENTINEL_THEME['font'], color=SENTINEL_THEME['text'], size=11),
        margin=dict(l=40, r=40, t=55, b=40),
        height=height,
        hovermode='closest',
        hoverlabel=dict(
            bgcolor='#0F172A',
            bordercolor=SENTINEL_THEME['cyan'],
            font=dict(color='#FFFFFF', family=SENTINEL_THEME['font'], size=12)
        ),
        legend=dict(
            font=dict(color=SENTINEL_THEME['text_muted'], size=11),
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1,
            bgcolor='rgba(0,0,0,0)'
        )
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor=SENTINEL_THEME['grid'],
        gridwidth=1,
        zeroline=False,
        linecolor=SENTINEL_THEME['border'],
        tickfont=dict(color=SENTINEL_THEME['text_muted'], size=10)
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=SENTINEL_THEME['grid'],
        gridwidth=1,
        zeroline=False,
        linecolor=SENTINEL_THEME['border'],
        tickfont=dict(color=SENTINEL_THEME['text_muted'], size=10)
    )
    return fig

def plot_class_distribution(summary: Optional[Dict[str, Any]] = None, df: Optional[pd.DataFrame] = None) -> go.Figure:
    """
    1. Genuine vs Fraudulent Transaction Distribution (Cyber Donut).
    """
    if summary:
        genuine = summary.get('total_genuine', 6354407)
        fraud = summary.get('total_fraud', 8213)
    elif df is not None:
        fraud = int((df['isFraud'] == 1).sum())
        genuine = int(len(df) - fraud)
    else:
        genuine, fraud = 6354407, 8213
        
    total = genuine + fraud
    fraud_pct = (fraud / total) * 100
    
    fig = go.Figure(data=[go.Pie(
        labels=['Nominal Operations', 'Fraudulent Exfiltrations'],
        values=[genuine, fraud],
        hole=0.74,
        marker=dict(
            colors=[SENTINEL_THEME['cyan'], SENTINEL_THEME['crimson']],
            line=dict(color='#080C14', width=2)
        ),
        textinfo='percent',
        hoverinfo='label+value+percent',
        textfont=dict(family=SENTINEL_THEME['font'], size=12, color='#FFFFFF')
    )])
    
    fig.add_annotation(
        text=f"<b style='font-size:22px;color:#00F2FE;'>{fraud_pct:.3f}%</b><br><span style='font-size:10px;color:#64748B;font-weight:700;letter-spacing:1px;'>INCIDENCE</span>",
        x=0.5, y=0.5,
        font=dict(family=SENTINEL_THEME['font']),
        showarrow=False
    )
    
    return apply_sentinel_layout(fig, "Class Imbalance Breakdown", height=340)

def plot_fraud_by_type(summary: Optional[Dict[str, Any]] = None, df: Optional[pd.DataFrame] = None) -> go.Figure:
    """
    2. Fraud Transactions by Transaction Type (Glowing Gradient Bars).
    """
    if summary and 'transaction_types' in summary and 'fraud_by_type' in summary:
        types = list(summary['transaction_types'].keys())
        fraud_counts = [summary['fraud_by_type'].get(t, 0) for t in types]
    elif df is not None:
        grouped = df.groupby('type').agg(fraud=('isFraud', 'sum')).reset_index()
        types = grouped['type'].tolist()
        fraud_counts = grouped['fraud'].tolist()
    else:
        types = ['CASH_OUT', 'PAYMENT', 'CASH_IN', 'TRANSFER', 'DEBIT']
        fraud_counts = [4116, 0, 0, 4097, 0]
        
    fig = go.Figure()
    bar_colors = [SENTINEL_THEME['crimson'] if c > 0 else 'rgba(255,255,255,0.06)' for c in fraud_counts]
    
    fig.add_trace(go.Bar(
        x=types,
        y=fraud_counts,
        name='Confirmed Attack Vector',
        marker=dict(
            color=bar_colors,
            line=dict(color=[SENTINEL_THEME['crimson'] if c > 0 else 'rgba(255,255,255,0.1)' for c in fraud_counts], width=1)
        ),
        text=[f"{c:,}" for c in fraud_counts],
        textposition='outside',
        textfont=dict(color=SENTINEL_THEME['text'], size=11, family=SENTINEL_THEME['font'])
    ))
    
    return apply_sentinel_layout(fig, "Attack Incidence by Operation Type", height=340)

def plot_amount_distribution(df: pd.DataFrame) -> go.Figure:
    """
    3. Transaction Amount Spectrum (Log Transformed Boxplot).
    """
    df_plot = df[['amount', 'isFraud']].copy()
    df_plot['log_amount'] = np.log10(np.clip(df_plot['amount'], 1.0, None))
    
    fig = go.Figure()
    fig.add_trace(go.Box(
        x=df_plot[df_plot['isFraud'] == 0]['log_amount'],
        name='Nominal',
        marker_color=SENTINEL_THEME['cyan'],
        boxpoints='outliers',
        jitter=0.2,
        line=dict(width=1.5, color=SENTINEL_THEME['cyan'])
    ))
    fig.add_trace(go.Box(
        x=df_plot[df_plot['isFraud'] == 1]['log_amount'],
        name='Fraudulent',
        marker_color=SENTINEL_THEME['crimson'],
        boxpoints='outliers',
        jitter=0.2,
        line=dict(width=1.5, color=SENTINEL_THEME['crimson'])
    ))
    
    fig.update_xaxes(title_text="Log10(Transaction Volume)", tickvals=[1, 2, 3, 4, 5, 6, 7], ticktext=['$10', '$100', '$1K', '$10K', '$100K', '$1M', '$10M'])
    return apply_sentinel_layout(fig, "Capital Velocity Spectrum (Log Scale)", height=340)

def plot_fraud_trends_by_step(summary: Optional[Dict[str, Any]] = None, df: Optional[pd.DataFrame] = None) -> go.Figure:
    """
    4. Diurnal Hourly Trends (Glowing Cyber Curve).
    """
    if summary and 'hourly_fraud' in summary:
        hours = list(range(24))
        fraud_trend = [summary['hourly_fraud'].get(str(h), summary['hourly_fraud'].get(h, 0)) for h in hours]
    elif df is not None:
        df_copy = df.copy()
        df_copy['hour'] = df_copy['step'] % 24
        grouped = df_copy.groupby('hour').agg(fraud=('isFraud', 'sum')).reset_index()
        hours = grouped['hour'].tolist()
        fraud_trend = grouped['fraud'].tolist()
    else:
        hours = list(range(24))
        fraud_trend = [340 + int(20 * np.sin(h)) for h in hours]
        
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hours,
        y=fraud_trend,
        mode='lines+markers',
        name='Attack Volume',
        line=dict(color=SENTINEL_THEME['crimson'], width=2.5, shape='spline'),
        marker=dict(size=6, color=SENTINEL_THEME['crimson'])
    ))
    
    fig.update_xaxes(title_text="Hour of Day (00:00 - 23:00 UTC)", dtick=2)
    fig.update_yaxes(title_text="Attack Frequency")
    return apply_sentinel_layout(fig, "24-Hour Diurnal Threat Rhythm", height=340)

def plot_balance_analysis(df: pd.DataFrame) -> go.Figure:
    """
    5. Balance Dynamics (Sender Liquidation vs Mule Influx).
    """
    sample_vis = df.sample(n=min(len(df), 2000), random_state=42).copy()
    sample_vis['orig_drain'] = sample_vis['oldbalanceOrg'] - sample_vis['newbalanceOrig']
    sample_vis['dest_gain'] = sample_vis['newbalanceDest'] - sample_vis['oldbalanceDest']
    
    fig = go.Figure()
    gen = sample_vis[sample_vis['isFraud'] == 0]
    fig.add_trace(go.Scatter(
        x=np.log10(np.clip(gen['orig_drain'], 1.0, None)),
        y=np.log10(np.clip(gen['dest_gain'], 1.0, None)),
        mode='markers',
        name='Nominal',
        marker=dict(color=SENTINEL_THEME['cyan'], size=4, opacity=0.35)
    ))
    
    fra = sample_vis[sample_vis['isFraud'] == 1]
    fig.add_trace(go.Scatter(
        x=np.log10(np.clip(fra['orig_drain'], 1.0, None)),
        y=np.log10(np.clip(fra['dest_gain'], 1.0, None)),
        mode='markers',
        name='Fraudulent',
        marker=dict(color=SENTINEL_THEME['crimson'], size=7, symbol='diamond', opacity=0.95)
    ))
    
    fig.update_xaxes(title_text="Log10(Sender Capital Drain)")
    fig.update_yaxes(title_text="Log10(Recipient Inflow Gain)")
    return apply_sentinel_layout(fig, "Ledger Mutation Dynamics", height=380)

def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    6. Feature Correlation Matrix (Obsidian Gradient).
    """
    num_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'isFraud']
    avail = [c for c in num_cols if c in df.columns]
    corr = df[avail].corr().round(2)
    
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale=[
            [0.0, '#090D16'],
            [0.3, '#1E293B'],
            [0.7, '#0284C7'],
            [1.0, '#00F2FE']
        ],
        text=corr.values,
        texttemplate="%{text}",
        textfont=dict(family=SENTINEL_THEME['font'], size=11, color='#FFFFFF'),
        colorbar=dict(
            title=dict(text='Corr', font=dict(color=SENTINEL_THEME['text_muted'], size=10)),
            tickfont=dict(color=SENTINEL_THEME['text_muted'], size=9)
        )
    ))
    return apply_sentinel_layout(fig, "Telemetry Correlation Heatmap", height=380)

def plot_model_comparison(metrics_payload: Dict[str, Any]) -> go.Figure:
    """
    7. Multi-Algorithm Benchmark Comparison.
    """
    models_dict = metrics_payload.get('models', {})
    model_names = list(models_dict.keys())
    
    metric_keys = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    display_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    colors = [SENTINEL_THEME['text_muted'], SENTINEL_THEME['violet'], SENTINEL_THEME['emerald'], SENTINEL_THEME['cyan'], SENTINEL_THEME['blue']]
    
    fig = go.Figure()
    for i, (m_key, d_name) in enumerate(zip(metric_keys, display_names)):
        values = [round(models_dict[m].get(m_key, 0.0) * 100, 2) for m in model_names]
        fig.add_trace(go.Bar(
            name=d_name,
            x=model_names,
            y=values,
            marker_color=colors[i],
            text=[f"{v:.1f}%" for v in values],
            textposition='outside',
            textfont=dict(size=9, color=SENTINEL_THEME['text'])
        ))
        
    fig.update_layout(barmode='group')
    fig.update_yaxes(range=[0, 115], title_text="Score (%)")
    return apply_sentinel_layout(fig, "Algorithm Evaluation Benchmark", height=420)

def plot_confusion_matrix(cm_data: list, model_name: str = "Best Model") -> go.Figure:
    """
    8. Interactive Confusion Matrix (Obsidian Slate).
    """
    z_matrix = cm_data
    labels = ['Nominal (0)', 'Fraud (1)']
    
    tn, fp = z_matrix[0][0], z_matrix[0][1]
    fn, tp = z_matrix[1][0], z_matrix[1][1]
    
    annot_text = [
        [f"TN: {tn:,}<br><span style='font-size:10px;color:#64748B'>TRUE NEGATIVE</span>",
         f"FP: {fp:,}<br><span style='font-size:10px;color:#FF3366'>FALSE POSITIVE</span>"],
        [f"FN: {fn:,}<br><span style='font-size:10px;color:#FF3366'>FALSE NEGATIVE</span>",
         f"TP: {tp:,}<br><span style='font-size:10px;color:#00F5A0'>TRUE POSITIVE</span>"]
    ]
    
    fig = go.Figure(data=go.Heatmap(
        z=z_matrix,
        x=labels,
        y=labels,
        colorscale=[
            [0.0, '#0E1626'],
            [0.5, '#1E293B'],
            [1.0, '#00F2FE']
        ],
        text=annot_text,
        texttemplate="%{text}",
        textfont=dict(family=SENTINEL_THEME['font'], size=12, color='#FFFFFF'),
        showscale=False
    ))
    
    fig.update_xaxes(title_text="Predicted Telemetry", side="bottom")
    fig.update_yaxes(title_text="Ground Truth State", autorange="reversed")
    return apply_sentinel_layout(fig, f"Confusion Matrix: {model_name}", height=380)

def plot_roc_curves(roc_data: Dict[str, Any]) -> go.Figure:
    """
    9. ROC Discrimination Curves.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Chance Line (AUC = 0.50)',
        line=dict(color='rgba(255,255,255,0.15)', dash='dash', width=1.5)
    ))
    
    colors = {
        'Logistic Regression': SENTINEL_THEME['text_muted'],
        'Decision Tree': SENTINEL_THEME['blue'],
        'Random Forest': SENTINEL_THEME['cyan'],
        'Isolation Forest': SENTINEL_THEME['violet']
    }
    
    for m_name, curve in roc_data.items():
        fpr = curve.get('fpr', [0, 1])
        tpr = curve.get('tpr', [0, 1])
        c = colors.get(m_name, SENTINEL_THEME['cyan'])
        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=m_name,
            line=dict(color=c, width=2.5)
        ))
        
    fig.update_xaxes(title_text="False Positive Rate (1 - Specificity)", range=[-0.02, 1.02])
    fig.update_yaxes(title_text="True Positive Rate (Sensitivity)", range=[-0.02, 1.05])
    return apply_sentinel_layout(fig, "ROC Curves (Discrimination Capacity)", height=400)

def plot_fraud_risk_gauge(probability: float) -> go.Figure:
    """
    10. Sentinel Cyber Threat Dial.
    """
    pct = probability * 100
    if probability < 0.30:
        bar_color = SENTINEL_THEME['emerald']
        level_text = "NOMINAL // LOW RISK"
    elif probability < 0.70:
        bar_color = SENTINEL_THEME['amber']
        level_text = "ELEVATED // MEDIUM RISK"
    else:
        bar_color = SENTINEL_THEME['crimson']
        level_text = "CRITICAL // HIGH RISK"
        
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={'suffix': "%", 'font': {'color': SENTINEL_THEME['text'], 'family': SENTINEL_THEME['font'], 'size': 32, 'weight': 700}},
        title={'text': f"THREAT LEVEL: {level_text}", 'font': {'color': bar_color, 'family': SENTINEL_THEME['font'], 'size': 12, 'weight': 700}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': SENTINEL_THEME['border'], 'tickfont': {'color': SENTINEL_THEME['text_muted'], 'size': 10}},
            'bar': {'color': bar_color, 'thickness': 0.28},
            'bgcolor': 'rgba(15, 23, 42, 0.6)',
            'borderwidth': 1,
            'bordercolor': SENTINEL_THEME['border'],
            'steps': [
                {'range': [0, 30], 'color': 'rgba(16, 185, 129, 0.12)'},
                {'range': [30, 70], 'color': 'rgba(245, 158, 11, 0.12)'},
                {'range': [70, 100], 'color': 'rgba(239, 68, 68, 0.15)'}
            ],
            'threshold': {
                'line': {'color': '#FFFFFF', 'width': 3},
                'thickness': 0.8,
                'value': pct
            }
        }
    ))
    return apply_sentinel_layout(fig, "Threat Probability Vector", height=280)
