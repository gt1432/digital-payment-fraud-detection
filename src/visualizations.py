"""
Digital Payment Fraud Detection - Visualization Module
SpaceX-Inspired Dark Minimalist Theme (#000000)
Author: Senior ML Engineer & UI/UX Designer
"""
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

SPACEX_THEME = {
    'bg': '#000000',
    'card_bg': '#0A0A0A',
    'grid': '#1A1A1A',
    'border': '#262626',
    'text': '#FFFFFF',
    'text_muted': '#8E8E93',
    'white': '#FFFFFF',
    'red': '#FF453A',
    'green': '#30D158',
    'orange': '#FF9F0A',
    'blue': '#0A84FF',
    'font': 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
}

def apply_spacex_layout(fig: go.Figure, title: str = "", height: int = 400) -> go.Figure:
    """
    Applies pure black SpaceX industrial minimalist aesthetics to any Plotly figure.
    """
    fig.update_layout(
        title={
            'text': title.upper(),
            'y': 0.95,
            'x': 0.05,
            'xanchor': 'left',
            'yanchor': 'top',
            'font': {
                'family': SPACEX_THEME['font'],
                'size': 13,
                'color': SPACEX_THEME['text'],
                'weight': 700
            }
        },
        paper_bgcolor=SPACEX_THEME['bg'],
        plot_bgcolor=SPACEX_THEME['bg'],
        font=dict(family=SPACEX_THEME['font'], color=SPACEX_THEME['text'], size=11),
        margin=dict(l=40, r=40, t=60, b=40),
        height=height,
        hovermode='closest',
        hoverlabel=dict(
            bgcolor='#141414',
            bordercolor=SPACEX_THEME['border'],
            font=dict(color=SPACEX_THEME['text'], family=SPACEX_THEME['font'], size=12)
        ),
        legend=dict(
            font=dict(color=SPACEX_THEME['text_muted'], size=11),
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
        gridcolor=SPACEX_THEME['grid'],
        gridwidth=1,
        zeroline=False,
        linecolor=SPACEX_THEME['border'],
        tickfont=dict(color=SPACEX_THEME['text_muted'], size=10)
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=SPACEX_THEME['grid'],
        gridwidth=1,
        zeroline=False,
        linecolor=SPACEX_THEME['border'],
        tickfont=dict(color=SPACEX_THEME['text_muted'], size=10)
    )
    return fig

def plot_class_distribution(summary: Optional[Dict[str, Any]] = None, df: Optional[pd.DataFrame] = None) -> go.Figure:
    """
    1. Genuine vs Fraudulent Transaction Distribution (SpaceX Donut Chart).
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
    genuine_pct = (genuine / total) * 100
    
    fig = go.Figure(data=[go.Pie(
        labels=['GENUINE TRANSACTIONS', 'FRAUDULENT TRANSACTIONS'],
        values=[genuine, fraud],
        hole=0.68,
        marker=dict(
            colors=['#FFFFFF', SPACEX_THEME['red']],
            line=dict(color=SPACEX_THEME['bg'], width=2)
        ),
        textinfo='percent',
        hoverinfo='label+value+percent',
        textfont=dict(family=SPACEX_THEME['font'], size=12, color='#000000')
    )])
    
    fig.add_annotation(
        text=f"<b>{fraud_pct:.3f}%</b><br><span style='font-size:10px;color:#8E8E93;'>FRAUD RATE</span>",
        x=0.5, y=0.5,
        font=dict(size=18, color=SPACEX_THEME['text'], family=SPACEX_THEME['font']),
        showarrow=False
    )
    
    return apply_spacex_layout(fig, "CLASS DISTRIBUTION (GENUINE VS FRAUD)", height=350)

def plot_fraud_by_type(summary: Optional[Dict[str, Any]] = None, df: Optional[pd.DataFrame] = None) -> go.Figure:
    """
    2. Fraud Transactions by Transaction Type (Stacked / Grouped Bar).
    """
    if summary and 'transaction_types' in summary and 'fraud_by_type' in summary:
        types = list(summary['transaction_types'].keys())
        total_counts = [summary['transaction_types'][t] for t in types]
        fraud_counts = [summary['fraud_by_type'].get(t, 0) for t in types]
    elif df is not None:
        grouped = df.groupby('type').agg(total=('isFraud', 'count'), fraud=('isFraud', 'sum')).reset_index()
        types = grouped['type'].tolist()
        total_counts = grouped['total'].tolist()
        fraud_counts = grouped['fraud'].tolist()
    else:
        types = ['CASH_OUT', 'PAYMENT', 'CASH_IN', 'TRANSFER', 'DEBIT']
        total_counts = [2237500, 2151495, 1399284, 532909, 41432]
        fraud_counts = [4116, 0, 0, 4097, 0]
        
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=types,
        y=fraud_counts,
        name='CONFIRMED FRAUD',
        marker=dict(color=SPACEX_THEME['red'], line=dict(color=SPACEX_THEME['red'], width=1)),
        text=[f"{c:,}" for c in fraud_counts],
        textposition='outside',
        textfont=dict(color=SPACEX_THEME['text'], size=10)
    ))
    
    return apply_spacex_layout(fig, "FRAUD INCIDENCE BY TRANSACTION TYPE", height=350)

def plot_amount_distribution(df: pd.DataFrame) -> go.Figure:
    """
    3. Transaction Amount Distribution (Log transformed comparison).
    """
    df_plot = df[['amount', 'isFraud']].copy()
    df_plot['log_amount'] = np.log10(np.clip(df_plot['amount'], 1.0, None))
    df_plot['Label'] = df_plot['isFraud'].map({0: 'GENUINE', 1: 'FRAUDULENT'})
    
    fig = go.Figure()
    fig.add_trace(go.Box(
        x=df_plot[df_plot['isFraud'] == 0]['log_amount'],
        name='GENUINE',
        marker_color='#FFFFFF',
        boxpoints='outliers',
        jitter=0.2,
        line=dict(width=1.2, color='#FFFFFF')
    ))
    fig.add_trace(go.Box(
        x=df_plot[df_plot['isFraud'] == 1]['log_amount'],
        name='FRAUDULENT',
        marker_color=SPACEX_THEME['red'],
        boxpoints='outliers',
        jitter=0.2,
        line=dict(width=1.2, color=SPACEX_THEME['red'])
    ))
    
    fig.update_xaxes(title_text="LOG10(AMOUNT IN USD)", tickvals=[1, 2, 3, 4, 5, 6, 7], ticktext=['$10', '$100', '$1K', '$10K', '$100K', '$1M', '$10M'])
    return apply_spacex_layout(fig, "TRANSACTION VALUE DISTRIBUTION (LOG SCALE)", height=350)

def plot_fraud_trends_by_step(summary: Optional[Dict[str, Any]] = None, df: Optional[pd.DataFrame] = None) -> go.Figure:
    """
    4. Fraud Trends Across Steps / Hours of Day.
    """
    if summary and 'hourly_fraud' in summary:
        hours = list(range(24))
        fraud_trend = [summary['hourly_fraud'].get(str(h), summary['hourly_fraud'].get(h, 0)) for h in hours]
        total_trend = [summary['hourly_total'].get(str(h), summary['hourly_total'].get(h, 0)) for h in hours]
    elif df is not None:
        df_copy = df.copy()
        df_copy['hour'] = df_copy['step'] % 24
        grouped = df_copy.groupby('hour').agg(total=('isFraud', 'count'), fraud=('isFraud', 'sum')).reset_index()
        hours = grouped['hour'].tolist()
        fraud_trend = grouped['fraud'].tolist()
        total_trend = grouped['total'].tolist()
    else:
        hours = list(range(24))
        fraud_trend = [340 + int(20 * np.sin(h)) for h in hours]
        total_trend = [250000 + int(50000 * np.sin(h)) for h in hours]
        
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hours,
        y=fraud_trend,
        mode='lines+markers',
        name='FRAUD CASES',
        line=dict(color=SPACEX_THEME['red'], width=2),
        marker=dict(size=5, color=SPACEX_THEME['red'])
    ))
    
    fig.update_xaxes(title_text="HOUR OF DAY (0 - 23 UTC)", dtick=2)
    fig.update_yaxes(title_text="FRAUDULENT COUNT")
    return apply_spacex_layout(fig, "DIURNAL FRAUD FREQUENCY SPECTRUM", height=350)

def plot_balance_analysis(df: pd.DataFrame) -> go.Figure:
    """
    5. Sender & Receiver Balance Discrepancy Scatter.
    """
    sample_vis = df.sample(n=min(len(df), 2000), random_state=42).copy()
    sample_vis['orig_drain'] = sample_vis['oldbalanceOrg'] - sample_vis['newbalanceOrig']
    sample_vis['dest_gain'] = sample_vis['newbalanceDest'] - sample_vis['oldbalanceDest']
    
    fig = go.Figure()
    
    # Genuine points
    gen = sample_vis[sample_vis['isFraud'] == 0]
    fig.add_trace(go.Scatter(
        x=np.log10(np.clip(gen['orig_drain'], 1.0, None)),
        y=np.log10(np.clip(gen['dest_gain'], 1.0, None)),
        mode='markers',
        name='GENUINE',
        marker=dict(color='#FFFFFF', size=3, opacity=0.3)
    ))
    
    # Fraud points
    fra = sample_vis[sample_vis['isFraud'] == 1]
    fig.add_trace(go.Scatter(
        x=np.log10(np.clip(fra['orig_drain'], 1.0, None)),
        y=np.log10(np.clip(fra['dest_gain'], 1.0, None)),
        mode='markers',
        name='FRAUDULENT',
        marker=dict(color=SPACEX_THEME['red'], size=6, symbol='x', opacity=0.9)
    ))
    
    fig.update_xaxes(title_text="LOG10(SENDER BALANCE DRAIN)")
    fig.update_yaxes(title_text="LOG10(RECEIVER BALANCE GAIN)")
    return apply_spacex_layout(fig, "BALANCE DYNAMICS: SENDER DRAIN VS RECIPIENT ACCUMULATION", height=380)

def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    6. Feature Correlation Heatmap (Industrial Monochrome).
    """
    num_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'isFraud']
    avail = [c for c in num_cols if c in df.columns]
    corr = df[avail].corr().round(2)
    
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale=[
            [0.0, '#000000'],
            [0.5, '#262626'],
            [0.8, '#8E8E93'],
            [1.0, '#FFFFFF']
        ],
        text=corr.values,
        texttemplate="%{text}",
        textfont=dict(family=SPACEX_THEME['font'], size=11, color='#FFFFFF'),
        colorbar=dict(
            title=dict(text='CORR', font=dict(color=SPACEX_THEME['text_muted'], size=10)),
            tickfont=dict(color=SPACEX_THEME['text_muted'], size=9)
        )
    ))
    return apply_spacex_layout(fig, "TELEMETRY CORRELATION MATRIX", height=380)

def plot_model_comparison(metrics_payload: Dict[str, Any]) -> go.Figure:
    """
    7. Multi-Algorithm Performance Comparison Matrix.
    """
    models_dict = metrics_payload.get('models', {})
    model_names = list(models_dict.keys())
    
    metric_keys = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    display_names = ['ACCURACY', 'PRECISION', 'RECALL', 'F1-SCORE', 'ROC-AUC']
    colors = ['#FFFFFF', '#8E8E93', SPACEX_THEME['green'], SPACEX_THEME['red'], SPACEX_THEME['blue']]
    
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
            textfont=dict(size=9, color=SPACEX_THEME['text'])
        ))
        
    fig.update_layout(barmode='group')
    fig.update_yaxes(range=[0, 115], title_text="PERCENT (%)")
    return apply_spacex_layout(fig, "MULTI-ALGORITHM EVALUATION BENCHMARK", height=420)

def plot_confusion_matrix(cm_data: list, model_name: str = "BEST MODEL") -> go.Figure:
    """
    8. Interactive Annotated Confusion Matrix Heatmap.
    """
    z_matrix = cm_data
    labels = ['GENUINE (0)', 'FRAUD (1)']
    
    tn, fp = z_matrix[0][0], z_matrix[0][1]
    fn, tp = z_matrix[1][0], z_matrix[1][1]
    
    annot_text = [
        [f"TN: {tn:,}<br><span style='font-size:10px;color:#8E8E93'>TRUE NEGATIVE</span>",
         f"FP: {fp:,}<br><span style='font-size:10px;color:#8E8E93'>FALSE POSITIVE</span>"],
        [f"FN: {fn:,}<br><span style='font-size:10px;color:#8E8E93'>FALSE NEGATIVE</span>",
         f"TP: {tp:,}<br><span style='font-size:10px;color:#8E8E93'>TRUE POSITIVE</span>"]
    ]
    
    fig = go.Figure(data=go.Heatmap(
        z=z_matrix,
        x=labels,
        y=labels,
        colorscale=[
            [0.0, '#0A0A0A'],
            [0.5, '#1F1F1F'],
            [1.0, '#FFFFFF']
        ],
        text=annot_text,
        texttemplate="%{text}",
        textfont=dict(family=SPACEX_THEME['font'], size=12, color='#FFFFFF'),
        showscale=False
    ))
    
    fig.update_xaxes(title_text="PREDICTED CLASS", side="bottom")
    fig.update_yaxes(title_text="ACTUAL CLASS", autorange="reversed")
    return apply_spacex_layout(fig, f"CONFUSION MATRIX: {model_name.upper()}", height=380)

def plot_roc_curves(roc_data: Dict[str, Any]) -> go.Figure:
    """
    9. Receiver Operating Characteristic (ROC) Comparison Curves.
    """
    fig = go.Figure()
    
    # 45-degree chance line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='RANDOM GUESS (AUC = 0.50)',
        line=dict(color='#333333', dash='dash', width=1.5)
    ))
    
    colors = {
        'Logistic Regression': '#8E8E93',
        'Decision Tree': SPACEX_THEME['blue'],
        'Random Forest': '#FFFFFF',
        'Isolation Forest': SPACEX_THEME['orange']
    }
    
    for model_name, curve in roc_data.items():
        fpr = curve.get('fpr', [0, 1])
        tpr = curve.get('tpr', [0, 1])
        c = colors.get(model_name, '#FFFFFF')
        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=model_name.upper(),
            line=dict(color=c, width=2)
        ))
        
    fig.update_xaxes(title_text="FALSE POSITIVE RATE (1 - SPECIFICITY)", range=[-0.02, 1.02])
    fig.update_yaxes(title_text="TRUE POSITIVE RATE (RECALL)", range=[-0.02, 1.05])
    return apply_spacex_layout(fig, "ROC CURVES (DISCRIMINATION ABILITY)", height=400)

def plot_fraud_risk_gauge(probability: float) -> go.Figure:
    """
    10. Minimalist Industrial Risk Gauge.
    """
    pct = probability * 100
    if probability < 0.30:
        bar_color = SPACEX_THEME['green']
        level_text = "LOW RISK"
    elif probability < 0.70:
        bar_color = SPACEX_THEME['orange']
        level_text = "MEDIUM RISK"
    else:
        bar_color = SPACEX_THEME['red']
        level_text = "CRITICAL / HIGH RISK"
        
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={'suffix': "%", 'font': {'color': SPACEX_THEME['text'], 'family': SPACEX_THEME['font'], 'size': 32, 'weight': 700}},
        title={'text': f"THREAT VECTOR: {level_text}", 'font': {'color': bar_color, 'family': SPACEX_THEME['font'], 'size': 12, 'weight': 700}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': SPACEX_THEME['border'], 'tickfont': {'color': SPACEX_THEME['text_muted'], 'size': 10}},
            'bar': {'color': bar_color, 'thickness': 0.25},
            'bgcolor': '#0A0A0A',
            'borderwidth': 1,
            'bordercolor': SPACEX_THEME['border'],
            'steps': [
                {'range': [0, 30], 'color': '#051A0B'},
                {'range': [30, 70], 'color': '#1F1405'},
                {'range': [70, 100], 'color': '#210808'}
            ],
            'threshold': {
                'line': {'color': SPACEX_THEME['white'], 'width': 3},
                'thickness': 0.8,
                'value': pct
            }
        }
    ))
    return apply_spacex_layout(fig, "COMPUTED FRAUD PROBABILITY", height=280)
