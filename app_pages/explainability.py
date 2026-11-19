"""
Page 6: Model Explainability with SHAP (SHapley Additive exPlanations).
Surfaces global feature attribution, inspects model mechanics,
and provides transparent, non-technical business interpretations.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def render_explainability_page(shap_data: dict, model_metadata: dict):
    """Renders the SHAP-based Model Explainability interface."""
    st.markdown("""
        <div style="margin-bottom: 24px;">
            <h2 style="color: #0F172A; font-weight: 700; margin-bottom: 6px;">Model Explainability & Feature Attribution</h2>
            <p style="color: #64748B; font-size: 15px;">
                Using cooperative game theory via SHAP (SHapley Additive exPlanations) to interpret feature contributions without black-box opacity.
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.info("""
        **Methodological Note:** SHAP values calculate the marginal contribution of each attribute towards shifting the model's churn probability from the base rate. 
        High attribution indicates strong predictive influence, not necessarily direct causal causation.
    """)

    if not shap_data or 'feature_importance_df' not in shap_data:
        st.warning("SHAP explanation data is not available. Please run `python -m src.train`.")
        return

    feat_df = pd.DataFrame(shap_data['feature_importance_df'])

    # Global Feature Importance Bar Chart
    st.markdown("### 1. Global Feature Importance (Mean |SHAP Value|)")
    st.markdown("Features at the top exhibit the strongest aggregate impact on deciding whether a customer stays or churns.")

    top_n = st.slider("Number of Top Features to Display", min_value=5, max_value=25, value=12)
    top_feats = feat_df.head(top_n).sort_values(by='Mean_SHAP_Value', ascending=True)

    fig_imp = px.bar(
        top_feats,
        x='Mean_SHAP_Value',
        y='Feature',
        orientation='h',
        color='Mean_SHAP_Value',
        color_continuous_scale='Blues',
        text=top_feats['Mean_SHAP_Value'].apply(lambda v: f"{v:.3f}")
    )
    fig_imp.update_layout(
        xaxis_title="Mean |SHAP Value| (Average Absolute Impact on Churn Log-Odds)",
        yaxis_title="",
        height=420,
        margin=dict(t=20, b=20, l=10, r=20),
        coloraxis_showscale=False
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("---")

    # Business Insights breakdown
    st.markdown("### 2. Strategic Factor Dynamics & Business Interpretations")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
            <div style="background-color: #FEF2F2; border-left: 4px solid #EF4444; padding: 16px; border-radius: 8px;">
                <h4 style="color: #DC2626; margin-top: 0;">Top Churn Accelerators (Risk Amplifiers)</h4>
                <ul style="color: #475569; font-size: 14px; line-height: 1.6; margin-bottom: 0;">
                    <li><strong>Month-to-Month Contract:</strong> Creates minimal switching friction, leaving customers susceptible to competitor promotions.</li>
                    <li><strong>Fiber Optic without Add-on Support:</strong> High monthly price point ($70–$105/mo) causes dissatisfaction if outages or configuration issues occur without premium tech support.</li>
                    <li><strong>Electronic Check Payment:</strong> Manual monthly action reminds customers of their recurring expense and lacks auto-renewal stickiness.</li>
                    <li><strong>Low Tenure (0–12 Months):</strong> Customers in early lifecycle lack brand loyalty and established product routine.</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
            <div style="background-color: #ECFDF5; border-left: 4px solid #10B981; padding: 16px; border-radius: 8px;">
                <h4 style="color: #059669; margin-top: 0;">Top Retention Drivers (Loyalty Stabilizers)</h4>
                <ul style="color: #475569; font-size: 14px; line-height: 1.6; margin-bottom: 0;">
                    <li><strong>Two-Year Contract Commitment:</strong> Highest single stabilizing factor; churn rates drop below 3%.</li>
                    <li><strong>Online Security & Tech Support:</strong> Subscribing to value-added security creates strong product dependency.</li>
                    <li><strong>Automated Bank Transfer / Credit Card:</strong> Frictionless, out-of-sight billing reduces payment delinquency and voluntary churn.</li>
                    <li><strong>High Tenure (48+ Months):</strong> Entrenched habits and relationship depth create strong retention inertia.</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    # Static High-Res Report Figure if available
    st.markdown("<br>", unsafe_allow_html=True)
    report_img_path = "reports/figures/feature_importance.png"
    if os.path.exists(report_img_path):
        with st.expander("View Publication-Quality Generated Figure (Static 300 DPI)", expanded=False):
            st.image(report_img_path, caption="Exported Academic Feature Importance Chart", use_container_width=True)
