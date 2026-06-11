"""
============================================================
AIRLINE LOYALTY — RETENTION INTELLIGENCE DASHBOARD
============================================================
A Streamlit prototype for non-technical marketing managers.
Powered by the final_customer_intelligence.csv output from
the Behavioral Intelligence Framework pipeline.

Run:  streamlit run app/app.py
============================================================
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ──────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Airline Loyalty — Retention Intelligence",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/final_customer_intelligence.csv")
    df["Segment_Name"] = df["Segment_Name"].fillna("Micro-Segment Outliers")
    # Ensure categorical ordering for risk
    risk_order = ["Low", "Medium", "High", "Critical"]
    df["Risk_Level"] = pd.Categorical(df["Risk_Level"], categories=risk_order, ordered=True)
    return df

df = load_data()

# ──────────────────────────────────────────────────────────
# COLOR PALETTES (consulting-style)
# ──────────────────────────────────────────────────────────
SEG_COLORS = {
    "Champions": "#2E86AB",
    "At-Risk Loyalists": "#F6AE2D",
    "Dormant Members": "#F26419",
    "Micro-Segment Outliers": "#6C757D",
}
RISK_COLORS = {
    "Low": "#4CAF50",
    "Medium": "#FFC107",
    "High": "#FF9800",
    "Critical": "#F44336",
}

# ──────────────────────────────────────────────────────────
# SIDEBAR FILTERS
# ──────────────────────────────────────────────────────────
st.sidebar.title("🔍 Filters")

seg_filter = st.sidebar.multiselect(
    "Segment",
    options=sorted(df["Segment_Name"].dropna().unique()),
    default=sorted(df["Segment_Name"].dropna().unique()),
)
risk_filter = st.sidebar.multiselect(
    "Risk Level",
    options=["Low", "Medium", "High", "Critical"],
    default=["Low", "Medium", "High", "Critical"],
)
min_clv = st.sidebar.slider(
    "Minimum CLV ($)", 0, int(df["CLV"].max()), 0, step=500
)
min_rar = st.sidebar.slider(
    "Minimum Revenue at Risk ($)", 0, int(df["Revenue_At_Risk"].max()), 0, step=100
)
action_filter = st.sidebar.multiselect(
    "Recommended Action",
    options=sorted(df["Recommended_Action"].unique()),
    default=sorted(df["Recommended_Action"].unique()),
)

# Apply filters
mask = (
    df["Segment_Name"].isin(seg_filter)
    & df["Risk_Level"].isin(risk_filter)
    & (df["CLV"] >= min_clv)
    & (df["Revenue_At_Risk"] >= min_rar)
    & df["Recommended_Action"].isin(action_filter)
)
filtered = df[mask].copy()

# ──────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────
st.title("✈️ Airline Loyalty — Retention Intelligence")
st.caption(
    "An operational dashboard for identifying at-risk loyalty members, "
    "understanding revenue exposure, and taking targeted retention actions."
)
st.markdown("---")

# ──────────────────────────────────────────────────────────
# 1. EXECUTIVE KPI CARDS
# ──────────────────────────────────────────────────────────
st.subheader("📊 Executive Summary")

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric("Total Customers", f"{len(filtered):,}")
high_crit = filtered[filtered["Risk_Level"].isin(["High", "Critical"])].shape[0]
kpi2.metric("High + Critical Risk", f"{high_crit:,}")
kpi3.metric("Revenue at Risk", f"${filtered['Revenue_At_Risk'].sum():,.0f}")
kpi4.metric("Avg Churn Probability", f"{filtered['Churn_Prob'].mean():.1%}")
kpi5.metric("Avg CLV", f"${filtered['CLV'].mean():,.0f}")

st.markdown("---")

# ──────────────────────────────────────────────────────────
# 2. SEGMENT OVERVIEW
# ──────────────────────────────────────────────────────────
st.subheader("🏷️ Segment Overview")

seg_stats = (
    filtered.groupby("Segment_Name")
    .agg(
        Customers=("Loyalty Number", "count"),
        Avg_CLV=("CLV", "mean"),
        Avg_Churn_Prob=("Churn_Prob", "mean"),
        Revenue_At_Risk=("Revenue_At_Risk", "sum"),
    )
    .reset_index()
)

col_a, col_b = st.columns(2)

with col_a:
    fig = px.bar(
        seg_stats,
        x="Segment_Name",
        y="Customers",
        color="Segment_Name",
        color_discrete_map=SEG_COLORS,
        title="Customers by Segment",
        text="Customers",
    )
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)

with col_b:
    fig = px.bar(
        seg_stats,
        x="Segment_Name",
        y="Revenue_At_Risk",
        color="Segment_Name",
        color_discrete_map=SEG_COLORS,
        title="Revenue at Risk by Segment ($)",
        text=seg_stats["Revenue_At_Risk"].apply(lambda x: f"${x/1e6:.2f}M"),
    )
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="$")
    st.plotly_chart(fig, use_container_width=True)

col_c, col_d = st.columns(2)

with col_c:
    fig = px.bar(
        seg_stats,
        x="Segment_Name",
        y="Avg_CLV",
        color="Segment_Name",
        color_discrete_map=SEG_COLORS,
        title="Average CLV by Segment ($)",
        text=seg_stats["Avg_CLV"].apply(lambda x: f"${x:,.0f}"),
    )
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="$")
    st.plotly_chart(fig, use_container_width=True)

with col_d:
    fig = px.bar(
        seg_stats,
        x="Segment_Name",
        y="Avg_Churn_Prob",
        color="Segment_Name",
        color_discrete_map=SEG_COLORS,
        title="Average Churn Probability by Segment",
        text=seg_stats["Avg_Churn_Prob"].apply(lambda x: f"{x:.1%}"),
    )
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Probability")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ──────────────────────────────────────────────────────────
# 3. RISK OVERVIEW
# ──────────────────────────────────────────────────────────
st.subheader("⚠️ Risk Overview")

risk_order = ["Low", "Medium", "High", "Critical"]
risk_stats = (
    filtered.groupby("Risk_Level", observed=False)
    .agg(
        Customers=("Loyalty Number", "count"),
        Revenue_At_Risk=("Revenue_At_Risk", "sum"),
    )
    .reindex(risk_order)
    .reset_index()
)

col_e, col_f = st.columns(2)

with col_e:
    fig = px.bar(
        risk_stats,
        x="Risk_Level",
        y="Customers",
        color="Risk_Level",
        color_discrete_map=RISK_COLORS,
        title="Customers by Risk Level",
        text="Customers",
        category_orders={"Risk_Level": risk_order},
    )
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)

with col_f:
    fig = px.bar(
        risk_stats,
        x="Risk_Level",
        y="Revenue_At_Risk",
        color="Risk_Level",
        color_discrete_map=RISK_COLORS,
        title="Revenue at Risk by Risk Level ($)",
        text=risk_stats["Revenue_At_Risk"].apply(lambda x: f"${x/1e6:.2f}M"),
        category_orders={"Risk_Level": risk_order},
    )
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="$")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ──────────────────────────────────────────────────────────
# 4. ACTION CENTER
# ──────────────────────────────────────────────────────────
st.subheader("🎯 Action Center")

action_stats = (
    filtered.groupby("Recommended_Action")
    .agg(
        Customers=("Loyalty Number", "count"),
        Avg_Churn_Prob=("Churn_Prob", "mean"),
        Total_RAR=("Revenue_At_Risk", "sum"),
    )
    .sort_values("Total_RAR", ascending=False)
    .reset_index()
)

fig = px.bar(
    action_stats,
    y="Recommended_Action",
    x="Customers",
    orientation="h",
    color="Total_RAR",
    color_continuous_scale="OrRd",
    title="Recommended Actions — Customer Count & Revenue Exposure",
    text="Customers",
)
fig.update_layout(yaxis_title="", xaxis_title="Customers", height=400)
st.plotly_chart(fig, use_container_width=True)

# Detailed action table — who, why, what, urgency
st.markdown("**Detailed Action Breakdown — Who, Why, What, Urgency**")
action_detail = (
    filtered[
        [
            "Loyalty Number",
            "Segment_Name",
            "Risk_Level",
            "Churn_Prob",
            "CLV",
            "Revenue_At_Risk",
            "Months_Since_Last_Active",
            "Recommended_Action",
        ]
    ]
    .sort_values("Revenue_At_Risk", ascending=False)
    .head(100)
)
st.dataframe(
    action_detail.style.format(
        {
            "Churn_Prob": "{:.1%}",
            "CLV": "${:,.0f}",
            "Revenue_At_Risk": "${:,.0f}",
            "Months_Since_Last_Active": "{:.1f} mo",
        }
    ),
    use_container_width=True,
    height=400,
)

st.markdown("---")

# ──────────────────────────────────────────────────────────
# 5. TOP 20 PRIORITY CUSTOMERS
# ──────────────────────────────────────────────────────────
st.subheader("🔥 Top 20 Priority Customers (by Revenue at Risk)")

top20 = filtered.nlargest(20, "Revenue_At_Risk")[
    [
        "Loyalty Number",
        "Segment_Name",
        "Risk_Level",
        "Churn_Prob",
        "CLV",
        "Revenue_At_Risk",
        "Recommended_Action",
    ]
]
st.dataframe(
    top20.style.format(
        {
            "Churn_Prob": "{:.1%}",
            "CLV": "${:,.0f}",
            "Revenue_At_Risk": "${:,.0f}",
        }
    ).background_gradient(subset=["Revenue_At_Risk"], cmap="OrRd"),
    use_container_width=True,
)

st.markdown("---")

# ──────────────────────────────────────────────────────────
# 6. CUSTOMER PRIORITIZATION TABLE (full, sortable)
# ──────────────────────────────────────────────────────────
st.subheader("📋 Full Customer Prioritization Table")
st.caption("Use column headers to sort. Use sidebar filters to narrow down.")

st.dataframe(
    filtered[
        [
            "Loyalty Number",
            "Segment_Name",
            "Risk_Level",
            "Churn_Prob",
            "CLV",
            "Revenue_At_Risk",
            "Flights_6M",
            "Points_6M",
            "Months_Since_Last_Active",
            "Recommended_Action",
        ]
    ]
    .sort_values("Revenue_At_Risk", ascending=False)
    .style.format(
        {
            "Churn_Prob": "{:.1%}",
            "CLV": "${:,.0f}",
            "Revenue_At_Risk": "${:,.0f}",
            "Months_Since_Last_Active": "{:.1f}",
        }
    ),
    use_container_width=True,
    height=500,
)

# ──────────────────────────────────────────────────────────
# 7. DOWNLOAD
# ──────────────────────────────────────────────────────────
st.markdown("---")
csv_download = filtered.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download Filtered Customers as CSV",
    data=csv_download,
    file_name="filtered_customers.csv",
    mime="text/csv",
)

# ──────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Behavioral Intelligence Framework for Airline Loyalty Retention · "
    "Built with Streamlit · Data: final_customer_intelligence.csv"
)
