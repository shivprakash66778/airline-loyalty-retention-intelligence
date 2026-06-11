# ✈️ Airline Loyalty — Retention Intelligence

**Behavioral Intelligence Framework for Airline Loyalty Programs**

A complete data science and business intelligence solution that identifies at-risk airline loyalty members, quantifies revenue exposure, and recommends specific retention actions — built for the Consulting & Analytics Club, IIT Guwahati Summer Projects '26.

---

## Problem Overview

Airline loyalty programs generate rich behavioral data, but most marketing teams still react to disengagement after the damage is done. Formal cancellation captures only ~12% of actual churn; the majority of disengaged members simply stop flying without opting out. This project builds a proactive system that catches disengagement early and converts model output into executable business actions.

## Business Objective

Help the marketing team of a mid-sized airline answer three questions:
1. **Who** is likely to disengage in the next six months?
2. **How much** revenue is at stake if they do?
3. **What** specific action should be taken for each customer?

## Dataset

Four integrated source files covering ~16,700 Canadian loyalty members from 2012–2018:
- **Customer Loyalty History** — Demographics, loyalty tier, CLV, cancellation records
- **Customer Flight Activity** — Monthly time-series of flights, distance, points accumulated/redeemed
- **Calendar** — Date dimension mapping months to quarters and seasons
- **Data Dictionary** — Full column definitions

## Methodology

### Churn Definition
Churn is defined **behaviorally**: a customer is labelled churned at month *t* if they show zero engagement (no flights, no points activity) in the six months following *t*. This captures silent disengagement that formal cancellation misses.

### Feature Engineering
18 features engineered from raw data, including:
- Rolling 3M/6M windows for flights, distance, points, redemptions (all shifted to prevent leakage)
- Flight Trend (acceleration/deceleration indicator)
- Months Since Last Active
- Redemption Ratio, CLV per Month, Points Per Flight
- Loyalty Card tier, Season, Never Active flag

### Leakage Prevention
- All rolling features use a 1-period shift (no current-month data in features)
- Last 6 months per customer excluded from churn labels (insufficient future window)
- Temporal train/test split: train on pre-2018, test on 2018+

### Modeling
| Model | ROC-AUC | Accuracy | Churn Precision | Churn Recall |
|-------|---------|----------|-----------------|--------------|
| Logistic Regression (baseline) | 0.959 | — | 0.58 | — |
| **XGBoost (primary)** | **0.977** | **97.4%** | **98%** | **87%** |

### Segmentation
Customer-level KMeans clustering (k=4, validated by silhouette score) produces three actionable segments:
- **Champions** (11,418) — High activity, low risk, protect at all costs
- **At-Risk Loyalists** (708) — Historically valuable, now disengaging, time-sensitive recovery
- **Dormant Members** (4,602) — Extended inactivity, high volume, largest revenue exposure

### Smart Retention Engine
Rule-based action mapping using Segment × Risk Level × Inactivity × CLV:
- 8 distinct action types, each specifying **who, when, why, and what form**
- From VIP outreach within 48 hours to standard monthly newsletters

## Prototype

A Streamlit dashboard designed for non-technical marketing managers:
- Executive KPI cards (customers, risk counts, revenue at risk, avg churn probability, avg CLV)
- Segment and risk level breakdowns with interactive Plotly charts
- Action center showing recommended actions and customer details
- Top 20 priority customer list
- Full sortable/filterable customer table
- CSV download for campaign execution

### How to Run

## How to Run

### Option 1: Open the Live Dashboard

You can directly use the deployed Streamlit app:

[Open Dashboard](https://czj3gjhrnzbzuwue9ypjrt.streamlit.app/)

### Option 2: Run Locally

```bash
# 1. Clone the repository
git clone <repo-url>
cd airline-loyalty-retention-intelligence

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run app/app.py

The app reads from `data/final_customer_intelligence.csv`. Run the command from the **project root**, not from inside the `app/` folder.

### Re-running the Raw Pipeline

The final customer intelligence CSV is already provided, so the dashboard can run directly.

To rerun the complete raw-data pipeline, place the original dataset files in `data/raw/`:

- `Customer Loyalty History.csv`
- `Customer Flight Activity.csv`
- `Calendar.csv`

Then run:

```bash
python src/Airline_Loyalty_final.py
```

The original pipeline was developed in Colab, but the submitted version uses the local `data/raw/` folder. Colab users may update `RAW_DATA_DIR` if they keep raw files in Google Drive.

**Micro-Segment Outliers:** 9 customers from the tiny KMeans micro-cluster are retained in the final output as `Micro-Segment Outliers` instead of being dropped or left blank.

## Folder Structure

```
airline-loyalty-retention-intelligence/
│
├── data/
│   ├── final_customer_intelligence.csv    # Final customer-level output (16,737 rows)
│   └── raw/
│       └── README.md                      # Place original source CSVs here to rerun pipeline
│
├── notebooks/
│   └── Airline_Loyality_Project.ipynb     # Exploratory notebook and initial modeling
│
├── src/
│   └── Airline_Loyalty_final.py           # Final improved Python pipeline
│
├── app/
│   └── app.py                             # Streamlit dashboard prototype
│
├── reports/
│   ├── Airline_Loyalty_Technical_Report.md # 6–8 page technical report
│   └── Executive_Summary.md               # 1-page executive summary
│
├── assets/
│   └── overview_charts.png                # Static visualization assets
│
├── requirements.txt
└── README.md
```

## Key Outputs

| Metric | Value |
|--------|-------|
| Total customers analyzed | 16,737 |
| High + Critical risk members | 2,340 (14%) |
| Total Revenue at Risk | $21.2M |
| Average churn probability | 15.5% |
| Average CLV | $7,989 |
| XGBoost ROC-AUC | 0.977 |

## Business Impact

- **$21.2M** in customer lifetime value identified as at-risk, enabling prioritized retention spend
- **Revenue-at-Risk prioritization** replaces mass discounting with targeted campaigns
- **Tiered retention program** matches intervention intensity to customer value and risk
- **Monthly monitoring dashboard** converts a one-time analysis into a living operational system

## Future Improvements

- Uplift modeling (predict who will respond to intervention, not just who will churn)
- A/B testing framework for retention actions
- Campaign cost optimization (balance incentive cost vs. expected recovery)
- Survival analysis for time-to-churn estimation
- SHAP explainability for individual predictions
- Integration with CRM/campaign management tools

---

*Built for the Consulting & Analytics Club, IIT Guwahati — Summer Projects '26*
