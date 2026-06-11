# Executive Summary

## Behavioral Intelligence Framework for Airline Loyalty Retention

---

**Problem.** A mid-sized airline's loyalty program has approximately 16,700 members, but the marketing team lacks visibility into which members are silently disengaging, how much revenue is at stake, and what specific actions to take. Formal cancellation captures only a fraction of actual churn; the majority of disengaged members simply stop flying without ever opting out.

**Approach.** We built an end-to-end behavioral intelligence pipeline that defines churn as zero engagement (no flights, no points activity) over a forward-looking six-month window, engineers 18 behavioral features from flight activity and loyalty data with strict leakage prevention, trains an XGBoost churn prediction model (ROC-AUC: 0.977, precision: 98%, recall: 87%), segments customers into three actionable groups (Champions, At-Risk Loyalists, Dormant Members), computes Revenue at Risk for every member, and maps each customer to a specific retention action through a rule-based recommendation engine.

**Key Findings.**

- Approximately **$21.2 million** in customer lifetime value is currently at risk across the loyalty base.
- **2,340 members** (14% of the base) are classified as High or Critical risk.
- **Dormant Members** account for the largest single revenue exposure at **$14.9 million**, driven by volume rather than individual severity.
- **At-Risk Loyalists**, though only 708 members, have the highest average churn probability at **82.4%** — representing a concentrated, time-sensitive recovery opportunity.
- **Champions** are largely stable (1.9% avg churn probability) but the ~40 members at Critical risk warrant immediate VIP intervention.

**Recommended Strategy.** Replace mass discounting with Revenue-at-Risk-prioritized campaigns. Launch a tiered retention program: premium protection for Champions, time-limited re-engagement for At-Risk Loyalists, and incentive-driven win-back for Dormant Members. Deploy a monthly churn monitoring dashboard to track risk migration, campaign effectiveness, and reactivation rates over time.

**Expected Business Impact.** Targeted retention reduces wasted marketing spend on already-loyal members, increases recovery rates by matching interventions to customer-specific behavioral signals, and provides finance and marketing teams with a shared metric (Revenue at Risk) for budget allocation and performance tracking.

**What the Prototype Enables.** The Streamlit dashboard allows a non-technical marketing manager to identify the highest-priority customers, understand their risk profile and segment, see the recommended action, and download filtered lists for campaign execution — all without technical guidance.

---

*Consulting & Analytics Club — IIT Guwahati Summer Projects '26*
