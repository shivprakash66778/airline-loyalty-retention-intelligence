# Behavioral Intelligence Framework for Airline Loyalty Retention

**Technical Report**

---

## 1. Executive Summary

Airline loyalty programs are designed to drive repeat business, but most still react to disengagement after the damage is done. This project builds an end-to-end behavioral intelligence system that identifies which loyalty members are likely to disengage, quantifies how much revenue is at stake, and recommends specific retention actions for each customer.

The solution covers four integrated capabilities: a churn prediction model that scores every member on their likelihood of going silent in the next six months, a customer segmentation framework that separates Champions from At-Risk Loyalists and Dormant Members, a revenue-at-risk engine that translates churn probability into dollar exposure, and a smart retention action engine that maps each customer to a specific campaign based on their segment, risk level, inactivity duration, and lifetime value.

The primary model (XGBoost) achieves a ROC-AUC of 0.977 and identifies churning members with 98% precision and 87% recall. Across 16,737 loyalty members analyzed, approximately $21.2 million in customer lifetime value is currently at risk, with Dormant Members alone accounting for $14.9 million of that exposure. The working prototype (Streamlit dashboard) allows a non-technical marketing manager to identify who needs attention, understand why, and know what to do — without reading a single line of code.

---

## 2. Problem Framing

### Why Cancellation Is Not Enough

The dataset includes formal cancellation records, but only about 12% of members have formally cancelled. If churn analysis relied solely on cancellation, the vast majority of disengaged customers would be invisible. A member who stops flying for 18 months but never cancels their account is functionally churned — they generate no revenue, but their loyalty card still sits in a drawer.

### Behavioral Churn Definition

We define churn behaviorally: a customer is labelled as churned at any given month if they show zero engagement in the following six months. Engagement is defined as either taking a flight or accumulating loyalty points. This captures the full spectrum of disengagement, not just the small fraction who formally opt out.

The six-month forward window was chosen because it balances sensitivity (shorter windows would flag normal seasonal gaps as churn) with timeliness (longer windows would delay detection past the point of intervention). This definition aligns with industry practice for airline loyalty programs, where seasonal travel patterns make shorter windows unreliable.

### Why This Matters

Inactive-but-enrolled members represent a hidden cost: they occupy CRM resources, dilute program metrics, and — most importantly — they often held significant value before going silent. Catching them early is the difference between a win-back and a write-off.

---

## 3. Data Cleaning and Preparation

The analysis uses three source files covering approximately 16,700 Canadian loyalty members from 2012 to 2018: Customer Loyalty History (demographics, loyalty tier, CLV, cancellation records), Customer Flight Activity (monthly time-series of flights, distance, points), and a Calendar dimension table.

**Salary anomalies.** Approximately 25% of salary records were missing and some contained negative values. Negative salaries were treated as data entry errors and set to missing. All missing salaries were then imputed using the median salary for each Gender × Education group, which preserves the demographic structure of the data without inventing precision that does not exist.

**Duplicate monthly records.** The flight activity file contained duplicate entries for the same customer in the same month. These were aggregated by summing flights, distance, and points, ensuring each customer has exactly one record per month.

**Monthly panel construction.** After cleaning, customer-level demographics were merged with the monthly flight panel to create a master dataset where each row represents one customer in one month. This panel structure is the foundation for rolling feature engineering and temporal train/test splitting.

All cleaning decisions are conservative: they remove noise without discarding information, and they do not introduce any forward-looking bias.

---

## 4. Feature Engineering

Every feature was designed to capture a specific behavioral signal that a marketing manager would recognize as meaningful. The features fall into several categories.

**Tenure and lifecycle.** Enrollment Age (in months) measures how long a member has been in the program. Longer tenure does not guarantee loyalty, but it provides context for interpreting activity levels.

**Recent activity windows.** Flights, Distance, Points Accumulated, and Points Redeemed are each computed over trailing 3-month and 6-month windows. These rolling windows capture the recency and intensity of engagement. A member who flew 10 times in the last 3 months is very different from one who flew 10 times two years ago.

**Flight Trend.** This compares 3-month activity against the 6-month average. A positive trend means the customer is accelerating; a negative trend signals deceleration — an early warning sign that precedes full disengagement.

**Months Since Last Active.** The number of months since a customer's most recent flight or points activity. This is the single most intuitive indicator of risk: the longer the silence, the harder the win-back.

**Redemption Ratio.** The proportion of accumulated points that a customer has redeemed. High redemption signals active program engagement; low redemption may indicate the customer does not value the program's rewards.

**CLV and CLV per Month.** Raw CLV is provided in the dataset. CLV per Month normalizes for tenure, revealing which customers generate value at a high rate versus those whose CLV is inflated simply by being enrolled for many years.

**Points Per Flight.** This measures earning efficiency and correlates with how deeply a customer engages with the loyalty program's earning mechanics (e.g., choosing airline partners, using co-branded cards).

**Loyalty Card Tier.** Encoded numerically. Higher-tier members have different behavioral baselines and expectations.

**Season.** Travel patterns vary significantly by quarter. Encoding the season helps the model distinguish genuine disengagement from normal seasonal dips.

**Never Active flag.** Some members enrolled but never flew. These are structurally different from dormant members who were once active.

---

## 5. Leakage Prevention and Validation

Data leakage — accidentally giving the model access to information it would not have in a real deployment — is the most dangerous failure mode in churn prediction. Our pipeline addresses this at three levels.

**Rolling features are shifted.** All trailing-window features (Flights_3M, Points_6M, etc.) are computed using a one-period shift. This means that a feature value at month t is calculated from data available through month t−1 only. The current month's activity is never used to predict the current month's churn label.

**Churn labels exclude the last six months.** For each customer, the final six monthly observations are assigned a null churn label (not 0 or 1). This is because there is insufficient future data to determine whether those months led to churn. Including them would introduce a survivorship bias where the model learns to associate end-of-dataset observations with retention.

**Time-based train/test split.** The model is trained on all data before January 2018 and tested on data from January 2018 onward. This temporal split simulates real-world deployment, where the model must predict the future using only the past. Random splitting would leak temporal patterns and inflate accuracy.

Together, these safeguards ensure that the model's reported performance is a realistic estimate of what it would achieve if deployed monthly in production.

---

## 6. Churn Model

### Model Comparison

Two models were trained and evaluated:

**Logistic Regression** serves as the interpretable baseline. It uses class balancing to handle the skewed churn distribution and provides coefficient-level insight into which features drive predictions. Its ROC-AUC is approximately 0.959, which is respectable but limited by its inability to capture non-linear interactions (e.g., the combined effect of high CLV and high inactivity).

**XGBoost** is the primary model. It achieves a ROC-AUC of 0.977, accuracy of 97.4%, churn precision of 98%, and churn recall of 87%. The precision figure means that when the model flags a customer as churning, it is correct 98% of the time — critical for avoiding wasted retention spend. The 87% recall means the model catches 87 out of every 100 actual churners, which is a strong detection rate given the conservative churn definition.

### Why XGBoost

XGBoost was selected because it handles non-linear feature interactions natively (e.g., a customer with high CLV and rising inactivity is qualitatively different from one with low CLV and rising inactivity), provides built-in feature importance rankings, does not require feature scaling, and consistently outperforms linear models on structured tabular data of this type.

### Probability, Not Binary

The model outputs a churn probability for each customer, not a binary yes/no. This is important because a customer with 35% churn probability and $20,000 CLV represents more revenue at risk than a customer with 90% churn probability and $1,000 CLV. Probability-based scoring enables the revenue-at-risk framework described in Section 8.

---

## 7. Segmentation and Customer Value

### Why CLV Alone Is Incomplete

The dataset provides a CLV figure for each customer, but CLV alone does not tell a marketing manager what to do. Two customers with identical CLV may require completely different interventions: one is a Champion who flies every month, the other is a Dormant Member who has not flown in a year. Segmentation adds the behavioral dimension that CLV misses.

### Segmentation Approach

Customer-level segmentation was performed using KMeans clustering on eight features: total flights, total distance, points accumulated, points redeemed, CLV, enrollment age, months since last active, and redemption ratio. All features were standardized before clustering.

Silhouette scores were computed for k = 2 through 7. The optimal k = 4 was selected based on a combination of silhouette score and business interpretability. One micro-cluster of only 9 customers was identified as noise and removed. The remaining three clusters were profiled and named based on their behavioral characteristics.

### The Three Segments

**Champions** (11,418 members): High flight frequency, low inactivity, active point earners and redeemers. These are the airline's core revenue engine. Average churn probability is just 1.9%. The priority is protection, not intervention.

**At-Risk Loyalists** (708 members): Historically valuable members showing clear signs of disengagement. Their average churn probability is 82.4%, the highest of any segment. They represent a concentrated pocket of recoverable value — if reached in time.

**Dormant Members** (4,602 members): Low or ceased activity over an extended period. Average churn probability is 39%. While individually they may have lower immediate recovery potential, their sheer number means they account for $14.9 million in total revenue at risk — the largest single exposure.

### Segment + Risk = Actionability

Neither segment alone nor risk level alone is sufficient for action planning. A Champion at Critical risk needs VIP intervention; a Dormant Member at Low risk needs a routine nudge. The combination of segment and risk level is what makes the retention engine operational.

---

## 8. Revenue at Risk

Revenue at Risk is computed as:

**Revenue_At_Risk = CLV × Churn_Prob**

This simple formula translates model output into a financial metric that leadership can act on. A customer with $15,000 CLV and 60% churn probability has $9,000 at risk — that is the expected value loss if no intervention occurs.

This metric serves three purposes. First, it prioritizes marketing spend: high-risk, high-value customers should receive investment before high-risk, low-value ones. Second, it aggregates to a portfolio-level number ($21.2 million total) that a CFO can use for budgeting. Third, it creates a common language between analytics, marketing, and finance teams.

Importantly, Revenue at Risk exposes a counterintuitive insight: Dormant Members, despite having a lower average churn probability than At-Risk Loyalists, account for $14.9 million in exposure because there are so many of them. Mass-market win-back campaigns for this segment may be more cost-effective than high-touch interventions for the smaller At-Risk Loyalist pool.

---

## 9. Smart Retention Action Engine

The recommendation engine maps each customer to a specific campaign using four inputs: Segment, Risk Level, Months Since Last Active, and CLV. The logic is rule-based and fully transparent — a marketing manager can trace exactly why any customer received a particular recommendation.

### Action Examples

**Champions + Critical Risk → Priority VIP Outreach.** Who: The airline's most valuable members showing sudden disengagement signals. When: Within 48 hours of risk score crossing the Critical threshold. Why: These customers have the highest recovery value and the shortest intervention window. What form: Direct call from a relationship manager, offer of tier status upgrade and dedicated account management.

**Champions + High Risk → VIP Retention Package.** Who: High-value members with elevated but not yet critical risk. When: Within the current campaign cycle (weekly). Why: Proactive intervention before the situation escalates. What form: 3X bonus miles on next two flights plus an exclusive airport lounge day pass.

**At-Risk Loyalists + High/Critical → Re-Engagement Campaign.** Who: Historically engaged members showing clear behavioral decline. When: Triggered by two consecutive months of declining flight trend. Why: These members have demonstrated they value the program — they need a reason to come back. What form: Double points on all flights for 60 days plus a co-branded credit card offer with waived annual fee.

**Dormant Members, inactive 12+ months → Win-Back Offer.** Who: Members who have been completely inactive for a year or more. When: Triggered by the 12-month inactivity threshold. Why: At this point, standard nudges are insufficient; a material incentive is required. What form: 50% fare discount coupon (single use) plus 10,000 bonus miles credited upon return flight.

**Dormant Members, inactive 6–12 months → Re-Activation.** Who: Members in the early-to-mid dormancy phase. When: At the 6-month mark. Why: Urgency framing works better before habits fully shift to a competitor. What form: Points expiry warning with 14-day urgency, plus option to transfer points to a partner program.

**Low Risk (any segment) → Standard Nurture.** Who: Engaged members with no immediate risk signals. When: Monthly cadence. Why: Maintaining awareness without over-investing. What form: Newsletter with tier progress update and seasonal travel inspiration.

The engine produces eight distinct action types, each with a clear who, when, why, and what form — meeting the problem statement's requirement that recommendations be specific enough for an operations team to execute immediately.

---

## 10. Business Recommendations

### Recommendation 1: Replace Mass Discounting with Revenue-at-Risk Campaigns

Instead of offering blanket promotions to all members, allocate retention budgets proportionally to Revenue at Risk. The top 2,340 High and Critical risk customers represent a disproportionate share of exposure. Focusing on this group reduces wasted spend on already-loyal members while increasing the probability of saving at-risk revenue.

**CFO benefit:** Marketing ROI improves because spend is directed where the expected recovery value is highest. Incentive costs are controlled because low-risk members receive low-cost nurture, not expensive promotions.

**CMO benefit:** Campaigns become personalized and segment-specific, improving member perception of the loyalty program. Response rates increase because offers are relevant to the customer's actual behavioral state.

### Recommendation 2: Launch a Tiered Retention Program by Segment and Risk

Design three distinct retention tracks:

Champions receive a premium experience track: exclusive access, recognition, and proactive relationship management. The goal is protection, not recovery.

At-Risk Loyalists receive a re-engagement track: time-limited incentives, status match offers, and personalized outreach. The goal is to reverse behavioral decline before it becomes permanent.

Dormant Members receive a win-back track: material incentives (fare discounts, bonus miles) with urgency framing (points expiry, limited-time offers). The goal is reactivation at a controlled cost.

Low-risk members across all segments receive a standard nurture track at minimal cost.

This tiered approach ensures that every dollar of retention spend is matched to the customer's actual need and recovery potential.

### Recommendation 3: Build a Monthly Churn Monitoring Dashboard

Deploy the Streamlit prototype (or its equivalent in the airline's BI stack) as a monthly operational tool. Track five metrics over time: total count of High/Critical risk members, total Revenue at Risk, segment migration (how many customers moved between segments month over month), campaign response rate by action type, and reactivation rate for Dormant Members.

Monthly monitoring converts the model from a one-time analysis into a living system. It allows the marketing team to detect deterioration early, measure the effectiveness of retention campaigns, and continuously refine the action engine based on observed outcomes.

---

## 11. Limitations and Next Steps

**No campaign response data.** The current system recommends actions but cannot measure their effectiveness. Once campaigns are launched, response and conversion data should be fed back into the model to enable uplift modeling — predicting not just who will churn, but who will respond to an intervention.

**Behavioral churn definition should be validated.** The six-month zero-engagement threshold is defensible but should be reviewed with business teams. Some customer segments (e.g., annual vacation travelers) may have naturally longer gaps. Business validation may refine the threshold or introduce segment-specific definitions.

**CLV is treated as given.** The CLV field is taken directly from the dataset. A more sophisticated approach would model CLV forward-looking using predicted future flight frequency, fare class mix, and ancillary revenue. This would improve Revenue at Risk accuracy.

**Future improvements.** The natural next steps include uplift modeling (targeting customers who would not have retained without intervention), A/B testing of retention actions, campaign cost optimization (balancing incentive cost against expected revenue recovery), survival analysis for time-to-churn estimation, and SHAP-based explainability for individual customer predictions.

---

## 12. Conclusion

This project transforms airline loyalty data from a static record of past behavior into an operational retention system. The behavioral churn definition captures disengagement that formal cancellation misses. The XGBoost model identifies at-risk members with high precision. The segmentation framework distinguishes Champions from At-Risk Loyalists and Dormant Members. The Revenue at Risk metric translates model output into a financial language that leadership can act on. And the smart retention engine maps every customer to a specific, executable action.

The Streamlit prototype closes the last-mile gap: a non-technical marketing manager can open the dashboard, identify who needs attention, understand why, and know exactly what to do — today.

---

*Report prepared as part of the "Unlocking Behavioral Intelligence in Airline Loyalty Programs" project.*
*Consulting & Analytics Club — IIT Guwahati Summer Projects '26.*
