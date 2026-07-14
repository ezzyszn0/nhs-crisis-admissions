# NHS Mental Health Crisis Admission Prediction

A machine learning project predicting mental health crisis/emergency admissions using NHS open data, combining clinical nursing experience with data science methods.

## Background
Built by a registered mental health nurse transitioning into NHS data science roles, this project applies statistical and ML modeling to publicly available NHS Digital datasets to forecast crisis admission trends.

## Status
🚧 In progress (v1 complete)

## Data Sources
- NHS Digital Mental Health Services Data Set (MHSDS) — Time Series for Selected Measures (April 2025 to March 2026)

## Tech Stack
- Python (pandas, scikit-learn, matplotlib)
- Jupyter notebooks

## Methodology
1. Loaded raw NHS MHSDS time series data (644,400 rows, 122 measures)
2. Filtered down to three crisis-relevant measures: Emergency Referrals to Crisis Care teams, Very Urgent Referrals to Crisis Care teams, and A&E liaison psychiatry referrals
3. Pivoted data into one row per provider per month
4. Cleaned suppressed values (NHS uses '*' to mask small counts under 5 for privacy)
5. Filtered to providers with at least 6 months of reliable reporting history
6. Engineered features: previous month's referral count, 3-month rolling average, and month of year
7. Identified and removed aggregate/non-provider rows ("England" national total, "UNKNOWN" placeholder) that were distorting model performance
8. Trained and compared Linear Regression vs Random Forest (with GridSearchCV tuning) using a time-based train/test split

## Key Findings

After testing both Linear Regression and Random Forest (with hyperparameter tuning via GridSearchCV), Linear Regression performed best. An initial version of the model included "England" (a national aggregate) mistakenly treated as an individual provider, which distorted results for both models; after identifying and removing this and other non-provider rows, performance improved substantially:

| Model | MAE | R² |
|---|---|---|
| Linear Regression (final, cleaned data) | 13.51 | 0.946 |
| Random Forest (cleaned data) | 14.81 | 0.899 |
| Linear Regression (before removing aggregate rows) | 22.75 | 0.876 |
| Random Forest (before removing aggregate rows, tuned) | 29.00 | 0.782 |

Recent referral history is by far the strongest predictor of next month's crisis referrals — last month's count alone accounts for the majority of the model's predictive power, with the 3-month average adding a modest stabilizing effect, and time of year playing only a small role.

## Model Performance Visualization

![Actual vs Predicted Emergency Referrals](reports/actual_vs_predicted.png)

The chart above shows the Linear Regression model's predictions closely tracking actual emergency crisis referrals across the held-out test set.

## Next Steps (v2)
- Incorporate the two remaining crisis measures (Very Urgent Referrals, A&E liaison referrals) as additional model features
- Explore Mental Health Act detention data as a complementary target
- Reorganize code into reusable modules under `src/`

## Experiment: Adding More Features (v2 exploration)

We tested whether adding two more crisis-related measures (Very Urgent Referrals, A&E liaison referrals from the same month) as additional predictors would improve the model. This required dropping providers who don't report all three measures, shrinking the usable dataset from 646 to 400 rows.

| Model | Rows Used | MAE | R² |
|---|---|---|---|
| Original (3 features: last month, 3-month average, month) | 646 | 13.51 | 0.946 |
| Expanded (5 features, adding Very Urgent + A&E liaison) | 400 | 13.17 | 0.872 |

While the expanded model's average error was marginally lower, its R² dropped meaningfully — likely because the smaller, filtered dataset made the model less reliable overall, not because the extra features were genuinely more predictive. We kept the original 3-feature model as the primary model, since it uses more data, generalizes better (higher R²), and is simpler to explain to a non-technical audience.

## Forward-Looking Predictions

Using the trained model, we generated genuine next-month forecasts for all providers with current, active reporting (i.e. excluding providers whose most recent report was more than one month behind the dataset's latest date). Predictions are saved in `reports/next_month_predictions.csv`.

**A note on timing:** NHS Digital publishes MHSDS data with an inherent reporting lag — this dataset's most recent month is March 2026, so our "next month" forecast is for April 2026, not the present date. This reflects a real, common constraint when working with official published NHS statistics rather than a limitation of the model itself. In an operational setting, this model would be re-run each month as fresh data is published, keeping forecasts as current as the underlying data allows.